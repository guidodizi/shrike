# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pipeline helper class to create pipelines loading modules from a flexible manifest.
"""
import os
import json
import logging
import argparse
import re
import webbrowser
import uuid

try:
    from dataclasses import dataclass
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    from flatten_dict import flatten

    import azureml
    from azureml.core import Datastore
    from azureml.core import Experiment
    from azureml.core import Dataset
    from azureml.pipeline.core import PipelineRun
    from azure.ml.component._core._component_definition import (
        ComponentDefinition,
        ComponentType,
    )
except ImportError as error:
    raise ImportError(
        f"{error.msg}. Please install using `pip install shrike[pipeline]`."
    )

from shrike import __version__
from shrike.pipeline.aml_connect import azureml_connect, current_workspace
from shrike.pipeline.canary_helper import get_repo_info
from shrike.pipeline.module_helper import AMLModuleLoader, module_loader_config
from shrike.pipeline.pipeline_config import default_config_dict, HDI_DEFAULT_CONF
from shrike.pipeline.telemetry_utils import TelemetryLogger


class AMLPipelineHelper:
    """Helper class for building pipelines"""

    BUILT_PIPELINE = None  # the hydra run decorator doesn't allow for return, we're using this variable instead (hack)

    def __init__(self, config, module_loader=None):
        """Constructs the pipeline helper.

        Args:
            config (DictConfig): config for this object
            module_loader (AMLModuleLoader): which module loader to (re)use
        """
        self.config = config

        if module_loader is None:
            print(f"Creating instance of AMLModuleLoader for {self.__class__.__name__}")
            self.module_loader = AMLModuleLoader(self.config)
        else:
            self.module_loader = module_loader

    ######################
    ### CUSTOM METHODS ###
    ######################

    @classmethod
    def get_config_class(cls):
        """Returns a dataclass containing config for this pipeline"""
        pass

    @classmethod
    def required_subgraphs(cls):
        """Dependencies on other subgraphs
        Returns:
            dict[str, AMLPipelineHelper]: dictionary of subgraphs used for building this one.
                keys are whatever string you want for building your graph
                values should be classes inherinting from AMLPipelineHelper.
        """
        return {}

    @classmethod
    def required_modules(cls):
        """Dependencies on modules/components

        Returns:
            dict[str, dict]: manifest
        """
        return {}

    def build(self, config):
        """Builds a pipeline function for this pipeline.

        Args:
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            pipeline_function: the function to create your pipeline
        """
        raise NotImplementedError("You need to implement your build() method.")

    def pipeline_instance(self, pipeline_function, config):
        """Creates an instance of the pipeline using arguments.

        Args:
            pipeline_function (function): the pipeline function obtained from self.build()
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            pipeline: the instance constructed using build() function
        """
        raise NotImplementedError(
            "You need to implement your pipeline_instance() method."
        )

    def canary(self, args, experiment, pipeline_run):
        """Tests the output of the pipeline"""
        pass

    ##################################
    ### USER FACING HELPER METHODS ###
    ##################################

    def workspace(self):
        """Gets the current workspace"""
        return current_workspace()

    def component_load(self, component_key):
        """Loads one component from the manifest"""
        return self.module_loader.load_module(component_key, self.required_modules())

    def module_load(self, module_key):
        """Loads one module from the manifest"""
        return self.module_loader.load_module(module_key, self.required_modules())

    def subgraph_load(self, subgraph_key):
        """Loads one subgraph from the manifest"""
        subgraph_class = self.required_subgraphs()[subgraph_key]

        print(f"Building subgraph [{subgraph_key} as {subgraph_class.__name__}]...")
        # NOTE: below creates subgraph with same pipeline_config
        subgraph_instance = subgraph_class(
            config=self.config, module_loader=self.module_loader
        )
        # subgraph_instance.setup(self.pipeline_config)
        return subgraph_instance.build(self.config)

    def dataset_load(self, name, version="latest"):
        """Loads a dataset by either id or name.

        Args:
            name (str): name or uuid of dataset to load
            version (str): if loading by name, used to specify version (default "latest")

        NOTE: in AzureML SDK there are 2 different methods for loading dataset
        one for id, one for name. This method just wraps them up in one."""
        # test if given name is a uuid
        try:
            parsed_uuid = uuid.UUID(name)
            print(f"Getting a dataset handle [id={name}]...")
            return Dataset.get_by_id(self.workspace(), id=name)
        except ValueError:
            print(f"Getting a dataset handle [name={name} version={version}]...")
            return Dataset.get_by_name(self.workspace(), name=name, version=version)

    @staticmethod
    def validate_experiment_name(name):
        """
        Check whether the experiment name is valid. It's required that
        experiment names must be between 1 to 250 characters, start with
        letters or numbers. Valid characters are letters, numbers, "_",
        and the "-" character.
        """
        if len(name) < 1 or len(name) > 250:
            raise ValueError("Experiment names must be between 1 to 250 characters!")
        if not re.match("^[a-zA-Z0-9]$", name[0]):
            raise ValueError("Experiment names must start with letters or numbers!")
        if not re.match("^[a-zA-Z0-9_-]*$", name):
            raise ValueError(
                "Valiad experiment names must only contain letters, numbers, underscore and dash!"
            )
        return True

    #######################
    ### HELPER BACKEND  ###
    #######################

    @classmethod
    def _default_config(cls):
        """Builds the default config for the pipeline class"""
        config_store = ConfigStore.instance()

        config_dict = default_config_dict()
        cls._build_config(config_dict)

        config_store.store(name="default", node=config_dict)
        return OmegaConf.structured(config_dict)

    @classmethod
    def _build_config(cls, config_dict, modules_config=None):
        """Builds the entire configuration object for this graph and all subgraphs."""
        self_config_class = cls.get_config_class()
        if self_config_class:
            config_dict[self_config_class.__name__] = self_config_class

        for subgraph_key, subgraph_class in cls.required_subgraphs().items():
            subgraph_class._build_config(config_dict)

    def _set_all_inputs_to(self, module_instance, input_mode):
        """Sets all module inputs to a given intput mode"""
        input_names = [
            a
            for a in dir(module_instance.inputs)
            if not a.startswith("_")
            and not callable(getattr(module_instance.inputs, a))
        ]
        for input_key in input_names:
            input_instance = getattr(module_instance.inputs, input_key)
            input_instance.configure(mode=input_mode)
            print(f"-- configured input {input_key} to use mode {input_mode}")

    def _set_all_outputs_to(self, module_instance, output_mode, compliant=True):
        """Sets all module outputs to a given output mode"""
        output_names = [
            a
            for a in dir(module_instance.outputs)
            if not a.startswith("_")
            and not callable(getattr(module_instance.outputs, a))
        ]
        datastore_name = (
            self.config.compute.compliant_datastore
            if compliant
            else self.config.compute.noncompliant_datastore
        )
        for output_key in output_names:
            output_instance = getattr(module_instance.outputs, output_key)
            if output_mode is None:
                output_instance.configure(
                    datastore=Datastore(
                        current_workspace(),
                        name=datastore_name,
                    )  # datastore name for storing outputs
                )
            else:
                output_instance.configure(
                    datastore=Datastore(
                        current_workspace(),
                        name=datastore_name,
                    ),  # datastore name for storing outputs
                    output_mode=output_mode,
                )
            print(
                f"-- configured output {output_key} to use mode {output_mode} and datastore {datastore_name}"
            )

    def _apply_windows_runsettings(
        self,
        module_name,
        module_instance,
        mpi=False,
        target=None,
        node_count=None,
        process_count_per_node=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a windows module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            mpi (bool): is job mpi ?
            target (str): force target compute over hydra conf
            node_count (int): force node_count over hydra conf
            process_count_per_node (int): force process_count_per_node over hydra conf
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            custom_runtime_arguments (dict): any additional custom args
        """
        if self.module_loader.is_local(module_name):
            target = (
                target
                if target is not None
                else self.config.compute.windows_cpu_dc_target
            )
        else:
            target = (
                target
                if target is not None
                else self.config.compute.windows_cpu_prod_target
            )

        print(
            f"Using windows compute target {target} to run {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        module_instance.runsettings.configure(target=target, **custom_runtime_arguments)
        if mpi:
            node_count = node_count if node_count is not None else 1
            process_count_per_node = (
                process_count_per_node if process_count_per_node is not None else 1
            )
            print(
                f"Using mpi with node_count={node_count} process_count_per_node={process_count_per_node}"
            )
            module_instance.runsettings.resource_layout.configure(
                node_count=node_count,
                process_count_per_node=process_count_per_node,
            )

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        else:
            self._set_all_inputs_to(
                module_instance, self.config.compute.windows_input_mode
            )

        if output_mode:
            self._set_all_outputs_to(module_instance, output_mode)
        else:
            self._set_all_outputs_to(
                module_instance, self.config.compute.windows_output_mode
            )

    def _apply_hdi_runsettings(
        self,
        module_name,
        module_instance,
        target=None,
        driver_cores=None,
        driver_memory=None,
        executor_memory=None,
        executor_cores=None,
        number_executors=None,
        conf=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a HDI module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            target (str): force target compute over hydra conf
            driver_cores (int): force driver_cores over hydra conf
            driver_memory (str): force driver_memory over hydra conf
            executor_memory (int): force executor_memory over hydra conf
            executor_cores (int): force executor_cores over hydra conf
            number_executors (int): force number_executors over hydra conf
            conf (str): force conf over hydra conf
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            custom_runtime_arguments (dict): any additional custom args
        """
        print(
            f"Using HDI compute target {self.config.compute.hdi_prod_target} to run {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        merged_conf = json.loads(HDI_DEFAULT_CONF)
        new_conf = (
            self.config.compute.hdi_conf if "hdi_conf" in self.config.compute else None
        )
        if conf is not None:
            new_conf = conf
        if new_conf is not None:
            if isinstance(new_conf, str):
                new_conf = json.loads(new_conf)
            elif isinstance(new_conf, DictConfig):
                new_conf = flatten(dict(new_conf), reducer="dot")
            else:
                raise ValueError(
                    "computed.hdi_conf is not a valid json string or a single tested configuration."
                )
            merged_conf.update(new_conf)

        module_instance.runsettings.configure(
            target=target
            if target is not None
            else self.config.compute.hdi_prod_target,
        )

        module_instance.runsettings.hdinsight.configure(
            driver_memory=driver_memory
            if driver_memory is not None
            else self.config.compute.hdi_driver_memory,
            driver_cores=driver_cores
            if driver_cores is not None
            else self.config.compute.hdi_driver_cores,
            executor_memory=executor_memory
            if executor_memory is not None
            else self.config.compute.hdi_executor_memory,
            executor_cores=executor_cores
            if executor_cores is not None
            else self.config.compute.hdi_executor_cores,
            number_executors=number_executors
            if number_executors is not None
            else self.config.compute.hdi_number_executors,
            conf=merged_conf,
            **custom_runtime_arguments,
        )

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        self._set_all_outputs_to(module_instance, output_mode)

    def _apply_parallel_runsettings(
        self,
        module_name,
        module_instance,
        windows=False,
        gpu=False,
        target=None,
        node_count=None,
        process_count_per_node=None,
        mini_batch_size=None,
        run_invocation_timeout=None,
        run_max_try=None,
        error_threshold=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a ParallelRunStep linux module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            windows (bool): is the module using Windows compute?
            gpu (bool): is the module using GPU compute?
            target (str): force target compute over hydra conf
            node_count (int): force node_count over hydra conf
            process_count_per_node (int): force process_count_per_node over hydra conf
            mini_batch_size (int): force mini_batch_size over hydra conf
            run_invocation_timeout (int): force run_invocation_timeout over hydra conf
            run_max_try (int): force run_max_try over hydra conf
            error_threshold (int): The number of file failures for the input FileDataset that should be ignored during processing.
                If the error count goes above this value, then the job will be aborted.
                Error threshold is for the entire input and not for individual mini-batches sent to run() method.
                The range is [-1, int.max]. -1 indicates ignoring all failures during processing.
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            custom_runtime_arguments (dict): any additional custom args
        """
        if self.module_loader.is_local(module_name):
            if windows:
                if gpu:
                    raise ValueError(
                        "A GPU compute target with Windows OS is not available yet!"
                    )
                else:
                    _target = self.config.compute.windows_cpu_dc_target
            else:
                if gpu:
                    _target = self.config.compute.linux_gpu_dc_target
                else:
                    _target = self.config.compute.linux_cpu_dc_target
        else:
            if windows:
                if gpu:
                    raise ValueError(
                        "A GPU compute target with Windows OS is not available yet!"
                    )
                else:
                    _target = self.config.compute.windows_cpu_prod_target
            else:
                if gpu:
                    _target = self.config.compute.linux_gpu_prod_target
                else:
                    _target = self.config.compute.linux_cpu_prod_target

        target = target if target is not None else _target

        print(
            f"Using parallelrunstep compute target {target} to run {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        module_instance.runsettings.configure(target=target)

        module_instance.runsettings.parallel.configure(
            node_count=node_count
            if node_count is not None
            else self.config.compute.parallel_node_count,
            process_count_per_node=process_count_per_node
            if process_count_per_node is not None
            else (
                self.config.compute.parallel_process_count_per_node
                if "parallel_process_count_per_node" in self.config.compute
                else None
            ),
            mini_batch_size=str(
                mini_batch_size
                if mini_batch_size is not None
                else self.config.compute.parallel_mini_batch_size
            ),
            run_invocation_timeout=run_invocation_timeout
            if run_invocation_timeout is not None
            else self.config.compute.parallel_run_invocation_timeout,
            run_max_try=run_max_try
            if run_max_try is not None
            else self.config.compute.parallel_run_max_try,
            error_threshold=error_threshold
            if error_threshold is not None
            else self.config.compute.parallel_error_threshold,
            **custom_runtime_arguments,
        )

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        if output_mode:
            self._set_all_outputs_to(module_instance, output_mode)

    def _apply_linux_runsettings(
        self,
        module_name,
        module_instance,
        mpi=False,
        gpu=False,
        target=None,
        node_count=None,
        process_count_per_node=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for linux module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            mpi (bool): is the job mpi?
            gpu (bool): is the job using GPU?
            target (str): force target compute over hydra conf
            node_count (int): force node_count over hydra conf
            process_count_per_node (int): force process_count_per_node over hydra conf
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            custom_runtime_arguments (dict): any additional custom args
        """
        if self.module_loader.is_local(module_name) and gpu:
            target = (
                target
                if target is not None
                else self.config.compute.linux_gpu_dc_target
            )
            print(
                f"Using target {target} for local code GPU module {module_name} from pipeline class {self.__class__.__name__}"
            )
        elif not self.module_loader.is_local(module_name) and gpu:
            target = (
                target
                if target is not None
                else self.config.compute.linux_gpu_prod_target
            )
            print(
                f"Using target {target} for registered GPU module {module_name} from pipeline class {self.__class__.__name__}"
            )
        elif self.module_loader.is_local(module_name) and not gpu:
            target = (
                target
                if target is not None
                else self.config.compute.linux_cpu_dc_target
            )
            print(
                f"Using target {target} for local CPU module {module_name} from pipeline class {self.__class__.__name__}"
            )
        elif not self.module_loader.is_local(module_name) and not gpu:
            target = (
                target
                if target is not None
                else self.config.compute.linux_cpu_prod_target
            )
            print(
                f"Using target {target} for registered CPU module {module_name} from pipeline class {self.__class__.__name__}"
            )

        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        module_instance.runsettings.configure(target=target, **custom_runtime_arguments)

        if mpi:
            node_count = node_count if node_count is not None else 1
            process_count_per_node = (
                process_count_per_node if process_count_per_node is not None else 1
            )
            print(
                f"Using mpi with node_count={node_count} process_count_per_node={process_count_per_node}"
            )
            module_instance.runsettings.resource_layout.configure(
                node_count=node_count,
                process_count_per_node=process_count_per_node,
            )

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        else:
            self._set_all_inputs_to(
                module_instance, self.config.compute.linux_input_mode
            )

        if output_mode:
            self._set_all_outputs_to(module_instance, output_mode)
        else:
            self._set_all_outputs_to(
                module_instance, self.config.compute.linux_output_mode
            )

    def _apply_scope_runsettings(
        self,
        module_name,
        module_instance,
        input_mode=None,
        output_mode=None,
        scope_param=None,
        custom_job_name_suffix=None,
        adla_account_name=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a scope module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            scope_param (str): Parameters to pass to scope e.g. Nebula parameters, VC allocation parameters etc.
            custom_job_name_suffix (str): Optional parameter defining custom string to append to job name
            adla_account_name (str): The name of the Cosmos-migrated Azure Data Lake Analytics account to submit scope job
            custom_runtime_arguments (dict): any additional custom args
        """
        print(
            f"Using scope compute target to run {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        self._set_all_outputs_to(module_instance, output_mode, compliant=False)

        module_instance.runsettings.scope.configure(
            adla_account_name=adla_account_name,
            scope_param=scope_param,
            custom_job_name_suffix=custom_job_name_suffix,
        )

    def _apply_datatransfer_runsettings(
        self,
        module_name,
        module_instance,
        target=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a HDI module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            target (str): force target compute over hydra conf
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf
            custom_runtime_arguments (dict): any additional custom args
        """
        print(
            f"Using datatransfer compute target {self.config.compute.datatransfer_target} to run {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        module_instance.runsettings.configure(
            target=target
            if target is not None
            else self.config.compute.datatransfer_target,
        )

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        self._set_all_outputs_to(module_instance, output_mode)

    def _check_module_runsettings_consistency(self, module_key, module_instance):
        """Verifies if entry at module_key matches the module instance description"""
        module_manifest_entry, _ = self.module_loader.get_module_manifest_entry(
            module_key, modules_manifest=self.required_modules()
        )

        if "name" in module_manifest_entry:
            if module_manifest_entry["name"] == module_instance.name:
                return
            if "namespace" in module_manifest_entry:
                module_entry_name = (
                    module_manifest_entry["namespace"]
                    + "://"
                    + module_manifest_entry["name"]
                )
                if module_entry_name == module_instance.name:
                    return
            raise Exception(
                f"During build() of graph class {self.__class__.__name__}, call to self.apply_recommended_runsettings() is wrong: key used as first argument ('{module_key}') maps to a module reference {module_manifest_entry} which name is different from the module instance provided as 2nd argument (name={module_instance.name}), did you use the wrong module key as first argument?"
            )

    def _get_component_name_from_instance(self, component_instance):
        component_manifest_list = self.config.modules.manifest
        component_name = component_instance.name
        for component_manifest_entry in component_manifest_list:
            try:
                if component_manifest_entry["name"] == component_name:
                    return component_manifest_entry["key"] or component_name
                if "namespace" in component_manifest_entry:
                    component_entry_name = (
                        component_manifest_entry["namespace"]
                        + "://"
                        + component_manifest_entry["name"]
                    )
                    if component_entry_name == component_name:
                        return component_manifest_entry["key"] or component_name
            except:
                pass
        raise ValueError(
            f"Could not find component matching {component_name}. Please check your spelling."
        )

    def apply_smart_runsettings(
        self,
        component_instance,
        gpu=False,  # can't autodetect that
        hdi="auto",
        windows="auto",
        parallel="auto",
        mpi="auto",
        scope="auto",
        datatransfer="auto",
        **custom_runtime_arguments,
    ):
        """Applies regular settings for a given component.

        Args:
            component_instance (Component): the AML component we need to add settings to
            gpu (bool): is the component using GPU?
            hdi (bool): is the component using HDI/Spark?
            windows (bool): is the component using Windows compute?
            parallel (bool): is the component using ParallelRunStep?
            mpi (bool): is the component using Mpi?
            custom_runtime_arguments (dict): any additional custom args
        """
        # infer component_name
        component_name = self._get_component_name_from_instance(component_instance)
        self.apply_recommended_runsettings(
            component_name,
            component_instance,
            gpu,
            hdi,
            windows,
            parallel,
            mpi,
            scope,
            datatransfer,
            **custom_runtime_arguments,
        )

    def apply_recommended_runsettings(
        self,
        module_name,
        module_instance,
        gpu=False,  # can't autodetect that
        hdi="auto",
        windows="auto",
        parallel="auto",
        mpi="auto",
        scope="auto",
        datatransfer="auto",
        **custom_runtime_arguments,
    ):
        """Applies regular settings for a given module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            gpu (bool): is the module using GPU?
            hdi (bool): is the module using HDI/Spark?
            windows (bool): is the module using Windows compute?
            parallel (bool): is the module using ParallelRunStep?
            mpi (bool): is the module using Mpi?
            custom_runtime_arguments (dict): any additional custom args
        """
        # verifies if module_name corresponds to module_instance
        self._check_module_runsettings_consistency(module_name, module_instance)

        # Auto detect runsettings
        if hdi == "auto":
            hdi = str(module_instance.type) == "HDInsightComponent"
            if hdi:
                print(f"Module {module_name} detected as HDI: {hdi}")

        if parallel == "auto":
            parallel = str(module_instance.type) == "ParallelComponent"
            if parallel:
                print(f"Module {module_name} detected as PARALLEL: {parallel}")

        if mpi == "auto":
            mpi = str(module_instance.type) == "DistributedComponent"
            if mpi:
                print(f"Module {module_name} detected as MPI: {mpi}")

        if scope == "auto":
            scope = str(module_instance.type) == "ScopeComponent"
            if scope:
                print(f"Module {module_name} detected as SCOPE: {scope}")

        if windows == "auto":
            if (
                str(module_instance.type) == "HDInsightComponent"
                or str(module_instance.type) == "ScopeComponent"
                or str(module_instance.type) == "DataTransferComponent"
            ):
                # HDI/scope/datatransfer modules might not have that environment object
                windows = False
            else:
                windows = (
                    module_instance._definition.environment.os.lower() == "windows"
                )
                if windows:
                    print(f"Module {module_name} detected as WINDOWS: {windows}")

        if datatransfer == "auto":
            datatransfer = str(module_instance.type) == "DataTransferComponent"
            if datatransfer:
                print(f"Module {module_name} detected as DATATRANSFER: {datatransfer}")

        if parallel:
            self._apply_parallel_runsettings(
                module_name,
                module_instance,
                windows=windows,
                gpu=gpu,
                **custom_runtime_arguments,
            )
            return

        if windows:
            # no detonation chamber, we an't use "use_local" here
            self._apply_windows_runsettings(
                module_name, module_instance, mpi=mpi, **custom_runtime_arguments
            )
            return

        if hdi:
            # no detonation chamber, we an't use "use_local" here
            self._apply_hdi_runsettings(
                module_name, module_instance, **custom_runtime_arguments
            )
            return

        if scope:
            self._apply_scope_runsettings(
                module_name, module_instance, **custom_runtime_arguments
            )
            return

        if datatransfer:
            self._apply_datatransfer_runsettings(
                module_name, module_instance, **custom_runtime_arguments
            )
            return

        self._apply_linux_runsettings(
            module_name, module_instance, mpi=mpi, gpu=gpu, **custom_runtime_arguments
        )

    def _parse_pipeline_tags(self):
        """Parse the tags specified in the pipeline yaml"""
        pipeline_tags = {}
        if self.config.run.tags:
            if isinstance(self.config.run.tags, str):
                try:
                    # json.load the tags string in the config
                    pipeline_tags = json.loads(self.config.run.tags)
                except ValueError:
                    print(
                        f"The pipeline tags {self.config.run.tags} is not a valid json-style string."
                    )
            elif isinstance(self.config.run.tags, DictConfig):
                pipeline_tags.update(self.config.run.tags)
            else:
                print(
                    f"The pipeline tags {self.config.run.tags} is not a valid DictConfig or json-style string."
                )
        return pipeline_tags

    ################
    ### MAIN/RUN ###
    ################

    def connect(self):
        """Connect to the AML workspace using internal config"""
        return azureml_connect(
            aml_subscription_id=self.config.aml.subscription_id,
            aml_resource_group=self.config.aml.resource_group,
            aml_workspace_name=self.config.aml.workspace_name,
            aml_auth=self.config.aml.auth,
            aml_tenant=self.config.aml.tenant,
            aml_force=self.config.aml.force,
        )  # NOTE: this also stores aml workspace in internal global variable

    def run(self):
        """Run pipeline using arguments"""
        # Log the telemetry information in the Azure Application Insights
        telemetry_logger = TelemetryLogger(
            enable_telemetry=not self.config.run.disable_telemetry
        )
        telemetry_logger.log_trace(
            message=f"shrike.pipeline=={__version__}",
            properties={"custom_dimensions": {"configuration": str(self.config)}},
        )

        # Check whether the experiment name is valid
        self.validate_experiment_name(self.config.run.experiment_name)

        repository_info = get_repo_info()
        print(f"Running from repository: {repository_info}")

        print("azureml.core.VERSION = {}".format(azureml.core.VERSION))
        self.connect()

        if self.config.run.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if self.config.run.resume:
            if not self.config.run.pipeline_run_id:
                raise Exception(
                    "To be able to use --resume you need to provide both --experiment-name and --run-id."
                )

            print(f"Resuming Experiment {self.config.run.experiment_name}...")
            experiment = Experiment(
                current_workspace(), self.config.run.experiment_name
            )
            print(f"Resuming PipelineRun {self.config.run.pipeline_run_id}...")
            # pipeline_run is of the class "azureml.pipeline.core.PipelineRun"
            pipeline_run = PipelineRun(experiment, self.config.run.pipeline_run_id)
        else:
            print(f"Building Pipeline [{self.__class__.__name__}]...")
            pipeline_function = self.build(self.config)

            print("Creating Pipeline Instance...")
            pipeline = self.pipeline_instance(pipeline_function, self.config)

            print("Validating...")
            pipeline.validate()

            if self.config.run.export:
                print(f"Exporting to {self.config.run.export}...")
                with open(self.config.run.export, "w") as export_file:
                    export_file.write(pipeline._get_graph_json())

            if self.config.run.submit:
                pipeline_tags = self._parse_pipeline_tags()
                pipeline_tags.update(repository_info)
                print(f"Submitting Experiment... [tags={pipeline_tags}]")

                # pipeline_run is of the class "azure.ml.component.run", which
                # is different from "azureml.pipeline.core.PipelineRun"
                pipeline_run = pipeline.submit(
                    experiment_name=self.config.run.experiment_name,
                    tags=pipeline_tags,
                    default_compute_target=self.config.compute.default_compute_target,
                    regenerate_outputs=self.config.run.regenerate_outputs,
                    continue_on_step_failure=self.config.run.continue_on_failure,
                )

                # Forece pipeline_run to be of the class "azureml.pipeline.core.PipelineRun"
                pipeline_run = PipelineRun(
                    experiment=pipeline_run._experiment,
                    run_id=pipeline_run._id,
                )

            else:
                print(
                    "Exiting now, if you want to submit please override run.submit=True"
                )
                self.__class__.BUILT_PIPELINE = (
                    pipeline  # return so we can have some unit tests done
                )
                return

        # launch the pipeline execution
        print(f"Pipeline Run Id: {pipeline_run.id}")
        print(
            f"""
#################################
#################################
#################################

Follow link below to access your pipeline run directly:
-------------------------------------------------------

{pipeline_run.get_portal_url()}

#################################
#################################
#################################
        """
        )

        if self.config.run.canary:
            print(
                "*** CANARY MODE ***\n----------------------------------------------------------"
            )
            pipeline_run.wait_for_completion(show_output=True)

            # azureml.pipeline.core.PipelineRun.get_status(): ["Running", "Finished", "Failed"]
            # azureml.core.run.get_status(): ["Running", "Completed", "Failed"]
            if pipeline_run.get_status() in ["Finished", "Completed"]:
                print("*** PIPELINE FINISHED, TESTING WITH canary() METHOD ***")
                self.canary(self.config, pipeline_run.experiment, pipeline_run)
                print("OK")
            elif pipeline_run.get_status() == "Failed":
                print("*** PIPELINE FAILED ***")
                raise Exception("Pipeline failed.")
            else:
                print("*** PIPELINE STATUS {} UNKNOWN ***")
                raise Exception("Pipeline status is unknown.")

        else:
            if not self.config.run.silent:
                webbrowser.open(url=pipeline_run.get_portal_url())

            # This will wait for the completion of the pipeline execution
            # and show the full logs in the meantime
            if self.config.run.resume or self.config.run.wait:
                print(
                    "Below are the raw debug logs from your pipeline execution:\n----------------------------------------------------------"
                )
                pipeline_run.wait_for_completion(show_output=True)

    @classmethod
    def main(cls):
        """Pipeline helper main function, parses arguments and run pipeline."""
        config_dict = cls._default_config()

        @hydra.main(config_name="default")
        def hydra_run(cfg: DictConfig):
            # merge cli config with default config
            cfg = OmegaConf.merge(config_dict, cfg)

            print("*** CONFIGURATION ***")
            print(OmegaConf.to_yaml(cfg))

            # create class instance
            main_instance = cls(cfg)

            # run
            main_instance.run()

        hydra_run()

        return cls.BUILT_PIPELINE  # return so we can have some unit tests done
