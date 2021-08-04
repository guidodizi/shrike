# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pipeline helper class to create pipelines loading modules from a flexible manifest.
"""
import os
import json
import logging
import argparse
from posixpath import splitext
import re
import webbrowser
import uuid
import shutil
import yaml
import jsonpath_ng
import sys

try:
    from dataclasses import dataclass
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    from flatten_dict import flatten

    import azureml
    from azureml.core import Datastore
    from azureml.core import Experiment
    from azureml.core import Dataset
    from azureml.pipeline.core import PipelineRun
    from azure.ml.component.component import Input, Output
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
                "Valid experiment names must only contain letters, numbers, underscore and dash!"
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
            if isinstance(getattr(module_instance.inputs, a), Input)
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
            if isinstance(getattr(module_instance.outputs, a), Output)
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
        compliant=True,
        target=None,
        input_mode=None,
        output_mode=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a HDI module.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            compliant (bool): destination datastore, `compliant_datastore` if True, else `noncompliant_datastore`
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
        self._set_all_outputs_to(module_instance, output_mode, compliant)

    def _apply_sweep_runsettings(
        self,
        module_name,
        module_instance,
        windows=False,
        gpu=False,
        target=None,
        input_mode=None,
        output_mode=None,
        algorithm=None,
        primary_metric=None,
        goal=None,
        policy_type=None,
        evaluation_interval=None,
        delay_evaluation=None,
        slack_factor=None,
        slack_amount=None,
        truncation_percentage=None,
        max_total_trials=None,
        max_concurrent_trials=None,
        timeout_minutes=None,
        **custom_runtime_arguments,
    ):
        """Applies settings for a sweep component.

        Args:
            module_name (str): name of the module from the module manifest (required_modules() method)
            module_instance (Module): the AML module we need to add settings to
            target (str): force target compute over hydra conf
            input_mode (str): force input_mode over hydra conf
            output_mode (str): force output_mode over hydra conf

            # For below sweep specific parameters configurations, see below doc link for more info:
            # https://componentsdk.azurewebsites.net/components/sweep_component.html#set-runsettings
            algorithm (str): sweep sampling method
            primary_metric (str): the primary metric of the hyperparameter tuning to optimize
            goal (str): whether the primary metric will be maximize or minimize when evaluating the trials
            policy_type (str): sweep early termination policy type
            evaluation_interval (int): frequency of applying the policy
            delay_evaluation (int): delays the first policy evaluation for a specified number of intervals
            slack_factor (float): the slack allowed with respect to the best performing training run, as a ratio
            slack_amount (float): the slack allowed with respect to the best performing training run, as an absolute ampunt. You should only specify either slack_factor or slack_amount, but not both.
            truncation_percentage (int): the percentage of lowest performing runs to terminate at each evaluation interval. An integer value between 1 and 99.
            max_total_trials (int): maximum number of trial runs. Must be an integer between 1 and 1000.
            max_concurrent_trials (int): maximum number of runs that can run concurrently. If not specified, all runs launch in parallel. If specified, must be an integer between 1 and 100.
            timeout_minutes (int): maximum duration, in minutes, of the hyperparameter tuning experiment. Runs after this duration are canceled.

            custom_runtime_arguments (dict): any additional custom args
        """
        if target is not None:
            target = target
        elif self.module_loader.is_local(module_name):
            if windows:
                if gpu:
                    raise ValueError(
                        "A GPU compute target with Windows OS is not available yet!"
                    )
                else:
                    target = self.config.compute.windows_cpu_dc_target
            else:
                if gpu:
                    target = self.config.compute.linux_gpu_dc_target
                else:
                    target = self.config.compute.linux_cpu_dc_target
        else:
            if windows:
                if gpu:
                    raise ValueError(
                        "A GPU compute target with Windows OS is not available yet!"
                    )
                else:
                    target = self.config.compute.windows_cpu_prod_target
            else:
                if gpu:
                    target = self.config.compute.linux_gpu_prod_target
                else:
                    target = self.config.compute.linux_cpu_prod_target
        print(
            f"Using target {target} to run sweep component {module_name} from pipeline class {self.__class__.__name__}"
        )
        if custom_runtime_arguments:
            print(f"Adding custom runtime arguments {custom_runtime_arguments}")

        if input_mode:
            self._set_all_inputs_to(module_instance, input_mode)
        if output_mode:
            self._set_all_outputs_to(module_instance, output_mode)

        if algorithm:
            module_instance.runsettings.sweep.algorithm = algorithm
        if primary_metric:
            module_instance.runsettings.sweep.objective.configure(
                primary_metric=primary_metric
            )
        if goal:
            module_instance.runsettings.sweep.objective.configure(goal=goal)
        if policy_type:
            module_instance.runsettings.sweep.early_termination.configure(
                policy_type=policy_type
            )
        if evaluation_interval:
            module_instance.runsettings.sweep.early_termination.configure(
                evaluation_interval=evaluation_interval
            )
        if delay_evaluation:
            module_instance.runsettings.sweep.early_termination.configure(
                delay_evaluation=delay_evaluation
            )
        if slack_factor:
            module_instance.runsettings.sweep.early_termination.configure(
                slack_factor=slack_factor
            )
        if slack_amount:
            module_instance.runsettings.sweep.early_termination.configure(
                slack_amount=slack_amount
            )
        if truncation_percentage:
            module_instance.runsettings.sweep.early_termination.configure(
                truncation_percentage=truncation_percentage
            )
        if max_total_trials:
            module_instance.runsettings.sweep.limits.configure(
                max_total_trials=max_total_trials
            )
        if max_concurrent_trials:
            module_instance.runsettings.sweep.limits.configure(
                max_concurrent_trials=max_concurrent_trials
            )
        if timeout_minutes:
            module_instance.runsettings.sweep.limits.configure(
                timeout_minutes=timeout_minutes
            )

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
        sweep="auto",
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
            scope (bool): is the component using scope?
            datatransfer (bool): is the component using datatransfer?
            sweep (bool): is the component using sweep?
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
            sweep,
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
        sweep="auto",
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

        if sweep == "auto":
            sweep = str(module_instance.type) == "SweepComponent"
            if sweep:
                print(f"Module {module_name} detected as SweepComponent: {sweep}")

        if windows == "auto":
            if (
                str(module_instance.type) == "HDInsightComponent"
                or str(module_instance.type) == "ScopeComponent"
                or str(module_instance.type) == "DataTransferComponent"
                or str(module_instance.type) == "SweepComponent"
            ):
                # HDI/scope/datatransfer/sweep modules might not have that environment object
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

        if sweep:
            self._apply_sweep_runsettings(
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

    def _check_if_spec_yaml_override_is_needed(self):
        if self.config.module_loader.use_local == "":
            print(
                "All components are using remote copy, so override will not be executed. For components you want submission-time override of images/tags/etc., please specify them in `use_local`."
            )
            return False, None
        if not self.config.tenant_overrides.allow_override:
            print(
                "Spec yaml file override is not allowed. If you want to use this feature, please set `tenant_overrides.allow_override` to True in your pipeline yaml."
            )
            return False, None
        cur_tenant = self.config.aml.tenant
        for tenant in self.config.tenant_overrides.mapping.keys():
            if tenant == cur_tenant:
                print(
                    f"Your tenant is inconsistent with spec yaml, We will override relevant fields in your spec yaml files based on entry `{tenant}` in your pipeline yaml file."
                )
                return True, self.config.tenant_overrides.mapping[tenant]
            if self.config.run.config_dir:
                config_file_path = os.path.join(
                    self.config.run.config_dir, "aml", tenant + ".yaml"
                )
                print(f"config_file_path is {config_file_path}")
                if os.path.exists(config_file_path):
                    with open(config_file_path, "r") as file:
                        config = yaml.safe_load(file)
                    if config["tenant"] == cur_tenant:
                        print(
                            f"Your tenant is inconsistent with spec yaml, We will override relevant fields in your spec yaml files based on entry `{config_file_path}` in your pipeline yaml file."
                        )
                        return True, self.config.tenant_overrides.mapping[tenant]
        return False, None

    def _override_spec_yaml(self, spec_mapping):
        module_keys = self.module_loader.modules_manifest
        yaml_to_be_recovered = []
        env_yaml_override_is_needed = (
            spec_mapping.remove_polymer_pkg_idx
            if "remove_polymer_pkg_idx" in spec_mapping
            else False
        )
        for module_key in module_keys:
            if not self.module_loader.is_local(module_key):
                print(
                    f"Component {module_key} is using the remote copy. Skipping overrides."
                )
                continue
            print(f"Overriding for component: {module_key}.")
            module_entry = self.module_loader.modules_manifest[module_key]
            spec_path = os.path.join(
                self.module_loader.local_steps_folder, module_entry["yaml"]
            )
            (
                old_spec_path,
                old_env_file_path,
                new_env_file_path,
            ) = self._override_single_spec_yaml(
                spec_path, spec_mapping, env_yaml_override_is_needed
            )

            yaml_to_be_recovered.append(
                [old_spec_path, spec_path, old_env_file_path, new_env_file_path]
            )
        return yaml_to_be_recovered

    def _update_value_given_flattened_key(self, nested_dict, dot_key, new_val):
        print(f"Updating key {dot_key}")
        split_key = dot_key.split(".")
        res = nested_dict
        path = ""
        for key in split_key[:-1]:
            path += key + "."
            if not isinstance(res, dict) or key not in res:
                raise KeyError(
                    f"Key {dot_key} not in {nested_dict}. It failed at {path}."
                )
            res = res[key]
        if isinstance(res, dict) and split_key[-1] in res:
            res[split_key[-1]] = new_val
            print(f"The field {dot_key} has been updated to {new_val} successfully.")
        else:
            raise KeyError(f"Key {dot_key} not in {nested_dict}.")

    def _override_single_spec_yaml(
        self, spec_path, spec_mapping, env_yaml_override_is_needed
    ):
        spec_filename, spec_ext = os.path.splitext(spec_path)
        old_spec_path = (
            spec_filename + ".not_used"
        )  # remove ".yaml" extension to avoid confusion
        shutil.copyfile(
            spec_path, old_spec_path
        )  # need to recover after pipeline submission

        with open(spec_path) as file:
            spec = yaml.safe_load(file)
        for key in spec_mapping:
            match = jsonpath_ng.parse(key).find(spec)
            if match:
                try:
                    print(f"Find a matching field to override: {key}.")
                    original_val = match[0].value
                    print(
                        f"Original value is {original_val}. Looking for new value now..."
                    )
                    spec_mapping_to_use = spec_mapping[key]
                    if isinstance(original_val, str):
                        print(f"The field to be updated is str.")
                        if original_val in spec_mapping_to_use:
                            new_val = spec_mapping_to_use[original_val]
                            print(f"The new value is: {new_val}.")
                            self._update_value_given_flattened_key(spec, key, new_val)
                        else:
                            for pattern in spec_mapping_to_use:
                                if re.match(pattern, original_val):
                                    new_val = spec_mapping_to_use[pattern]
                                    print(f"The new pattern is: {new_val}.")
                                    self._update_value_given_flattened_key(
                                        spec,
                                        key,
                                        re.sub(pattern, new_val, original_val),
                                    )
                    elif isinstance(original_val, dict):
                        print("The field to be updated is dict")
                        for spec_mapping_key in spec_mapping_to_use:
                            self._update_value_given_flattened_key(
                                spec,
                                ".".join([key, spec_mapping_key]),
                                spec_mapping_to_use[spec_mapping_key],
                            )
                    else:
                        print(
                            f"Override for key {key} is not supported yet. Please open a [feature request](https://github.com/Azure/shrike/issues) if necessary."
                        )
                except KeyError:
                    print(f"Key {key} does not in file {spec_path}. Skip overrides.")
        new_env_file_path = None
        old_env_file_path = None
        if env_yaml_override_is_needed:
            spec_dirname = os.path.dirname(spec_path)
            polymer_pkg_idx = "--index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/"
            print(f"Will remove {polymer_pkg_idx} from environment.conda.")
            try:
                conda_dependencies_file = spec["environment"]["conda"][
                    "conda_dependencies_file"
                ]
                print("conda_dependencies_file exists.")
                (
                    found_index_url,
                    new_file,
                    new_env_file_path,
                    old_env_file_path,
                ) = self._remove_polymer_pkg_idx_if_exists_and_save_new(
                    spec_dirname, conda_dependencies_file, polymer_pkg_idx
                )
                if found_index_url:
                    spec["environment"]["conda"]["conda_dependencies_file"] = new_file
            except KeyError:
                # conda_dependencies_file does not exist
                pass
            try:
                conda_dependencies = spec["environment"]["conda"]["conda_dependencies"][
                    "dependencies"
                ]
                print("conda_dependencies_file exists.")
                for idx, dependency in enumerate(conda_dependencies):
                    if isinstance(dependency, dict) and "pip" in dependency:
                        pip_dependencies = dependency["pip"]
                        if polymer_pkg_idx in pip_dependencies:
                            pip_dependencies.remove(polymer_pkg_idx)
                            dependency["pip"] = pip_dependencies
                            conda_dependencies[idx] = dependency
                spec["environment"]["conda"]["conda_dependencies"][
                    "dependencies"
                ] = conda_dependencies
            except KeyError:
                # conda_dependencies does not exist
                pass
            try:
                pip_requirements_file = spec["environment"]["conda"][
                    "pip_requirements_file"
                ]
                print("pip_requirements_file exists.")
                (
                    found_index_url,
                    new_file,
                    new_env_file_path,
                    old_env_file_path,
                ) = self._remove_polymer_pkg_idx_if_exists_and_save_new(
                    spec_dirname, pip_requirements_file, polymer_pkg_idx
                )
                if found_index_url:
                    spec["environment"]["conda"]["pip_requirements_file"] = new_file
            except KeyError:
                # pip_requirements_file does not exist
                pass

        with open(spec_path, "w") as file:
            yaml.safe_dump(spec, file)
        return old_spec_path, old_env_file_path, new_env_file_path

    def _remove_polymer_pkg_idx_if_exists_and_save_new(
        self, spec_dirname, file, polymer_pkg_idx
    ):
        found_index_url = False
        new_file = ""
        new_file_path = ""
        old_file_path = ""
        with open(os.path.join(spec_dirname, file), "r") as f:
            lines = f.readlines()
        for line in lines:
            if polymer_pkg_idx in line:
                found_index_url = True
                lines.remove(line)
        if found_index_url:
            file_name, file_ext = os.path.splitext(file)
            new_file = file_name + "_" + self.config.aml.tenant + file_ext
            new_file_path = os.path.join(spec_dirname, new_file)
            with open(new_file_path, "w") as f:
                f.writelines(lines)
            old_file_path = os.path.join(
                spec_dirname, os.path.splitext(file)[0] + ".not_used"
            )
            shutil.move(os.path.join(spec_dirname, file), old_file_path)
        return found_index_url, new_file, new_file_path, old_file_path

    def _recover_spec_yaml(self, spec_file_pairs, keep_modified_files):
        print(
            f"Reverting changes to spec yaml files. Keeping modified spec yaml files: {keep_modified_files}."
        )
        for (
            old_spec_path,
            new_spec_path,
            old_env_file_path,
            new_env_file_path,
        ) in spec_file_pairs:
            # Question: do we want to keep a copy of the modified files?
            if keep_modified_files:
                filename, ext = os.path.splitext(new_spec_path)
                shutil.move(
                    new_spec_path, filename + "_" + self.config.aml.tenant + ext
                )
                if os.path.exists(filename + ".additional_includes"):
                    shutil.copyfile(
                        filename + ".additional_includes",
                        filename
                        + "_"
                        + self.config.aml.tenant
                        + ".additional_includes",
                    )
            else:
                if new_env_file_path:
                    os.remove(new_env_file_path)
            shutil.move(old_spec_path, new_spec_path)
            if old_env_file_path:
                shutil.move(
                    old_env_file_path, os.path.splitext(old_env_file_path)[0] + ".yaml"
                )

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

    def build_and_submit_new_pipeline(self):
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
            pipeline_tags.update(self.repository_info)
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
            return pipeline_run

        else:
            print("Exiting now, if you want to submit please override run.submit=True")
            self.__class__.BUILT_PIPELINE = (
                pipeline  # return so we can have some unit tests done
            )
            return

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

        self.repository_info = get_repo_info()
        print(f"Running from repository: {self.repository_info}")

        print("azureml.core.VERSION = {}".format(azureml.core.VERSION))
        self.connect()

        if self.config.run.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        pipeline_run = None
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
            keep_modified_files = False
            if self.config.tenant_overrides.allow_override:
                print("Check if tenant is consistent with spec yaml")
                yaml_to_be_recovered = []
                override, mapping = self._check_if_spec_yaml_override_is_needed()
                if override:
                    try:
                        tenant = self.config.aml.tenant
                        print(
                            f"Performing spec yaml override to adapt to tenant: {tenant}."
                        )
                        yaml_to_be_recovered = self._override_spec_yaml(mapping)
                        keep_modified_files = (
                            self.config.tenant_overrides.keep_modified_files
                        )

                        pipeline_run = self.build_and_submit_new_pipeline()
                        self._recover_spec_yaml(
                            yaml_to_be_recovered, keep_modified_files
                        )
                    except:
                        print("An error occured, override is not successful.")

            else:
                pipeline_run = self.build_and_submit_new_pipeline()

        if not pipeline_run:
            # not submitting code, exit now
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

            arg_parser = argparse.ArgumentParser()
            arg_parser.add_argument("--config-dir")
            args, _ = arg_parser.parse_known_args()
            cfg.run.config_dir = os.path.join(
                HydraConfig.get().runtime.cwd, args.config_dir
            )

            print("*** CONFIGURATION ***")
            print(OmegaConf.to_yaml(cfg))

            # create class instance
            main_instance = cls(cfg)

            # run
            main_instance.run()

        hydra_run()

        return cls.BUILT_PIPELINE  # return so we can have some unit tests done
