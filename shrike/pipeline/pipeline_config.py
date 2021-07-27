# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Configuration dataclasses for AMLPipelineHelper """
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional, Any, Dict
from shrike.pipeline.module_helper import module_loader_config, module_manifest

# Default config for HDI components
HDI_DEFAULT_CONF = '{"spark.yarn.appMasterEnv.DOTNET_ASSEMBLY_SEARCH_PATHS":"./udfs","spark.yarn.maxAppAttempts":"1","spark.yarn.appMasterEnv.PYSPARK_PYTHON":"/usr/bin/anaconda/envs/py37/bin/python3","spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON":"/usr/bin/anaconda/envs/py37/bin/python3"}'


@dataclass
class pipeline_cli_config:  # pylint: disable=invalid-name
    """Pipeline config for command line parameters"""

    regenerate_outputs: bool = False
    continue_on_failure: bool = False
    disable_telemetry: bool = False
    verbose: bool = False
    submit: bool = False
    resume: bool = False
    canary: bool = False
    export: Optional[str] = None
    silent: bool = False
    wait: bool = False
    experiment_name: str = MISSING
    pipeline_run_id: str = MISSING
    tags: Optional[Any] = None
    config_dir: Optional[str] = None


@dataclass
class aml_connection_config:  # pylint: disable=invalid-name
    """AML connection configuration"""

    subscription_id: str = MISSING
    resource_group: str = MISSING
    workspace_name: str = MISSING
    tenant: Optional[str] = None
    auth: str = "interactive"
    force: bool = False


@dataclass
class pipeline_compute_config:  # pylint: disable=invalid-name
    """AML workspace compute targets and I/O modes"""

    default_compute_target: str = MISSING
    linux_cpu_dc_target: str = MISSING
    linux_cpu_prod_target: str = MISSING
    linux_gpu_dc_target: str = MISSING
    linux_gpu_prod_target: str = MISSING
    linux_input_mode: str = "mount"
    linux_output_mode: str = "mount"

    windows_cpu_prod_target: str = MISSING
    windows_cpu_dc_target: str = MISSING
    windows_input_mode: str = "download"
    windows_output_mode: str = "upload"

    hdi_prod_target: str = MISSING
    hdi_driver_memory: str = "4g"
    hdi_driver_cores: int = 2
    hdi_executor_memory: str = "3g"
    hdi_executor_cores: int = 2
    hdi_number_executors: int = 10
    hdi_conf: Optional[Any] = MISSING

    parallel_node_count: int = 10
    parallel_process_count_per_node: Optional[int] = MISSING
    parallel_run_invocation_timeout: int = 10800
    parallel_run_max_try: int = 3
    parallel_mini_batch_size: int = 1
    parallel_error_threshold: int = -1

    datatransfer_target: Optional[str] = MISSING

    compliant_datastore: str = MISSING
    noncompliant_datastore: Optional[str] = MISSING


@dataclass
class tenant_override_config:
    allow_override: bool = False
    keep_modified_files: bool = False
    mapping: Dict[str, Any] = field(default_factory=lambda: {})


def default_config_dict():
    """Constructs the config dictionary for the pipeline helper settings"""
    return {
        "aml": aml_connection_config,
        "run": pipeline_cli_config,
        "compute": pipeline_compute_config,
        "module_loader": module_loader_config,
        "modules": module_manifest,
        "tenant_overrides": tenant_override_config,
    }
