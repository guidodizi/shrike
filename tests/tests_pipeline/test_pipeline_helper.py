# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for pipeline_helper"""

import pytest
import sys
from omegaconf import OmegaConf
from pathlib import Path
from shrike.pipeline.pipeline_helper import AMLPipelineHelper

# Load the testing configuration YAML file
config = OmegaConf.load(Path(__file__).parent / "data/test_configuration.yaml")

# Initiate AMLPipelineHelper class with the testing configuration
pipeline_helper = AMLPipelineHelper(config=config)
pipeline_helper.connect()


def test_validate_experiment_name():
    """Unit tests for validate_experiment_name function"""
    with pytest.raises(ValueError):
        AMLPipelineHelper.validate_experiment_name("")
    with pytest.raises(ValueError):
        AMLPipelineHelper.validate_experiment_name("_exp-name")
    with pytest.raises(ValueError):
        AMLPipelineHelper.validate_experiment_name("wront.period")
    assert AMLPipelineHelper.validate_experiment_name("Correct-NAME_")
    assert AMLPipelineHelper.validate_experiment_name("ALLARELETTERS")
    assert AMLPipelineHelper.validate_experiment_name("12344523790")


def test_get_component_name_from_instance():
    """Unit tests for _get_component_name_from_instance"""
    component_instance = pipeline_helper.component_load(component_key="dummy_key")
    step_instance = component_instance()
    component_name = pipeline_helper._get_component_name_from_instance(step_instance)
    assert component_name == "dummy_key"


def test_parse_pipeline_tags(capsys):
    """Unit tests for _parse_pipeline_tags"""
    assert pipeline_helper._parse_pipeline_tags() == {"test_key": "test_value"}

    pipeline_helper.config.run.tags = '{"WRONG_JSON": 1'
    pipeline_helper._parse_pipeline_tags()
    out, _ = capsys.readouterr()
    sys.stdout.write(out)
    assert 'The pipeline tags {"WRONG_JSON": 1 is not a valid json-style string.' in out

    pipeline_helper.config.run.tags = '{"test_key": "test_value"}'


@pytest.mark.parametrize(
    "windows,gpu", [(True, True), (True, False), (False, True), (False, False)]
)
def test_apply_parallel_runsettings(capsys, windows, gpu):
    """Unit tests for _apply_parallel_runsettings()"""
    # Create a module instance
    module_instance_fun = pipeline_helper.component_load(
        component_key="prscomponentlinux"
    )
    module_instance = module_instance_fun(input_dir="foo")

    if windows and gpu:
        with pytest.raises(ValueError):
            pipeline_helper._apply_parallel_runsettings(
                module_name="prscomponentlinux",
                module_instance=module_instance,
                windows=windows,
                gpu=gpu,
            )
    else:
        pipeline_helper._apply_parallel_runsettings(
            module_name="prscomponentlinux",
            module_instance=module_instance,
            windows=windows,
            gpu=gpu,
        )

        # Testing the stdout
        out, _ = capsys.readouterr()
        sys.stdout.write(out)
        assert "Using parallelrunstep compute target" in out
        assert f"to run {module_instance.name}" in out

        # Testing parallel runsetting parameter configuration
        assert module_instance.runsettings.parallel.error_threshold == -1
        assert module_instance.runsettings.parallel.mini_batch_size == "1"
        assert module_instance.runsettings.parallel.node_count == 10
        assert module_instance.runsettings.parallel.process_count_per_node is None
        assert module_instance.runsettings.parallel.run_invocation_timeout == 10800
        assert module_instance.runsettings.parallel.run_max_try == 3

        # Testing compute target configuration
        if windows and not gpu:
            assert module_instance.runsettings.target == "cpu-dc-win"
        elif not windows and gpu:
            assert module_instance.runsettings.target == "gpu-cluster"
        elif not windows and not gpu:
            assert module_instance.runsettings.target == "cpu-cluster"


def test_apply_scope_runsettings():
    module_instance = pipeline_helper.component_load("convert2ss")
    step_instance = module_instance(TextData="foo", ExtractionClause="foo")

    adla_account_name = "office-adhoc-c14"
    custom_job_name_suffix = "test"
    scope_param = "-tokens 50"
    pipeline_helper._apply_scope_runsettings(
        "convert2ss",
        step_instance,
        adla_account_name=adla_account_name,
        custom_job_name_suffix=custom_job_name_suffix,
        scope_param=scope_param,
    )

    assert step_instance.runsettings.scope.adla_account_name == adla_account_name
    assert (
        step_instance.runsettings.scope.custom_job_name_suffix == custom_job_name_suffix
    )
    assert step_instance.runsettings.scope.scope_param == scope_param


def test_apply_datatransfer_runsettings():
    module_instance = pipeline_helper.component_load("data_transfer")
    step_instance = module_instance(source_data="foo")
    pipeline_helper._apply_datatransfer_runsettings("data_Transfer", step_instance)

    assert step_instance.runsettings.target == "data-factory"


@pytest.mark.parametrize("mpi", [True, False])
def test_apply_windows_runsettings(capsys, mpi):
    """Unit tests for _apply_windows_runsettings()"""
    # Create a module instance
    module_name = (
        "stats_passthrough_windows_mpi" if mpi else "stats_passthrough_windows"
    )
    module_instance_fun = pipeline_helper.component_load(component_key=module_name)
    module_instance = module_instance_fun(input_path="foo")

    pipeline_helper._apply_windows_runsettings(
        module_name=module_name,
        module_instance=module_instance,
        mpi=mpi,
        node_count=2,
        process_count_per_node=3,
    )

    # Testing the stdout
    out, _ = capsys.readouterr()
    sys.stdout.write(out)
    assert (
        f"Using windows compute target cpu-dc-win to run {module_name} from pipeline class AMLPipelineHelper"
        in out
    )
    assert f"to run {module_instance.name}" in out

    # Testing mpi runsetting parameter configuration
    if mpi:
        assert module_instance.runsettings.resource_layout.node_count == 2
        assert module_instance.runsettings.resource_layout.process_count_per_node is 3

    # Testing compute target configuration
    assert module_instance.runsettings.target == "cpu-dc-win"

    # Testing input and output mode
    assert module_instance.inputs.input_path.mode == "download"
    assert module_instance.outputs.output_path.output_mode == "upload"


def test_apply_hdi_runsettings(capsys):
    """Unit tests for _apply_hdi_runsettings()"""
    # Create a module instance
    module_name = "SparkHelloWorld"
    module_instance_fun = pipeline_helper.component_load(component_key=module_name)
    module_instance = module_instance_fun(input_path="foo")

    pipeline_helper._apply_hdi_runsettings(
        module_name=module_name,
        module_instance=module_instance,
        conf='{"spark.yarn.maxAppAttempts": 1, "spark.sql.shuffle.partitions": 3000}',
    )

    # Testing the stdout
    out, _ = capsys.readouterr()
    sys.stdout.write(out)
    assert (
        "Using HDI compute target cpu-cluster to run SparkHelloWorld from pipeline class AMLPipelineHelper"
        in out
    )

    # Testing HDI runsetting parameter configuration
    assert module_instance.runsettings.hdinsight.driver_memory == "2g"
    assert module_instance.runsettings.hdinsight.driver_cores == 2
    assert module_instance.runsettings.hdinsight.executor_memory == "2g"
    assert module_instance.runsettings.hdinsight.executor_cores == 2
    assert module_instance.runsettings.hdinsight.number_executors == 2
    assert (
        module_instance.runsettings.hdinsight.conf[
            "spark.yarn.appMasterEnv.DOTNET_ASSEMBLY_SEARCH_PATHS"
        ]
        == "./udfs"
    )
    assert module_instance.runsettings.hdinsight.conf["spark.yarn.maxAppAttempts"] == 1
    assert (
        module_instance.runsettings.hdinsight.conf[
            "spark.yarn.appMasterEnv.PYSPARK_PYTHON"
        ]
        == "/usr/bin/anaconda/envs/py37/bin/python3"
    )
    assert (
        module_instance.runsettings.hdinsight.conf[
            "spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON"
        ]
        == "/usr/bin/anaconda/envs/py37/bin/python3"
    )
    assert (
        module_instance.runsettings.hdinsight.conf["spark.sql.shuffle.partitions"]
        == 3000
    )

    # Testing compute target configuration
    assert module_instance.runsettings.target == "cpu-cluster"

    # Testing input and output mode
    assert module_instance.inputs.input_path.mode is None
    assert module_instance.outputs.output_path.output_mode is None
