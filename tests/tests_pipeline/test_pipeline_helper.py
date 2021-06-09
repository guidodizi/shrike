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
            assert module_instance.runsettings.target == "cpu-win"
        elif not windows and gpu:
            assert module_instance.runsettings.target == "gpu-cluster"
        elif not windows and not gpu:
            assert module_instance.runsettings.target == "cpu-cluster"


def test_get_component_name_from_instance():
    component_instance = pipeline_helper.component_load(component_key="dummy_key")
    step_instance = component_instance()
    component_name = pipeline_helper._get_component_name_from_instance(step_instance)
    assert component_name == "dummy_key"
