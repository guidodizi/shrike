# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for pipeline_helper"""


from functools import lru_cache
import pytest
import sys
from omegaconf import OmegaConf
from pathlib import Path
from shrike.pipeline import AMLPipelineHelper
import yaml
import os
import shutil


from shrike._core import stream_handler


@lru_cache(maxsize=None)
def pipeline_helper():
    """
    Initiate AMLPipelineHelper class with the testing configuration.
    """
    config = OmegaConf.load(Path(__file__).parent / "data/test_configuration.yaml")
    rv = AMLPipelineHelper(config=config)
    rv.connect()
    return rv


# Define tenant_id
tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47"


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
    component_instance = pipeline_helper().component_load(component_key="dummy_key")
    step_instance = component_instance()
    component_name = pipeline_helper()._get_component_name_from_instance(step_instance)
    assert component_name == "dummy_key"


def test_parse_pipeline_tags():
    """Unit tests for _parse_pipeline_tags"""
    assert pipeline_helper()._parse_pipeline_tags() == {"test_key": "test_value"}

    pipeline_helper().config.run.tags = '{"WRONG_JSON": 1'

    with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
        pipeline_helper()._parse_pipeline_tags()

    assert (
        'The pipeline tags {"WRONG_JSON": 1 is not a valid json-style string.'
        in str(handler)
    )

    pipeline_helper().config.run.tags = '{"test_key": "test_value"}'


@pytest.mark.parametrize(
    "windows,gpu", [(True, True), (True, False), (False, True), (False, False)]
)
def test_apply_parallel_runsettings(windows, gpu):
    """Unit tests for _apply_parallel_runsettings()"""
    # Create a module instance
    module_instance_fun = pipeline_helper().component_load(
        component_key="prscomponentlinux"
    )
    module_instance = module_instance_fun(input_dir="foo")

    if windows and gpu:
        with pytest.raises(ValueError):
            pipeline_helper()._apply_parallel_runsettings(
                module_name="prscomponentlinux",
                module_instance=module_instance,
                windows=windows,
                gpu=gpu,
            )
    else:
        with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
            pipeline_helper()._apply_parallel_runsettings(
                module_name="prscomponentlinux",
                module_instance=module_instance,
                windows=windows,
                gpu=gpu,
            )

        # Testing the stdout
        out = str(handler)
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
    module_instance = pipeline_helper().component_load("convert2ss")
    step_instance = module_instance(TextData="foo", ExtractionClause="foo")

    adla_account_name = "office-adhoc-c14"
    custom_job_name_suffix = "test"
    scope_param = "-tokens 50"
    pipeline_helper()._apply_scope_runsettings(
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
    module_instance = pipeline_helper().component_load("data_transfer")
    step_instance = module_instance(source_data="foo")
    pipeline_helper()._apply_datatransfer_runsettings("data_Transfer", step_instance)

    assert step_instance.runsettings.target == "data-factory"


def test_apply_sweep_runsettings():
    module_instance = pipeline_helper().component_load("sweep_component")
    step_instance = module_instance()

    algorithm = "random"
    primary_metric = "result"
    goal = "maximize"
    policy_type = "median_stopping"
    evaluation_interval = 1
    delay_evaluation = 1

    # Important tip: note that we can only either specify slack_factor, or slack_amount, but not both
    # See this sweep doc for more details about sweep component: https://componentsdk.azurewebsites.net/components/sweep_component.html
    slack_factor = 0.1
    # slack_amount=0.2

    truncation_percentage = 10
    max_total_trials = 5
    max_concurrent_trials = 4
    timeout_minutes = 30

    pipeline_helper()._apply_sweep_runsettings(
        "sweep_component",
        step_instance,
        algorithm=algorithm,
        primary_metric=primary_metric,
        goal=goal,
        policy_type=policy_type,
        evaluation_interval=evaluation_interval,
        delay_evaluation=delay_evaluation,
        slack_factor=slack_factor,
        truncation_percentage=truncation_percentage,
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials,
        timeout_minutes=timeout_minutes,
    )

    assert step_instance.runsettings.sweep.algorithm == algorithm
    assert step_instance.runsettings.sweep.objective.primary_metric == primary_metric
    assert step_instance.runsettings.sweep.objective.goal == goal
    assert step_instance.runsettings.sweep.early_termination.policy_type == policy_type
    assert (
        step_instance.runsettings.sweep.early_termination.evaluation_interval
        == evaluation_interval
    )
    assert (
        step_instance.runsettings.sweep.early_termination.delay_evaluation
        == delay_evaluation
    )
    assert (
        step_instance.runsettings.sweep.early_termination.slack_factor == slack_factor
    )
    assert (
        step_instance.runsettings.sweep.early_termination.truncation_percentage
        == truncation_percentage
    )
    assert step_instance.runsettings.sweep.limits.max_total_trials == max_total_trials
    assert (
        step_instance.runsettings.sweep.limits.max_concurrent_trials
        == max_concurrent_trials
    )
    assert step_instance.runsettings.sweep.limits.timeout_minutes == timeout_minutes


def test2_apply_sweep_runsettings():
    module_instance = pipeline_helper().component_load("sweep_component")
    step_instance = module_instance()

    algorithm = "random"
    primary_metric = "result"
    goal = "maximize"
    policy_type = "median_stopping"
    evaluation_interval = 1
    delay_evaluation = 1

    # Note that we can only either specify slack_factor, or slack_amount, but not both
    # See this sweep doc for more details about sweep component: https://componentsdk.azurewebsites.net/components/sweep_component.html
    # slack_factor = 0.1
    slack_amount = 0.2

    truncation_percentage = 10
    max_total_trials = 5
    max_concurrent_trials = 4
    timeout_minutes = 30

    pipeline_helper()._apply_sweep_runsettings(
        "sweep_component",
        step_instance,
        algorithm=algorithm,
        primary_metric=primary_metric,
        goal=goal,
        policy_type=policy_type,
        evaluation_interval=evaluation_interval,
        delay_evaluation=delay_evaluation,
        slack_amount=slack_amount,
        truncation_percentage=truncation_percentage,
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials,
        timeout_minutes=timeout_minutes,
    )

    assert step_instance.runsettings.sweep.algorithm == algorithm
    assert step_instance.runsettings.sweep.objective.primary_metric == primary_metric
    assert step_instance.runsettings.sweep.objective.goal == goal
    assert step_instance.runsettings.sweep.early_termination.policy_type == policy_type
    assert (
        step_instance.runsettings.sweep.early_termination.evaluation_interval
        == evaluation_interval
    )
    assert (
        step_instance.runsettings.sweep.early_termination.delay_evaluation
        == delay_evaluation
    )
    assert (
        step_instance.runsettings.sweep.early_termination.slack_amount == slack_amount
    )
    assert (
        step_instance.runsettings.sweep.early_termination.truncation_percentage
        == truncation_percentage
    )
    assert step_instance.runsettings.sweep.limits.max_total_trials == max_total_trials
    assert (
        step_instance.runsettings.sweep.limits.max_concurrent_trials
        == max_concurrent_trials
    )
    assert step_instance.runsettings.sweep.limits.timeout_minutes == timeout_minutes


@pytest.mark.parametrize(
    "compliant,datastore",
    [(True, "fake_compliant_datastore"), (False, "workspaceblobstore")],
)
def test_apply_recommended_runsettings_datatransfer_datastore(compliant, datastore):
    module_instance = pipeline_helper().component_load("data_transfer")
    step_instance = module_instance(source_data="foo")
    pipeline_helper().apply_recommended_runsettings(
        "data_transfer", step_instance, datatransfer=True, compliant=compliant
    )

    assert step_instance.outputs.destination_data.datastore.name == datastore


@pytest.mark.parametrize("mpi", [True, False])
def test_apply_windows_runsettings(mpi):
    """Unit tests for _apply_windows_runsettings()"""
    # Create a module instance
    module_name = (
        "stats_passthrough_windows_mpi" if mpi else "stats_passthrough_windows"
    )
    module_instance_fun = pipeline_helper().component_load(component_key=module_name)
    module_instance = module_instance_fun(input_path="foo")

    with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
        pipeline_helper()._apply_windows_runsettings(
            module_name=module_name,
            module_instance=module_instance,
            mpi=mpi,
            node_count=2,
            process_count_per_node=3,
        )

    # Testing the stdout
    out = str(handler)
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


def test_apply_hdi_runsettings():
    """Unit tests for _apply_hdi_runsettings()"""
    # Create a module instance
    module_name = "SparkHelloWorld"
    module_instance_fun = pipeline_helper().component_load(component_key=module_name)
    module_instance = module_instance_fun(input_path="foo")

    with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
        pipeline_helper()._apply_hdi_runsettings(
            module_name=module_name,
            module_instance=module_instance,
            conf='{"spark.yarn.maxAppAttempts": 1, "spark.sql.shuffle.partitions": 3000}',
        )

    # Testing the stdout
    out = str(handler)
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


@pytest.mark.parametrize(
    "mpi,gpu", [(True, True), (True, False), (False, True), (False, False)]
)
def test_apply_linux_runsettings(mpi, gpu):
    """Unit tests for _apply_linux_runsettings()"""
    # Create a module instance
    module_name = "stats_passthrough_mpi" if mpi else "stats_passthrough"
    module_instance_fun = pipeline_helper().component_load(component_key=module_name)
    module_instance = module_instance_fun(input_path="foo")

    with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
        pipeline_helper()._apply_linux_runsettings(
            module_name=module_name,
            module_instance=module_instance,
            mpi=mpi,
            gpu=gpu,
            node_count=2 if mpi else None,
            process_count_per_node=3 if mpi else None,
        )

    out = str(handler)
    sys.stdout.write(out)

    # Testing mpi runsetting parameter configuration
    if mpi:
        assert module_instance.runsettings.resource_layout.node_count == 2
        assert module_instance.runsettings.resource_layout.process_count_per_node == 3

    # Testing compute target configuration
    if gpu:
        assert module_instance.runsettings.target == "gpu-cluster"
        assert (
            f"Using target gpu-cluster for local code GPU module {module_name} from pipeline class AMLPipelineHelper"
            in out
        )
    else:
        assert module_instance.runsettings.target == "cpu-cluster"
        assert (
            f"Using target cpu-cluster for local CPU module {module_name} from pipeline class AMLPipelineHelper"
            in out
        )

    # Testing input and output mode
    assert module_instance.inputs.input_path.mode == "download"
    assert module_instance.outputs.output_path.output_mode == "upload"


@pytest.mark.parametrize(
    "module_name,expected_stdout",
    [
        ("MultiNodeTrainer", "Module MultiNodeTrainer detected as MPI: True"),
        ("SparkHelloWorld", "Module SparkHelloWorld detected as HDI: True"),
        ("stats_passthrough", ""),
        (
            "stats_passthrough_windows",
            "Module stats_passthrough_windows detected as WINDOWS: True",
        ),
        (
            "stats_passthrough_windows_mpi",
            "Module stats_passthrough_windows_mpi detected as WINDOWS: True",
        ),
        (
            "stats_passthrough_windows_mpi",
            "Module stats_passthrough_windows_mpi detected as MPI: True",
        ),
        ("stats_passthrough_mpi", "Module stats_passthrough_mpi detected as MPI: True"),
        ("convert2ss", "Module convert2ss detected as SCOPE: True"),
        ("prscomponentlinux", "Module prscomponentlinux detected as PARALLEL: True"),
        ("dummy_key", ""),
        ("data_transfer", "Module data_transfer detected as DATATRANSFER: True"),
    ],
)
def test_apply_recommended_runsettings(module_name, expected_stdout):
    """Unit tests for apply_recommended_runsettings()"""
    module_instance_fun = pipeline_helper().component_load(component_key=module_name)
    if module_name == "convert2ss":
        module_instance = module_instance_fun(
            TextData="AnyFile", ExtractionClause="foo"
        )
    elif module_name == "data_transfer":
        module_instance = module_instance_fun(source_data="foo", source_type="foo")
    elif module_name == "dummy_key":
        module_instance = module_instance_fun()
    elif module_name == "MultiNodeTrainer":
        module_instance = module_instance_fun(
            vocab_file="foo", train_file="foo", validation_file="foo"
        )
    elif module_name == "prscomponentlinux":
        module_instance = module_instance_fun(input_dir="foo")
    elif module_name == "SparkHelloWorld":
        module_instance = module_instance_fun(
            input_path="foo", in_file_type="csv", percent_take=1, out_file_type="csv"
        )
    else:
        module_instance = module_instance_fun(input_path="foo")

    with stream_handler("shrike.pipeline.pipeline_helper", level="INFO") as handler:
        pipeline_helper().apply_recommended_runsettings(
            module_name=module_name,
            module_instance=module_instance,
        )

    out = str(handler)

    assert expected_stdout in out


def test_check_if_spec_yaml_override_is_needed_allow_override_false():
    pipeline_helper().config.tenant_overrides.allow_override = False
    override, _ = pipeline_helper()._check_if_spec_yaml_override_is_needed()
    assert override == False
    pipeline_helper().config.tenant_overrides.allow_override = True


def test_check_if_spec_yaml_override_is_needed_given_tenant_id():
    override, mapping = pipeline_helper()._check_if_spec_yaml_override_is_needed()
    assert override == True
    assert mapping == {
        "environment.docker.image": {
            "polymerprod.azurecr.io/training/pytorch:scpilot-rc2": "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
        },
        "tags": {"Office": "aml-ds"},
        "remove_polymer_pkg_idx": True,
    }


def test_check_if_spec_yaml_override_is_needed_given_config_filename():
    with open(Path(__file__).parent / "data/test_configuration.yaml", "r") as file:
        spec = yaml.safe_load(file)
    mapping = spec["tenant_overrides"]["mapping"][tenant_id]
    mapping["description"] = "test"
    spec["tenant_overrides"]["mapping"]["amlds"] = mapping
    spec["tenant_overrides"]["mapping"].pop(tenant_id)

    test_pipeline_helper = AMLPipelineHelper(config=OmegaConf.create(spec))
    test_pipeline_helper.connect()
    test_pipeline_helper.config.run.config_dir = os.path.join(
        Path(__file__).parent / "sample/conf"
    )

    assert "amlds" in test_pipeline_helper.config.tenant_overrides.mapping
    assert tenant_id not in test_pipeline_helper.config.tenant_overrides.mapping
    override, mapping = test_pipeline_helper._check_if_spec_yaml_override_is_needed()
    assert override == True
    assert mapping == {
        "environment.docker.image": {
            "polymerprod.azurecr.io/training/pytorch:scpilot-rc2": "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
        },
        "tags": {"Office": "aml-ds"},
        "description": "test",
        "remove_polymer_pkg_idx": True,
    }


def test_recover_spec_yaml_with_keeping_modified_files():
    file_list = [
        "tmp/spec_path.not_used",
        "tmp/spec_path.yaml",
        "tmp/env_path.not_used",
        "tmp/env_path_" + tenant_id + ".yaml",
    ]
    os.makedirs("tmp")
    for i in file_list:
        with open(i, "w") as file:
            pass
    pipeline_helper()._recover_spec_yaml([file_list], True)
    assert os.path.exists("tmp/spec_path.yaml")
    assert os.path.exists("tmp/env_path.yaml")
    assert os.path.exists("tmp/spec_path_" + tenant_id + ".yaml")
    assert os.path.exists("tmp/env_path_" + tenant_id + ".yaml")
    assert len(os.listdir("tmp")) == 4
    shutil.rmtree("tmp")


def test_recover_spec_yaml_without_keeping_modified_files():
    file_list = [
        "tmp/spec_path.not_used",
        "tmp/spec_path.yaml",
        "tmp/env_path.not_used",
        "tmp/env_path_" + tenant_id + ".yaml",
    ]
    os.makedirs("tmp")
    for i in file_list:
        with open(i, "w") as file:
            pass
    pipeline_helper()._recover_spec_yaml([file_list], False)
    assert os.path.exists("tmp/spec_path.yaml")
    assert os.path.exists("tmp/env_path.yaml")
    assert not os.path.exists("tmp/spec_path_" + tenant_id + ".yaml")
    assert not os.path.exists("tmp/env_path_" + tenant_id + ".yaml")
    assert len(os.listdir("tmp")) == 2
    shutil.rmtree("tmp")


def test_update_value_given_flattened_key():
    d = {"a": {"b": {"c": 1}}}
    pipeline_helper()._update_value_given_flattened_key(d, "a.b.c", 2)
    assert d["a"]["b"]["c"] == 2
    with pytest.raises(KeyError):
        pipeline_helper()._update_value_given_flattened_key(d, "a.b.c.d", 2)
    with pytest.raises(KeyError):
        pipeline_helper()._update_value_given_flattened_key(d, "a.c.d", 2)


def test_override_spec_yaml():
    spec_mapping = pipeline_helper().config.tenant_overrides["mapping"][tenant_id]
    yaml_to_be_recovered = pipeline_helper()._override_spec_yaml(spec_mapping)
    assert len(yaml_to_be_recovered) == len(
        pipeline_helper().module_loader.modules_manifest
    )
    for pair in yaml_to_be_recovered:
        assert len(pair) == 4
        for file_path in pair:
            if file_path:
                assert os.path.exists(file_path)
    pipeline_helper()._recover_spec_yaml(yaml_to_be_recovered, False)


def test_override_single_spec_yaml_without_environment_override():
    folder_path = os.path.join(
        Path(__file__).parent, "sample/steps/multinode_trainer_copy"
    )
    shutil.copytree(
        os.path.join(Path(__file__).parent, "sample/steps/multinode_trainer"),
        folder_path,
    )
    spec_path = os.path.join(folder_path, "module_spec.yaml")
    with open(spec_path, "r") as file:
        spec = yaml.safe_load(file)
    assert (
        spec["environment"]["docker"]["image"]
        == "polymerprod.azurecr.io/training/pytorch:scpilot-rc2"
    )
    assert not spec["tags"]["Office"]
    spec_mapping = pipeline_helper().config.tenant_overrides["mapping"][tenant_id]
    (
        old_spec_path,
        old_env_file_path,
        new_env_file_path,
    ) = pipeline_helper()._override_single_spec_yaml(spec_path, spec_mapping, False)
    assert old_spec_path == os.path.join(folder_path, "module_spec.not_used")
    assert not old_env_file_path
    assert not new_env_file_path
    with open(spec_path, "r") as file:
        spec = yaml.safe_load(file)
    assert (
        spec["environment"]["docker"]["image"]
        == "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
    )
    assert spec["tags"]["Office"] == "aml-ds"
    shutil.rmtree(folder_path)


def test_override_single_spec_yaml_with_environment_override():
    folder_path = os.path.join(
        Path(__file__).parent, "sample/steps/multinode_trainer_copy"
    )
    shutil.copytree(
        os.path.join(Path(__file__).parent, "sample/steps/multinode_trainer"),
        folder_path,
    )
    spec_path = os.path.join(folder_path, "module_spec.yaml")
    spec_mapping = pipeline_helper().config.tenant_overrides["mapping"][tenant_id]
    (
        old_spec_path,
        old_env_file_path,
        new_env_file_path,
    ) = pipeline_helper()._override_single_spec_yaml(spec_path, spec_mapping, True)
    with open(old_env_file_path, "r") as file:
        lines = file.readlines()
    assert (
        "--index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/"
        in " ".join(lines)
    )
    with open(new_env_file_path, "r") as file:
        lines = file.readlines()
    assert (
        "--index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/"
        not in " ".join(lines)
    )
    shutil.rmtree(folder_path)


def test_remove_polymer_pkg_idx_if_exists_and_save_new():
    os.makedirs("tmp")
    with open("tmp/test.txt", "w") as file:
        file.writelines(
            [
                "- --index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/",
                "foo",
            ]
        )
    (
        found_index_url,
        new_file,
        new_file_path,
        old_file_path,
    ) = pipeline_helper()._remove_polymer_pkg_idx_if_exists_and_save_new(
        "tmp",
        "test.txt",
        "--index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/",
    )
    assert found_index_url
    assert new_file == "test_" + tenant_id + ".txt"
    assert new_file_path == os.path.join("tmp", new_file)
    assert old_file_path == os.path.join("tmp", "test.not_used")
    shutil.rmtree("tmp")

    os.makedirs("tmp")
    with open("tmp/test.txt", "w"):
        pass
    (
        found_index_url,
        new_file,
        new_file_path,
        old_file_path,
    ) = pipeline_helper()._remove_polymer_pkg_idx_if_exists_and_save_new(
        "tmp",
        "test.txt",
        "--index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/",
    )
    assert not found_index_url
    assert not new_file
    assert not new_file_path
    assert not old_file_path
    shutil.rmtree("tmp")
