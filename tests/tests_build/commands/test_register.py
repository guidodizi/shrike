# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the `register` module.

Consider decorating long end-to-end tests with `@pytest.mark.order(-1)`.
"""

import os
import pytest
import subprocess
import yaml
from random import uniform
from pathlib import Path


from shrike.build.commands import register
from shrike.build.commands import prepare
from shrike.build.core.configuration import (
    Configuration,
    load_configuration_from_args_and_env,
)


CANARY_WORKSPACE = "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-eus2"


def test_validate_branch_not_match():
    with pytest.raises(ValueError):
        reg = register.Register()
        reg.config = Configuration(
            source_branch="NOT_MATCH_BRANCH", compliant_branch="^refs/heads/main$"
        )
        reg.validate_branch()


def test_validate_branch(caplog):
    reg = register.Register()
    reg.config = Configuration(
        source_branch="refs/heads/main", compliant_branch="^refs/heads/main$"
    )
    with caplog.at_level("INFO"):
        reg.validate_branch()
    assert "Expected branch: ^refs/heads/main$" in caplog.text
    assert "Current branch: refs/heads/main" in caplog.text
    assert "Current branch matches configured regular expression." in caplog.text


@pytest.mark.order(-1)
@pytest.mark.parametrize("mode", ["foo", "aether", "aml"])
def test_find_signed_component_specification_files(caplog, mode):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    if mode in ["aether", "aml"]:
        # Create catlog files for aether or aml mode
        prep = prepare.Prepare()
        prep.config = Configuration(signing_mode=mode)
        built_component_files = prep.build_all_components(
            prep.find_component_specification_files()
        )
        prep.create_catalog_files(built_component_files)

    reg = register.Register()
    reg.config = Configuration()
    reg.attach_workspace(CANARY_WORKSPACE)

    with caplog.at_level("INFO"):
        reg.find_signed_component_specification_files()

    if mode == "aether":
        assert "Find a signed component for Aether:" in caplog.text
        assert "Cannot find any signed components for AML." in caplog.text
    elif mode == "aml":
        assert "Find 4 signed components for AML." in caplog.text
    else:
        assert "Cannot find any signed components for AML." in caplog.text


@pytest.mark.order(-1)
def test_list_registered_component(caplog):
    reg = register.Register()
    reg.config = Configuration()
    reg.attach_workspace(CANARY_WORKSPACE)

    with caplog.at_level("INFO"):
        reg.list_registered_component()

    assert "DefaultVersion" in caplog.text
    assert not reg._errors

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_register_all_signed_components_already_exist_component_version(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create a temporary component folder and generate a random float number as
    # the version number so that we can "almost surely" have a new version number
    # every time we re-run this unit test.
    tmp_dir = str(get_target_path_in_steps_directory("component4/tmp_dir"))
    random_version_number = str(uniform(0.1, 1.0))
    tmp_yaml = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": random_version_number,
        "type": "CommandComponent",
        "command": "pip freeze",
    }
    os.mkdir(tmp_dir)
    with open(tmp_dir + "/spec.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml, tmp_spec)

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml", fail_if_version_exists=True)
    built_component_files = prep.build_all_components(
        prep.find_component_specification_files()
    )
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    reg.config = Configuration(fail_if_version_exists=True)
    reg.attach_workspace(CANARY_WORKSPACE)

    already_exist_component_path = str(
        get_target_path_in_steps_directory("component4/.build/spec.yaml")
    )
    not_yet_exist_component_path = str(
        get_target_path_in_steps_directory("component4/tmp_dir/.build/spec.yaml")
    )
    with caplog.at_level("INFO"):
        reg.register_all_signed_components(
            [already_exist_component_path, not_yet_exist_component_path]
        )
    assert reg._component_statuses[already_exist_component_path]["register"] == "failed"
    assert (
        reg._component_statuses[not_yet_exist_component_path]["register"] == "succeeded"
    )
    assert len(reg._errors) == 1
    assert (
        "(UserError) Version 0.0.1 already exists in component convert2ss."
        in caplog.text
    )
    assert '"displayName": "dummy_component"' in caplog.text
    assert (
        f"{random_version_number} is production-ready. Setting as default."
        in caplog.text
    )

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_register_all_signed_components_use_build_number(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml")
    built_component_files = prep.build_all_components(
        prep.find_component_specification_files()
    )
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    build_number = os.environ.get("BUILD_BUILDNUMBER") or str(uniform(0.1, 1.0))
    build_sourcebranch = os.environ.get("BUILD_SOURCEBRANCH") or "refs/heads/main"
    reg.config = load_configuration_from_args_and_env(
        args=["--use-build-number"],
        env={
            "BUILD_SOURCEBRANCH": build_sourcebranch,
            "BUILD_BUILDNUMBER": build_number,
        },
    )
    reg.attach_workspace(CANARY_WORKSPACE)

    component_path = str(
        get_target_path_in_steps_directory("component4/.build/spec.yaml")
    )
    with caplog.at_level("INFO"):
        reg.register_all_signed_components([component_path])

    assert reg._component_statuses[component_path]["register"] == "succeeded"
    assert len(reg._errors) == 0

    # Note: There is a trasient error of empty json output generated by az cli.
    # Will uncomment the following line once this error is resolved.
    # assert '"displayName": "Convert Text to StructureStream"' in caplog.text
    assert (
        f"Overwrite the component version with the specified value {build_number}"
        in caplog.text
    )
    assert "is production-ready. Setting as default." in caplog.text
    assert "is not production-ready. NOT setting as default." not in caplog.text

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_set_default_version_given_prod(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create a temporary component folder and generate a random float number as
    # the version number so that we can "almost surely" have a new version number
    # every time we re-run this unit test.
    tmp_dir = str(get_target_path_in_steps_directory("tmp_dir"))
    random_version_number = uniform(0.1, 1.0)
    version_numbers = [str(random_version_number) + i for i in ["", ".1", ".1.1"]] + [
        str(int(random_version_number * 100000000))
    ]
    os.mkdir(tmp_dir)
    for idx, version in enumerate(version_numbers):
        tmp_yaml = {
            "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
            "name": "dummy_component",
            "version": version,
            "type": "CommandComponent",
            "command": "pip freeze",
        }

        with open(tmp_dir + f"/spec{idx}.yaml", "w") as tmp_spec:
            yaml.dump(tmp_yaml, tmp_spec)

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml")
    built_component_files = prep.build_all_components(
        files=[tmp_dir + f"/spec{idx}.yaml" for idx in range(len(version_numbers))]
    )
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    reg.config = Configuration()
    reg.attach_workspace(CANARY_WORKSPACE)

    component_path = [
        str(get_target_path_in_steps_directory(f"tmp_dir/.build/spec{idx}.yaml"))
        for idx in range(len(version_numbers))
    ]

    with caplog.at_level("INFO"):
        reg.register_all_signed_components(component_path)
    assert len(reg._errors) == 0
    for idx in range(len(version_numbers)):
        assert reg._component_statuses[component_path[idx]]["register"] == "succeeded"
        assert (
            f"{version_numbers[idx]} is production-ready. Setting as default."
            in caplog.text
        )

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_set_default_version_given_dev(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create a temporary component folder and generate a random float number as
    # the version number so that we can "almost surely" have a new version number
    # every time we re-run this unit test.
    tmp_dir = str(get_target_path_in_steps_directory("tmp_dir"))
    random_version_number = uniform(0.1, 1.0)
    version_numbers = [
        str(random_version_number) + i
        for i in ["-dev", "-beta0", "-alpha.1", "-rc1", "alpha"]
    ]
    os.mkdir(tmp_dir)
    for idx, version in enumerate(version_numbers):
        tmp_yaml = {
            "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
            "name": "dummy_component",
            "version": version,
            "type": "CommandComponent",
            "command": "pip freeze",
        }

        with open(tmp_dir + f"/spec{idx}.yaml", "w") as tmp_spec:
            yaml.dump(tmp_yaml, tmp_spec)

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml")
    built_component_files = prep.build_all_components(
        files=[tmp_dir + f"/spec{idx}.yaml" for idx in range(len(version_numbers))]
    )
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    reg.config = Configuration()
    reg.attach_workspace(CANARY_WORKSPACE)

    component_path = [
        str(get_target_path_in_steps_directory(f"tmp_dir/.build/spec{idx}.yaml"))
        for idx in range(len(version_numbers))
    ]

    with caplog.at_level("INFO"):
        reg.register_all_signed_components(component_path)
    assert len(reg._errors) == 0
    for idx in range(len(version_numbers)):
        assert reg._component_statuses[component_path[idx]]["register"] == "succeeded"
        assert (
            f"{version_numbers[idx]} is not production-ready. NOT setting as default."
            in caplog.text
        )

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


def test_read_component_version(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Start registration
    reg = register.Register()
    reg.config = Configuration()

    reg.read_component_version("tests/tests_build/steps/component3/spec.yaml")
    reg.read_component_version("tests/tests_build/steps/component2/spec.yaml")
    assert (
        "Component does not have version attribute, attempting to read module version."
        in caplog.text
    )
    assert "Component tests/tests_build/steps/spec.yaml has version 0.0.4."

    # Clean
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_register_component_command(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create a temporary component folder and generate a random float number as
    # the version number so that we can "almost surely" have a new version number
    # every time we re-run this unit test.
    tmp_dir = str(get_target_path_in_steps_directory("tmp_dir"))
    random_version_number = str(uniform(0.1, 1.0))
    tmp_yaml = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": random_version_number,
        "type": "CommandComponent",
        "command": "pip freeze",
    }
    tmp_yaml2 = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": random_version_number + ".0",
        "type": "CommandComponent",
        "command": "pip freeze",
    }
    tmp_yaml3 = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": random_version_number + ".0-beta",
        "type": "CommandComponent",
        "command": "pip freeze",
    }
    os.mkdir(tmp_dir)
    with open(tmp_dir + "/spec.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml, tmp_spec)
    with open(tmp_dir + "/spec2.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml2, tmp_spec)
    with open(tmp_dir + "/spec3.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml3, tmp_spec)

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml")
    built_component_files = prep.build_all_components(
        prep.find_component_specification_files()
    )
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    reg.config = Configuration()
    reg.attach_workspace(CANARY_WORKSPACE)

    component_path_1 = str(
        get_target_path_in_steps_directory("tmp_dir/.build/spec.yaml")
    )
    component_path_2 = str(
        get_target_path_in_steps_directory("tmp_dir/.build/spec2.yaml")
    )
    component_path_3 = str(
        get_target_path_in_steps_directory("tmp_dir/.build/spec3.yaml")
    )

    with caplog.at_level("INFO"):
        reg.register_all_signed_components(
            [component_path_1, component_path_2, component_path_3]
        )

    assert (
        f"Register command is ml component create --file {component_path_1} --label default\n"
        in caplog.text
    )
    assert (
        f"Register command is ml component create --file {component_path_2} --label default\n"
        in caplog.text
    )
    assert (
        f"Register command is ml component create --file {component_path_3}\n"
        in caplog.text
    )

    build_number = os.environ.get("BUILD_BUILDNUMBER") or str(uniform(0.1, 1.0))
    build_sourcebranch = os.environ.get("BUILD_SOURCEBRANCH") or "refs/heads/main"
    reg.config = load_configuration_from_args_and_env(
        args=["--use-build-number"],
        env={
            "BUILD_SOURCEBRANCH": build_sourcebranch,
            "BUILD_BUILDNUMBER": build_number,
        },
    )
    with caplog.at_level("INFO"):
        reg.register_all_signed_components([component_path_1])
    assert (
        f"Register command is ml component create --file {component_path_1} --version {build_number} --label default\n"
        in caplog.text
    )

    # Clean
    subprocess.run(["git", "clean", "-xdf"])


@pytest.mark.order(-1)
def test_run_with_config_runs_end_to_end(caplog):
    # Clean up .build directories
    subprocess.run(["git", "clean", "-xdf"])

    # Create a temporary component folder and generate a random float number as
    # the version number so that we can "almost surely" have a new version number
    # every time we re-run this unit test.
    tmp_dir = str(get_target_path_in_steps_directory("tmp_dir"))
    random_version_number = str(uniform(0.1, 1.0))
    tmp_yaml = {
        "$schema": "http://azureml/sdk-2-0/CommandComponent.json",
        "name": "dummy_component",
        "version": random_version_number,
        "type": "CommandComponent",
        "command": "pip freeze",
    }
    os.mkdir(tmp_dir)
    with open(tmp_dir + "/spec.yaml", "w") as tmp_spec:
        yaml.dump(tmp_yaml, tmp_spec)

    # Create catlog files
    prep = prepare.Prepare()
    prep.config = Configuration(signing_mode="aml")
    built_component_files = prep.build_all_components(files=[tmp_dir + "/spec.yaml"])
    prep.create_catalog_files(built_component_files)

    # Start registration
    reg = register.Register()
    reg.config = Configuration(
        workspaces=[CANARY_WORKSPACE], source_branch="refs/heads/main"
    )
    with caplog.at_level("INFO"):
        reg.run_with_config()

    print(caplog.text)

    assert len(reg._errors) == 0
    assert '"displayName": "dummy_component"' in caplog.text
    assert random_version_number in caplog.text
    assert "Find 1 signed components for AML." in caplog.text
    assert "List of components in workspace before current registration." in caplog.text
    assert "List of components in workspace after current registration." in caplog.text

    # Cleanup
    subprocess.run(["git", "clean", "-xdf"])


def get_target_path_in_steps_directory(target) -> Path:
    end_of_path = "steps/" + target
    res = Path(__file__).parent.parent.resolve() / end_of_path
    return res
