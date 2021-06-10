# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from shrike.build.core.configuration import load_configuration_from_args_and_env
import pytest


def test_load_configuration_from_args_and_env_respects_args():
    args = ["--verbose", "--use-build-number"]
    config = load_configuration_from_args_and_env(
        args, {"BUILD_SOURCEBRANCH": "refs/heads/main", "BUILD_BUILDNUMBER": "1.1.1"}
    )
    assert config.verbose == True
    assert config.use_build_number == True
    assert config.all_component_version == "1.1.1"


def test_load_configuration_all_component_version(caplog):
    args = ["--all-component-version", "2.2.2"]
    with caplog.at_level("INFO"):
        config = load_configuration_from_args_and_env(args, {})
    assert config.all_component_version == "2.2.2"


def test_build_number_overwrite_all_component_version(caplog):
    args = ["--use-build-number", "--all-component-version", "2.2.2"]
    with caplog.at_level("INFO"):
        config = load_configuration_from_args_and_env(
            args,
            {"BUILD_SOURCEBRANCH": "refs/heads/main", "BUILD_BUILDNUMBER": "1.1.1"},
        )
    assert config.use_build_number == True
    assert config.all_component_version == "1.1.1"
    assert (
        "The build number 1.1.1 overwrites the value of all_component_version 2.2.2"
        in caplog.text
    )


def test_both_args_and_file(tmp_path):
    config_path = tmp_path / "aml-build-configuration.yml"
    config_path.write_text("verbose: False")

    args = [
        "--configuration-file",
        str(config_path),
        "--component-specification-glob",
        "*.yaml",
    ]

    config = load_configuration_from_args_and_env(
        args, {"BUILD_SOURCEBRANCH": "refs/heads/main"}
    )

    assert config.configuration_file == str(config_path)
    assert config.component_specification_glob == "*.yaml"
    assert config.verbose == False


def test_configuration_file_alone_works_fine(tmp_path):
    config_path = tmp_path / "aml-build-configuration.yml"
    config_path.write_text("use_build_number: True\nall_component_version: 0.0.0")

    args = ["--configuration-file", str(config_path)]

    # Assert: does not raise.
    config = load_configuration_from_args_and_env(
        args, {"BUILD_SOURCEBRANCH": "refs/heads/main", "BUILD_BUILDNUMBER": "x.x.x"}
    )

    assert config.use_build_number == True
    assert config.all_component_version == "x.x.x"


def test_load_configuration_from_args_and_env_deprecation(caplog, tmp_path):
    config_path = tmp_path / "aml-build-configuration.yml"
    config_path.write_text("fail_if_version_exists: True")
    args1 = ["--allow-duplicate-versions"]
    args2 = ["--allow-duplicate-versions", "--fail-if-version-exists"]
    args3 = ["--allow-duplicate-versions", "--configuration-file", str(config_path)]
    args4 = ["--configuration-file", str(config_path)]

    with pytest.warns(UserWarning) as record:
        config1 = load_configuration_from_args_and_env(args1, {})

    assert config1.fail_if_version_exists == False
    warning_message = [str(i.message) for i in record]
    assert (
        "We recommend against using the parameter allow_duplicate_versions. Please specify fail_if_version_exists instead."
        in warning_message
    )

    with pytest.raises(ValueError):
        config2 = load_configuration_from_args_and_env(args2, {})
        assert config2.fail_if_version_exists

    with pytest.raises(ValueError):
        config3 = load_configuration_from_args_and_env(args3, {})
        assert config3.fail_if_version_exists

    config4 = load_configuration_from_args_and_env(args4, {})
    assert config4.fail_if_version_exists


def test_configuration_path_does_not_exist(capsys, tmp_path):
    config_path = tmp_path / "file_does_not_exist.yaml"
    args = [
        "--configuration-file",
        str(config_path),
        "--component-specification-glob",
        "*.yaml",
    ]
    config = load_configuration_from_args_and_env(args, {})
    captured = capsys.readouterr()

    assert config.component_specification_glob == "*.yaml"
    assert config.configuration_file == str(config_path)
    assert "ERROR: the configuration file path provided" in captured.out


def test_load_configuration_from_env_and_override_behaviour(capsys, tmp_path):
    config_file_path = tmp_path / "aml-build-configuration.yml"
    config_file_path.write_text(
        "fail_if_version_exists: True\nactivation_method: smart\nverbose: False"
    )
    cli_args = [
        "--verbose",
        "--signing-mode",
        "aether",
        "--configuration-file",
        str(config_file_path),
    ]
    env = {"ACTIVATION_METHOD": "all", "signing_mode": "aml"}

    config = load_configuration_from_args_and_env(cli_args, env)
    captured = capsys.readouterr()

    # Testing loading configuration from env is successful
    assert (
        "Merge the config in the environment variables with the config in the command line."
        in captured.out
    )
    assert (
        "Load the config in the environment variables: {'activation_method': 'all', 'signing_mode': 'aml'}"
        in captured.out
    )

    # Testing cli overrides env
    assert config.signing_mode == "aether"

    # Testing env overrides config file
    assert config.activation_method == "all"

    # Testing cli overrides config file
    assert config.verbose is True

    # Testing config file overrides default config
    assert config.fail_if_version_exists is True
