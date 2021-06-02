# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
import pytest
import sys
from unittest import mock


from shrike.build.core import command_line
from shrike.build.core.configuration import Configuration


class CommandForExecution(command_line.Command):
    """
    Stub subclass of `Command` for testing `execute_command`.
    """

    def __init__(self, timeout: int = 10):
        super().__init__()
        self.config = Configuration(shell_command_timeout_in_seconds=timeout)

    def run_with_config(self):
        pass


@pytest.mark.parametrize(
    "command,expected",
    [
        (["python", "--version"], f"Python {sys.version_info.major}"),
    ],
)
def test_execute_command_basic_sanity_check(command, expected, caplog):
    with caplog.at_level("INFO"):
        res = CommandForExecution().execute_command(command)

    assert expected in caplog.text
    assert res


def test_execute_command_fails_explicitly_if_attempt_run_azure_cli():
    with pytest.raises(ValueError):
        CommandForExecution().execute_command(["az", "--version"])


@pytest.mark.order(-1)
@pytest.mark.parametrize(
    "command,expected",
    [
        ("--version", "azure-cli"),
        ("extension list", "["),
        ("find --help", "I'm an AI robot"),
    ],
)
def test_execute_azure_cli_command_basic_sanity_check(command, expected, caplog):
    with caplog.at_level("INFO"):
        # Allow standard error because Azure CLI may warn that upgrades exist.
        res = CommandForExecution(20).execute_azure_cli_command(
            command, stderr_is_failure=False
        )

    assert expected in caplog.text
    assert res


@pytest.mark.parametrize(
    "command,expected",
    [
        (["pytest", "--version"], f"pytest {pytest.__version__}"),
    ],
)
def test_execute_command_stderr_and_zero_exit_code_fails(command, expected, caplog):
    res = CommandForExecution().execute_command(command)

    assert expected in caplog.text
    assert not res


@pytest.mark.parametrize(
    "command,timeout,expected",
    [
        (["pwsh", "-c", "write hi; start-sleep 10"], 8, "hi"),
    ],
)
def test_execute_command_fails_as_expected_with_timeout(
    command, timeout, expected, caplog
):
    with caplog.at_level("INFO"):
        res = CommandForExecution(timeout).execute_command(command)

    assert expected in caplog.text
    assert not res


@pytest.mark.parametrize("line", ["sample log line"])
def test_log_emphasize(caplog, line):

    with caplog.at_level("INFO"):
        with command_line._LogEmphasize(line):
            pass

    assert caplog.text.count(line) == 2


def test_command_returns_nonzero_exit_code_if_error(caplog):
    class DummyCommand(command_line.Command):
        def __init__(self):
            super().__init__()

        def run_with_config(self):
            self.register_error("error")

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=argparse.Namespace()
    ):
        with pytest.raises(SystemExit):
            DummyCommand().run()

    assert "errors!" in caplog.text


@pytest.mark.parametrize(
    "arm_id,sub_id,rg,workspace",
    [
        ("/_/sub/_/rg/_/_/_/ws", "sub", "rg", "ws"),
        (
            "/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-wus2",
            "48bbc269-ce89-4f6f-9a12-c6f91fcb772d",
            "aml1p-rg",
            "aml1p-ml-wus2",
        ),
    ],
)
def test_parse_workspace_arm_id_works_as_expected(arm_id, sub_id, rg, workspace):
    (
        actual_sub_id,
        actual_rg,
        actual_ws,
    ) = CommandForExecution().parse_workspace_arm_id(arm_id)
    assert actual_sub_id == sub_id
    assert actual_rg == rg
    assert actual_ws == workspace


def current_drive() -> str:
    """
    Not super robust, but needed since Azure DevOps jobs run in the `D:\\` drive
    of their machines.
    """
    return __file__.split(":")[0]


@pytest.mark.parametrize(
    "path,dir,expected",
    [
        (r"\\a\\b", False, f"{current_drive()}:/a/b"),
        (r"\\a\\b", True, f"{current_drive()}:/a/b/"),
        (Path("\\a"), True, f"{current_drive()}:/a/"),
    ],
)
def test_normalize_path(path, dir, expected):
    rv = CommandForExecution().normalize_path(path, dir)

    assert expected == rv


def test_workspace_not_configured(caplog):
    with caplog.at_level("ERROR"):
        CommandForExecution().attach_workspace()
    assert (
        "Workspace is not configured. Please update in your PATH/TO/CONFIGURATION_FILE using `--configuration-file PATH/TO/CONFIGURATION_FILE`."
        in caplog.text
    )
