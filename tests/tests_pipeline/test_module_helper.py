# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for testing module_helper.py:

> Status: this code relates to the _recipe_ and is a _proposition_
"""
import pytest
from shrike.pipeline.module_helper import (
    AMLModuleLoader,
    module_loader_config,
    _check_use_local_syntax_valid,
)
from pathlib import Path
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "use_local,expected",
    [
        ("*", True),
        ("KEY1, KEY2", True),
        ("!KEY1, !KEY2", True),
        ("!KEY1, KEY2", False),
        ("KEY1, !KEY2", False),
    ],
)
def test_check_use_local_syntax_valid(use_local, expected):
    use_local_list = [x.strip() for x in use_local.split(",")]
    assert _check_use_local_syntax_valid(use_local_list) == expected


@pytest.mark.parametrize(
    "use_local,use_local_except_for,module_key,expected",
    [
        ("*", None, "KEY1", True),
        ("KEY1, KEY2", False, "KEY1", True),
        ("KEY1, KEY2", False, "KEY3", False),
        ("!KEY1, !KEY2", True, "KEY3", True),
        ("!KEY1, !KEY2", True, "KEY1", False),
    ],
)
def test_is_local(use_local, use_local_except_for, module_key, expected):
    # Load the testing configuration YAML file
    config = OmegaConf.load(Path(__file__).parent / "data/test_configuration.yaml")
    OmegaConf.update(config, "module_loader.use_local", use_local)

    module_loader = AMLModuleLoader(config)
    assert module_loader.use_local_except_for == use_local_except_for
    assert module_loader.is_local(module_key) == expected
