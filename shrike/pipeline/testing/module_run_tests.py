# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for testing if run.py is aligned with module specification:

> Status: this code relates to the _recipe_ and is a _proposition_
"""
import os
import sys
from pathlib import Path
import traceback
import argparse
from collections import namedtuple
import warnings
import pytest

from ruamel import yaml
from azure.ml.component._core._component_definition import CommandComponentDefinition

from .importer import import_and_test_class
from .importer import dynamic_import_module
from .components import component_spec_yaml_exists_and_is_parsable
from .components import component_uses_private_acr
from .components import component_uses_private_python_feed
from .components import component_run_py_import
from .components import component_run_get_arg_parser
from .components import if_arguments_from_component_spec_match_script_argparse


def module_spec_yaml_exists_and_is_parsable(module):
    """Tests the presence of module specifications in yaml (and return it)"""
    return component_spec_yaml_exists_and_is_parsable(
        os.path.join(module, "module_spec.yaml")
    )


def module_uses_private_acr(module, acr_url):
    """Tests base image in private ACR"""
    component_uses_private_acr(os.path.join(module, "module_spec.yaml"), acr_url)


def module_uses_private_python_feed(module, feed_url):
    """Tests private python feed referenced in conda"""
    component_uses_private_python_feed(
        os.path.join(module, "module_spec.yaml"), feed_url
    )


def run_py_import(module):
    """Try importing run.py, just to check if basic script passes syntax/imports checks"""
    component_run_py_import(os.path.join(module, "module_spec.yaml"))


def module_run_get_arg_parser(module):
    """Tests is module run.py has function get_arg_parser(parser)"""
    component_run_get_arg_parser(os.path.join(module, "module_spec.yaml"))


def if_arguments_from_module_spec_match_script_argparse(module):
    """Tests alignment between module_spec arguments and script parser arguments"""
    if_arguments_from_component_spec_match_script_argparse(
        os.path.join(module, "module_spec.yaml")
    )
