# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tooling for making it easy to create Azure DevOps build pipelines for
validating, "building", signing, and registering components in eyes-off
Azure Machine Learning workspaces.
"""

try:
    from .commands import prepare, register
except ImportError as error:
    raise ImportError(f"{error.msg}. Please install using `pip install shrike[build]`.")
