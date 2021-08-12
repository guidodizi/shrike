# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper code for connecting to AzureML and sharing one workspace accross code.
"""


import argparse
from azureml.core import Workspace
from collections import namedtuple
import logging
import os


log = logging.getLogger(__name__)


CURRENT_AML_WORKSPACE = None


def current_workspace(workspace=None):
    """Sets/Gets the current AML workspace used all accross code.

    Args:
        workspace (azureml.core.Workspace): any given workspace

    Returns:
        azureml.core.Workspace: current (last) workspace given to current_workspace()
    """
    global CURRENT_AML_WORKSPACE
    if workspace:
        CURRENT_AML_WORKSPACE = workspace

    if not CURRENT_AML_WORKSPACE:
        raise Exception(
            "You need to initialize current_workspace() with an AML workspace"
        )

    return CURRENT_AML_WORKSPACE


def add_cli_args(parser):
    """Adds parser arguments for connecting to AzureML

    Args:
        parser (argparse.ArgumentParser): parser to add AzureML arguments to

    Returns:
        argparse.ArgumentParser: that same parser
    """
    parser.add_argument(
        "--aml-subscription-id",
        dest="aml_subscription_id",
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--aml-resource-group",
        dest="aml_resource_group",
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--aml-workspace", dest="aml_workspace_name", type=str, required=False, help=""
    )
    parser.add_argument(
        "--aml-config",
        dest="aml_config",
        type=str,
        required=False,
        help="path to aml config.json file",
    )

    parser.add_argument(
        "--aml-auth",
        dest="aml_auth",
        type=str,
        choices=["azurecli", "msi", "interactive"],
        default="interactive",
    )
    parser.add_argument(
        "--aml-tenant",
        dest="aml_tenant",
        type=str,
        default=None,
        help="tenant to use for auth (default: auto)",
    )
    parser.add_argument(
        "--aml-force",
        dest="aml_force",
        type=lambda x: (
            str(x).lower() in ["true", "1", "yes"]
        ),  # we want to use --aml-force True
        default=False,
        help="force tenant auth (default: False)",
    )

    return parser


def azureml_connect(**kwargs):
    """Calls azureml_connect_cli with an argparse-like structure
    based on keyword arguments"""
    keys = [
        "aml_subscription_id",
        "aml_resource_group",
        "aml_workspace_name",
        "aml_config",
        "aml_auth",
        "aml_tenant",
        "aml_force",
    ]
    aml_args = dict([(k, kwargs.get(k)) for k in keys])

    azureml_argparse_tuple = namedtuple("AzureMLArguments", aml_args)
    aml_argparse = azureml_argparse_tuple(**aml_args)
    return azureml_connect_cli(aml_argparse)


def azureml_connect_cli(args):
    """Connects to an AzureML workspace.

    Args:
        args (argparse.Namespace): arguments to connect to AzureML

    Returns:
        azureml.core.Workspace: AzureML workspace
    """
    if args.aml_auth == "msi":
        from azureml.core.authentication import MsiAuthentication

        auth = MsiAuthentication()
    elif args.aml_auth == "azurecli":
        from azureml.core.authentication import AzureCliAuthentication

        auth = AzureCliAuthentication()
    elif args.aml_auth == "interactive":
        from azureml.core.authentication import InteractiveLoginAuthentication

        auth = InteractiveLoginAuthentication(
            tenant_id=args.aml_tenant, force=args.aml_force
        )
    else:
        auth = None

    if args.aml_config:
        config_dir = os.path.dirname(args.aml_config)
        config_file_name = os.path.basename(args.aml_config)

        aml_ws = Workspace.from_config(
            path=config_dir, _file_name=config_file_name, auth=auth
        )
    else:
        aml_ws = Workspace.get(
            subscription_id=args.aml_subscription_id,
            name=args.aml_workspace_name,
            resource_group=args.aml_resource_group,
            auth=auth,
        )

    log.info("Connected to workspace:")
    log.info(f"\tsubscription: {aml_ws.subscription_id}")
    log.info(f"\tname: {aml_ws.name}")
    log.info(f"\tAzure region: {aml_ws.location}")
    log.info(f"\tresource group: {aml_ws.resource_group}")

    return current_workspace(aml_ws)


def main():
    """Main function (for testing)"""
    parser = argparse.ArgumentParser(description=__doc__)

    group = parser.add_argument_group("AzureML connect arguments")
    add_cli_args(group)

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        log.warning(f"You have provided unknown arguments {unknown_args}")

    return azureml_connect_cli(args)


if __name__ == "__main__":
    main()
