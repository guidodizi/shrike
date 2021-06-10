# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This is a test AML Module for SmartCompose.
    This copies inputs to outputs and displays some stats.

    Author: Jeff Omhover for Microsoft
    Contact: Jeff.Omhover@microsoft.com
"""
import os
import sys
import argparse
import logging
import traceback
from distutils import dir_util


def get_arg_parser(parser=None):
    """Adds module arguments to a given argument parser (or creates one if none given).

    Args:
        parser (argparse.parser): the argument parser instance

    Returns:
        argparse.ArgumentParser: the argument parser with constructed arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    # those are examples
    group = parser.add_argument_group("Module arguments [specific]")
    group.add_argument(
        "--input_path",
        dest="input_path",
        default=None,
        type=str,
        required=True,
        help="path to input data",
    )
    group.add_argument(
        "--output_path",
        dest="output_path",
        default=None,
        type=str,
        required=True,
        help="path to output data",
    )

    return parser


def run(args):
    """Module run function, this does the actual work

    Args:
        args (argparse.NameSpace): arguments as specified in get_arg_parser()
    """
    # at the end of your run, this utility function will count files and size in the provided path
    # if using --log-in-aml=True inside an AML experiment this will send this in AML
    # IMPORTANT: please make sure you use this in your module
    os.makedirs(args.output_path, exist_ok=True)
    dir_util.copy_tree(args.input_path, args.output_path)


def main(cli_args=None):
    """Main function

    Args:
        cli_args (List[Str]): to force parsing of specific arguments (ex: from unit tests)
    """
    parser = argparse.ArgumentParser(__doc__)

    get_arg_parser(parser)

    # screams an exception and sys.exit() if there's an error
    # sends a warning on logs if there's an unknown arg
    args, _ = parser.parse_known_args(cli_args)

    run(args)


if __name__ == "__main__":
    main()
