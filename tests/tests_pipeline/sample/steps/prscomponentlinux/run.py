# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""run.py for prs component"""
import argparse
import os
import ntpath
import time
from azureml_user.parallel_run import EntryScript
from shrike.compliant_logging.constants import DataCategory
from shrike.compliant_logging.exceptions import prefix_stack_trace
from shrike.compliant_logging import enable_compliant_logging
import logging


def get_parser():
    """Parse the command line arguments for merge using argparse
    Returns:
        ArgumentParser: the argument parser instance
    """
    # add arguments that are specific to the component
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)

    return parser


def init():
    global log, args
    enable_compliant_logging()
    log = logging.getLogger(__name__)
    log.info("job started. ", category=DataCategory.PUBLIC)

    parser = get_parser()
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)


def run(input_files):
    resultList = []

    for input_file in input_files:
        output_file = os.path.join(
            EntryScript().output_dir, ntpath.basename(input_file)
        )
        log.info("job for the input file: " + input_file, category=DataCategory.PUBLIC)
        with open(input_file, "r") as infile, open(output_file, "w") as out:
            for line in infile:
                out.write(line + " -  length of line is " + str(len(line)))

        log.info(
            "input file, " + input_file + ", was processed.",
            category=DataCategory.PUBLIC,
        )

        # Pause for a while
        time.sleep(5)
        resultList.append("dummy")

    log.info("job completed.", category=DataCategory.PUBLIC)

    return resultList
