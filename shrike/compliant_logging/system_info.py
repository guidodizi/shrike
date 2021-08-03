# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import platform

from shrike.compliant_logging.constants import DataCategory


def provide_system_info(
    logger: logging.Logger = None,
    library_checks: list = None,
) -> None:
    """
    Provides a list of information about the current system.

    Args:
        logger (logging.Logger, optional): Logger to use. Defaults to None.
        additional_libs (list, optional): List of libraries that should be checked
            for import. Defaults to None.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # retrieve information about the current system
    uname = platform.uname()
    python_version = platform.python_version()
    config_str = "### System Config ###"
    logger.info(config_str, category=DataCategory.PUBLIC)
    logger.info(f"| CPU     = {uname.processor}", category=DataCategory.PUBLIC)
    logger.info(f"| Machine = {uname.machine}", category=DataCategory.PUBLIC)
    logger.info(
        f"| System  = {uname.system} ({uname.release})",
        category=DataCategory.PUBLIC,
    )
    logger.info(f"| Python  = {python_version}", category=DataCategory.PUBLIC)
    logger.info("#" * len(config_str), category=DataCategory.PUBLIC)

    # check for libraries
    if library_checks is not None:
        # iterate through all libs
        for lib in library_checks:
            try:
                __import__(lib)
                logger.info(f"Library {lib} available", category=DataCategory.PUBLIC)
            except ModuleNotFoundError:
                logger.warning(
                    f"Library {lib} is not found and could not be imported",
                    category=DataCategory.PUBLIC,
                )
            except Exception as ex:
                logger.warning(
                    f"Library {lib} could not be imported: {ex}",
                    category=DataCategory.PUBLIC,
                )
