# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import platform
import logging

from shrike.compliant_logging import enable_compliant_logging, provide_system_info
from test_logging import StreamHandlerContext


def test_provide_system_info():
    enable_compliant_logging(prefix="SystemLog:")

    log = logging.getLogger()
    log.setLevel("INFO")

    uname = platform.uname()
    python_version = platform.python_version()

    provide_system_info(logger=log)

    with StreamHandlerContext(log, "") as context:
        provide_system_info(logger=log, library_checks=["shrike"])
        logs_1 = str(context)

    assert "### System Config ###" in logs_1
    assert f"| CPU     = {uname.processor}" in logs_1
    assert f"| Machine = {uname.machine}" in logs_1
    assert f"| System  = {uname.system} ({uname.release})" in logs_1
    assert f"| Python  = {python_version}" in logs_1
    assert "#" * 21 in logs_1
    assert "Library shrike available" in logs_1

    with StreamHandlerContext(log, "") as context:
        provide_system_info(
            logger=log, library_checks=["shrike", "SOME_UNKNOWN_LIBRARY"]
        )
        logs_2 = str(context)

    assert (
        "Library SOME_UNKNOWN_LIBRARY is not found and could not be imported" in logs_2
    )
    assert "Library shrike available" in logs_2
