# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from shrike._core import TelemetryLogger


def test_telemetry_logger(caplog):
    """Unit tests for utils class of opencensus azure monitor"""
    telemetry_logger = TelemetryLogger()
    assert (
        telemetry_logger.instrumentation_key == "aaefce9e-d109-4fac-bb9f-8277c68e91ac"
    )
    assert telemetry_logger.enable_telemetry

    telemetry_logger = TelemetryLogger(enable_telemetry=False)
    assert not telemetry_logger.enable_telemetry
    with caplog.at_level("INFO"):
        telemetry_logger.log_trace(message="A unit test message. Please ignore it.")
    assert (
        "Sending trace log messages to application insight has been disabled."
        in caplog.text
    )

    telemetry_logger = TelemetryLogger(enable_telemetry=True)
    telemetry_logger.log_trace(
        message="A unit test message of shrike.pipeline. Please ignore it.",
        level=logging.INFO,
    )
