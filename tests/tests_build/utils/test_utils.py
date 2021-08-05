# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pytest
import logging

from shrike.build.utils import utils


@pytest.mark.parametrize(
    "file_name,expected_hash",
    [
        (
            "another_file_for_component1.txt",
            "9E640A18FC586A6D87716F9B4C6728B7023819E58E07C4562E0D2C14DFC3CF5B",
        ),
        (
            "spec.additional_includes",
            "50407DAA1E6DA1D91E1CE88DDDF18B3DFDA62E08B780EC9B2E8642536DD36C05",
        ),
    ],
)
def test_create_SHA_256_hash_of_file_matches_cosmic_build_tool(
    file_name, expected_hash
):
    """
    Cross-check the local "SHA 256 hash" implementation against the "Cosmic"
    build tool. These hashes were obtained from the build:

    https://dev.azure.com/msdata/Vienna/_build/results?buildId=29991141&view=artifacts&pathAsName=false&type=publishedArtifacts
    """
    file_path = str(Path(__file__).parent.parent / "steps/component1" / file_name)
    hash = utils.create_SHA_256_hash_of_file(file_path)

    assert hash == expected_hash


def test_telemetry_logger(caplog):
    """Unit tests for utils class of opencensus azure monitor"""
    telemetry_logger = utils.TelemetryLogger()
    assert (
        telemetry_logger.instrumentation_key == "aaefce9e-d109-4fac-bb9f-8277c68e91ac"
    )
    assert telemetry_logger.enable_telemetry

    telemetry_logger = utils.TelemetryLogger(enable_telemetry=False)
    assert not telemetry_logger.enable_telemetry
    with caplog.at_level("INFO"):
        telemetry_logger.log_trace(message="A unit test message. Please ignore it.")
    assert (
        "Sending trace log messages to application insight has been disabled."
        in caplog.text
    )

    telemetry_logger = utils.TelemetryLogger(enable_telemetry=True)
    telemetry_logger.log_trace(
        message="A unit test message of shrike.build. Please ignore it.",
        level=logging.INFO,
    )
