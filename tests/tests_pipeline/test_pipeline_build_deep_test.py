# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for deep testing parameters built inside a pipeline.
"""
import os
import pytest
import sys
from unittest.mock import patch

from shrike.pipeline.testing.pipeline_class_test import (
    deeptest_graph_comparison,
)

from .sample.pipeline.passthrough_test import CanaryPipelineStatsPassthrough
from .sample.pipeline.multinode_training_test import MultiNodeTrainingPipeline
from .sample.pipeline.spark_hello import SparkHelloPipeline
from .sample.pipeline.convert_tsv_to_ss_test import ConvertTsvToSSPipeline


@pytest.fixture()
def pipeline_config_path():
    """Locates the pipeline config folder for unit tests.

    Returns:
        str: path to config file in temporary folder
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample", "conf")


def test_spark_hello_build_deep_test(pipeline_config_path):
    """Tests the graph by running the main function itself (which does .validate())"""
    # path where to generate the tested graph
    json_export_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "spark_hello_graph.json"
    )

    # arguments for the main function
    testargs = [
        "prog",
        "--config-dir",
        pipeline_config_path,
        "--config-name",
        "pipelines/spark_hello_test",
        "module_loader.use_local='*'",
        "+run.disable_telemetry=True",
        f"+run.export={json_export_path}",
    ]
    # will do everything except submit :)
    with patch.object(sys, "argv", testargs):
        pipeline_instance = SparkHelloPipeline.main()

    # checks if pipeline instance is actually returned
    assert (
        pipeline_instance is not None
    ), "main function should return a pipeline instance"

    # checks the exported file in temp dir
    assert os.path.isfile(json_export_path), "main function should generate a json file"

    deeptest_graph_comparison(
        # exported graph
        json_export_path,
        # reference graph
        os.path.join(
            os.path.dirname(__file__), "data", "spark_hello_export_reference.json"
        ),
    )


def test_stats_passthrough_build_deep_test(pipeline_config_path):
    """Tests the graph by running the main function itself (which does .validate())"""
    # path where to generate the tested graph
    json_export_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stats_passthrough_graph.json"
    )

    # arguments for the main function
    testargs = [
        "prog",
        "--config-dir",
        pipeline_config_path,
        "--config-name",
        "pipelines/passthrough_test_uuid",
        "module_loader.use_local='*'",
        "+run.disable_telemetry=True",
        f"+run.export={json_export_path}",
    ]
    # will do everything except submit :)
    with patch.object(sys, "argv", testargs):
        pipeline_instance = CanaryPipelineStatsPassthrough.main()

    # checks if pipeline instance is actually returned
    assert (
        pipeline_instance is not None
    ), "main function should return a pipeline instance"

    deeptest_graph_comparison(
        # exported graph
        json_export_path,
        # reference graph
        os.path.join(
            os.path.dirname(__file__), "data", "stats_passthrough_export_reference.json"
        ),
    )


def test_multinode_training_build_deep_test(pipeline_config_path):
    """Tests the graph by running the main function itself (which does .validate())"""
    # path where to generate the tested graph
    json_export_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multinode_training_graph.json"
    )

    # arguments for the main function
    testargs = [
        "prog",
        "--config-dir",
        pipeline_config_path,
        "--config-name",
        "pipelines/multinode_training_test",
        "module_loader.use_local='*'",
        "+run.disable_telemetry=True",
        f"+run.export={json_export_path}",
    ]
    # will do everything except submit :)
    with patch.object(sys, "argv", testargs):
        pipeline_instance = MultiNodeTrainingPipeline.main()

    # checks if pipeline instance is actually returned
    assert (
        pipeline_instance is not None
    ), "main function should return a pipeline instance"

    deeptest_graph_comparison(
        # exported graph
        json_export_path,
        # reference graph
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "multinode_training_export_reference.json",
        ),
    )


def test_convert_tsv_to_ss(pipeline_config_path):
    """Tests the graph by running the main function itself (which does .validate())"""
    # path where to generate the tested graph
    json_export_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "convert_tsv_to_ss_test.json"
    )

    # arguments for the main function
    testargs = [
        "prog",
        "--config-dir",
        pipeline_config_path,
        "--config-name",
        "pipelines/convert_tsv_to_ss_test",
        "module_loader.use_local='*'",
        f"+run.export={json_export_path}",
    ]
    # will do everything except submit :)
    with patch.object(sys, "argv", testargs):
        pipeline_instance = ConvertTsvToSSPipeline.main()

    # checks if pipeline instance is actually returned
    assert (
        pipeline_instance is not None
    ), "main function should return a pipeline instance"

    deeptest_graph_comparison(
        # exported graph
        json_export_path,
        # reference graph
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "convert_tsv_to_ss_reference.json",
        ),
    )
