# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from shrike import compliant_logging
from shrike.compliant_logging.constants import DataCategory
from shrike.compliant_logging.logging import CompliantLogger, get_aml_context
from shrike.compliant_logging.exceptions import PublicRuntimeError
from pathlib import Path
import logging
import pytest
import re
import sys
import vaex
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    FloatType,
    IntegerType,
    StringType,
)

from shrike._core import stream_handler


def test_basic_config():
    logging.warning("before basic config")

    logging.basicConfig()
    logging.warning("warning from test_basic_config")

    log = logging.getLogger("foo")
    log.warning("warning from foo logger")


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
def test_data_category_and_log_info_works_as_expected(level):
    compliant_logging.enable_compliant_logging()

    log = logging.getLogger()
    log.setLevel(level.upper())

    assert isinstance(log, compliant_logging.logging.CompliantLogger)

    with stream_handler(
        log, format="%(prefix)s%(levelname)s:%(name)s:%(message)s"
    ) as context:
        func = getattr(log, level)
        func("PRIVATE")
        func("public", category=DataCategory.PUBLIC)
        logs = str(context)

    assert re.search(r"^SystemLog\:.*public$", logs, flags=re.MULTILINE)
    assert not re.search(r"^SystemLog\:.*\:PRIVATE", logs, flags=re.MULTILINE)


def test_non_category_aware_logging_works_as_expected():
    compliant_logging.enable_compliant_logging()

    log = logging.getLogger()
    extra = {"test_name": "", "test_id": ""}
    assert isinstance(log, compliant_logging.logging.CompliantLogger)
    with stream_handler(
        log, "%(test_name)s:%(test_id)s %(prefix)s%(levelname)s:%(name)s:%(message)s"
    ) as context:
        log.log(1, "message", extra={"test_name": "Test", "test_id": 1})
        log.debug("message", extra={"test_name": "Test2", "test_id": 0})
        log.info("message", extra=extra)
        log.warning("message", extra={"test_name": "My", "test_id": "a"})
        try:
            1 / 0
        except Exception as e:
            logging.error(
                "Error at division",
                exc_info=e,
                stack_info=True,
                extra={"test_name": "Test", "test_id": 1},
            )
        log.critical("message", extra=extra, stack_info=True)
        logs = str(context)

    assert re.search(r"^Test:1 Level 1:root:message$", logs, flags=re.MULTILINE)
    assert re.search(r"^Test2:0 DEBUG:root:message$", logs, flags=re.MULTILINE)
    assert re.search(r"^: INFO:root:message$", logs, flags=re.MULTILINE)
    assert re.search(r"^My:a WARNING:root:message$", logs, flags=re.MULTILINE)
    assert re.search(
        r"^Test:1 ERROR:root:Error at division\nTraceback(.*\n){4}Stack",
        logs,
        flags=re.MULTILINE,
    )
    assert re.search(r"^: CRITICAL:root:message\nStack", logs, flags=re.MULTILINE)


@pytest.mark.parametrize("exec_type,message", [(ArithmeticError, "1+1 != 3")])
def test_exception_works_as_expected(exec_type, message):
    compliant_logging.enable_compliant_logging()
    log = logging.getLogger()
    assert isinstance(log, compliant_logging.logging.CompliantLogger)

    with stream_handler(log, "%(prefix)s%(levelname)s:%(name)s:%(message)s") as context:
        try:
            raise exec_type(message)
        except exec_type:
            log.error("foo", category=DataCategory.PUBLIC)
        logs = str(context)

    assert re.search(r"^SystemLog\:.*foo$", logs, flags=re.MULTILINE)


def test_all_the_stuff():
    compliant_logging.enable_compliant_logging()
    log = logging.getLogger("foo")
    log.info("public", category=DataCategory.PUBLIC)
    log.info("PRIVATE", category=DataCategory.PRIVATE)

    log.info("PRIVATE2")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python >= 3.8")
def test_enable_compliant_logging_sets_force():
    # Pytest adds handlers to the root logger by default.
    initial_handlers = list(logging.root.handlers)

    compliant_logging.enable_compliant_logging()

    assert len(logging.root.handlers) == 1
    assert all(h not in logging.root.handlers for h in initial_handlers)


def test_warn_if_root_handlers_already_exist(capsys):
    # Pytest adds handlers to the root logger by default.

    compliant_logging.enable_compliant_logging()

    # https://docs.pytest.org/en/stable/capture.html
    stderr = capsys.readouterr().err
    assert "SystemLog:The root logger already has handlers set!" in stderr


def test_deprecated_enable_confidential_logging(capsys):
    """Pytest the pending deprecation of enable_confidential_logging"""

    compliant_logging.enable_confidential_logging()

    # https://docs.pytest.org/en/stable/capture.html
    stderr = capsys.readouterr().err
    assert (
        "SystemLog: The function enable_confidential_logging() is on the way "
        "to deprecation. Please use enable_compliant_logging() instead." in stderr
    )


def test_get_aml_context():
    """Pytest CompliantLogger._get_aml_context"""
    assert (
        compliant_logging.logging.CompliantLogger(
            name="test_get_aml_context"
        )._get_aml_context()
        is get_aml_context()
    )


def test_logging_aml_metric_single_value():
    """Pytest CompliantLogger.metric_value"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()
    log.metric_value(name="test_log_value", value=0, category=DataCategory.PUBLIC)
    with stream_handler(log, "") as context:
        log.metric_value(name="test_log_value", value=0, category=DataCategory.PRIVATE)
        logs = str(context)
    assert "NumbericMetric  | test_log_value:None | 0" in logs


def test_logging_aml_metric_image():
    """Pytest CompliantLogger.metric_image"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()
    log.metric_image(
        name="dummy_image",
        path=str(Path(__file__).parent.resolve() / "data/dummy_image.png"),
        category=DataCategory.PUBLIC,
    )

    log.metric_image(
        path=str(Path(__file__).parent.resolve() / "data/dummy_image.png"),
        category=DataCategory.PUBLIC,
    )

    with stream_handler(log, "") as context:
        log.metric_image(
            name="test_log_image",
            path=str(Path(__file__).parent.resolve() / "data/dummy_image.png"),
            category=DataCategory.PRIVATE,
        )
        logs = str(context)

    assert "Unable to log image metric test_log_image as private, skipping." in logs


def test_logging_aml_metric_list_tuple():
    """Pytest CompliantLogger.metric_list"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()
    log.metric_list(
        name="test_log_list", value=[1, 2, 3, 4], category=DataCategory.PUBLIC
    )
    log.metric_list(
        name="test_log_tuple", value=("1", "2", "test"), category=DataCategory.PUBLIC
    )
    with stream_handler(log, "") as context:
        log.metric_list(
            name="test_log_empty_list", value=[], category=DataCategory.PUBLIC
        )
        log.metric_list(name="test_log_tupe_private", value=("1", "2", "test", None))
        logs = str(context)
    assert "List Value for Metric test_log_empty_list is empty. Skipping." in logs
    assert "ListMetric      | test_log_tupe_private | ['1', '2', 'test', None]" in logs


def test_logging_aml_metric_row():
    """Pytest COmpliantLogger.metric_row"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    log.metric_row(
        name="test_row",
        description="stats",
        category=DataCategory.PUBLIC,
        total_size=100,
        file_count=200,
    )

    with stream_handler(log, "") as context:
        log.metric_row(
            name="test_row",
            description="stats",
            category=DataCategory.PRIVATE,
            total_size=100,
            file_count=200,
        )
        logs = str(context)
    assert "RowMetric      | test_row | total_size:100 | file_count:200" in logs


def test_logging_aml_metric_table():
    """Pytest CompliantLogger.metric_table"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    test_table1 = {"name": ["James", "Robert", "Michael"], "number": [2, 3, 1, 5]}
    test_table2 = {"name": ["James", "Robert", "Michael"], "number": 2}
    test_table3 = {"name": 2, "number": 4}
    test_table4 = {"name": ["James", "Robert", "Michael"], "number": [2, 3, None]}

    log.metric_table(
        name="test_table1", value=test_table1, category=DataCategory.PUBLIC
    )
    with stream_handler(log, "") as context:
        log.metric_table(name="test_table1", value=test_table1)
        logs = str(context)
    assert "TableMetric     | test_table" in logs
    assert "TableMetric     | Index | name            | number" in logs
    assert "TableMetric     | 00000 | James           | 2              " in logs
    assert "TableMetric     | 00001 | Robert          | 3              " in logs
    assert "TableMetric     | 00002 | Michael         | 1              " in logs
    assert "TableMetric     | 00003 |                 | 5              " in logs

    # Checking empty value
    with stream_handler(log, "") as context:
        log.metric_table(name="empty_input", value={}, category=DataCategory.PUBLIC)
        logs = str(context)
    assert "Dictionary Value for Metric empty_input is empty. Skipping." in logs

    # Checking mixed types
    with stream_handler(log, "") as context:
        log.metric_table(
            name="mixed_type", value=test_table2, category=DataCategory.PUBLIC
        )
        logs = str(context)
    assert (
        "The provided dictionary for metric mixed_type appears to be unstructured!"
        in logs
    )

    log.metric_table(
        name="test_table3", value=test_table3, category=DataCategory.PUBLIC
    )

    with stream_handler(log, "") as context:
        log.metric_table(name="test_table4", value=test_table4)
        logs = str(context)
    assert "TableMetric     | test_table" in logs
    assert "TableMetric     | Index | name            | number" in logs
    assert "TableMetric     | 00000 | James           | 2              " in logs
    assert "TableMetric     | 00001 | Robert          | 3              " in logs
    assert "TableMetric     | 00002 | Michael         |                " in logs


def test_logging_aml_metric_residual():
    """Pytest CompliantLogger.metric_residual"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    # Testing vaex dataframe
    test_input1 = vaex.from_arrays(
        x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=[1.1, 2.0, 3.2, 3.8, 5, 6, 7, 8, 9, 10]
    )
    log.metric_residual(
        name="test_log_residual",
        value=test_input1,
        col_predict="x",
        col_target="y",
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_residual(
            name="test_log_residual",
            value=test_input1,
            category=DataCategory.PUBLIC,
        )
    with stream_handler(log, "") as context:
        log.metric_residual(
            name="test_log_residual",
            value=test_input1,
            col_predict="x",
            col_target="y",
        )
        logs = str(context)
    assert "Logging Residuals to text is not yet implemented" in logs

    # Testing spark dataframe
    test_input2 = (
        SparkSession.builder.appName("SparkUnitTests")
        .getOrCreate()
        .createDataFrame(
            data=[(1.0, 1.1), (2.0, 2.0), (3.0, 3.1), (4.0, 3.8), (5.0, 5.2)],
            schema=StructType(
                [
                    StructField("prediction", FloatType(), True),
                    StructField("target", FloatType(), True),
                ]
            ),
        )
    )
    log.metric_residual(
        name="test_log_residual",
        value=test_input2,
        col_predict="prediction",
        col_target="target",
        category=DataCategory.PUBLIC,
    )

    # Testing panda dataframe
    test_input3 = pd.DataFrame(
        [[1.0, 1.1], [2.0, 2.0], [3.0, 3.1], [4.0, 3.8], [100.0, 5.2]],
        columns=["prediction", "target"],
    )
    log.metric_residual(
        name="test_log_residual",
        value=test_input3,
        col_predict="prediction",
        col_target="target",
        category=DataCategory.PUBLIC,
    )

    # Testing un-supported data type
    test_input4 = [1, 2, 3, 4]
    with pytest.raises(PublicRuntimeError):
        log.metric_residual(
            name="test_log_residual",
            value=test_input4,
            col_predict="x",
            col_target="y",
            category=DataCategory.PUBLIC,
        )

    # Testing dict
    test_input5 = {
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "prediction": [1.1, 2.0, 3.2, 3.8, 5, 6, 7, 8, 9, 10],
    }
    test_input6 = {
        "schema_type": "residuals",
        "schema_version": "1.0.0",
        "data": {
            "bin_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
            "bin_counts": [0.0, 0.0, 0.0, 0.1],
        },
    }
    test_input7 = {
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "prediction": [1.1],
    }
    log.metric_residual(
        name="test_log_residual",
        value=test_input5,
        col_predict="prediction",
        col_target="target",
        bin_edges=3,
        category=DataCategory.PUBLIC,
    )
    log.metric_residual(
        name="test_log_residual",
        value=test_input6,
        col_predict="prediction",
        col_target="target",
        bin_edges=4,
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_residual(
            name="test_log_residual",
            value=test_input7,
            col_predict="prediction",
            col_target="target",
            category=DataCategory.PUBLIC,
        )


def test_logging_aml_metric_predictions():
    """Pytest CompliantLogger.metric_predictions"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    # Testing vaex dataframe
    test_input1 = vaex.from_arrays(
        x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=[1.1, 2.0, 3.2, 3.8, 5, 6, 7, 8, 9, 10]
    )
    log.metric_predictions(
        name="test_log_predictions",
        value=test_input1,
        col_predict="x",
        col_target="y",
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_predictions(
            name="test_log_predictions",
            value=test_input1,
            category=DataCategory.PUBLIC,
        )
    with stream_handler(log, "") as context:
        log.metric_predictions(
            name="test_log_predictions",
            value=test_input1,
            col_predict="x",
            col_target="y",
        )
        logs = str(context)
    assert "Logging Predictions to text is not yet implemented" in logs

    # Testing spark dataframe
    test_input2 = (
        SparkSession.builder.appName("SparkUnitTests")
        .getOrCreate()
        .createDataFrame(
            data=[(1.0, 1.1), (2.0, 2.0), (3.0, 3.1), (4.0, 3.8), (5.0, 5.2)],
            schema=StructType(
                [
                    StructField("prediction", FloatType(), True),
                    StructField("target", FloatType(), True),
                ]
            ),
        )
    )
    log.metric_predictions(
        name="test_log_predictions",
        value=test_input2,
        col_predict="prediction",
        col_target="target",
        category=DataCategory.PUBLIC,
    )

    # Testing panda dataframe
    test_input3 = pd.DataFrame(
        [[1.0, 1.1], [2.0, 2.0], [3.0, 3.1], [4.0, 3.8], [100.0, 5.2]],
        columns=["prediction", "target"],
    )
    log.metric_predictions(
        name="test_log_predictions",
        value=test_input3,
        col_predict="prediction",
        col_target="target",
        category=DataCategory.PUBLIC,
    )

    # Testing un-supported data type
    test_input4 = [1, 2, 3, 4]
    with pytest.raises(PublicRuntimeError):
        log.metric_predictions(
            name="test_log_predictions",
            value=test_input4,
            col_predict="x",
            col_target="y",
            category=DataCategory.PUBLIC,
        )

    # Testing dict
    test_input5 = {
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "prediction": [1.1, 2.0, 3.2, 3.8, 5, 6, 7, 8, 9, 10],
    }
    test_input6 = {
        "schema_type": "predictions",
        "schema_version": "1.0.0",
        "data": {
            "bin_averages": [1, 2, 3, 4],
            "bin_errors": [0.0, 0.0, 0.0, 0.0],
            "bin_counts": [0, 0, 0, 0],
            "bin_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
    }
    test_input7 = {
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "prediction": [1.1],
    }
    log.metric_predictions(
        name="test_log_predictions",
        value=test_input5,
        col_predict="prediction",
        col_target="target",
        bin_edges=3,
        category=DataCategory.PUBLIC,
    )
    log.metric_predictions(
        name="test_log_predictions",
        value=test_input6,
        col_predict="prediction",
        col_target="target",
        bin_edges=4,
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_predictions(
            name="test_log_predictions",
            value=test_input7,
            col_predict="prediction",
            col_target="target",
            category=DataCategory.PUBLIC,
        )


def test_logging_aml_metric_confusion_matrix():
    """Pytest CompliantLogger.metric_confusion_matrix"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    # Testing vaex dataframe
    test_input1 = vaex.from_arrays(
        x=["cat", "ant", "cat", "cat", "ant", "bird"],
        y=["ant", "ant", "cat", "cat", "ant", "cat"],
    )
    log.metric_confusion_matrix(
        name="test_log_confusion_matrix",
        value=test_input1,
        idx_true="x",
        idx_pred="y",
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_confusion_matrix(
            name="test_log_confusion_matrix",
            value=test_input1,
            category=DataCategory.PUBLIC,
        )
    with stream_handler(log, "") as context:
        log.metric_confusion_matrix(
            name="test_log_confusion_matrix",
            value=test_input1,
            idx_true="x",
            idx_pred="y",
        )
        logs = str(context)
    assert "Logging Confusion Matrices to text is not yet implemented" in logs

    # Testing spark dataframe
    test_input2 = (
        SparkSession.builder.appName("SparkUnitTests")
        .getOrCreate()
        .createDataFrame(
            data=[(1, 1), (2, 2), (1, 2), (2, 2), (1, 2)],
            schema=StructType(
                [
                    StructField("prediction", IntegerType(), True),
                    StructField("target", IntegerType(), True),
                ]
            ),
        )
    )
    log.metric_confusion_matrix(
        name="test_log_confusion_matrix",
        value=test_input2,
        idx_true="target",
        idx_pred="prediction",
        category=DataCategory.PUBLIC,
    )

    # Testing panda dataframe
    test_input3 = pd.DataFrame(
        [[1, 1], [2, 2], [1, 2], [1, 2], [1, 1]],
        columns=["prediction", "target"],
    )
    log.metric_confusion_matrix(
        name="test_log_confusion_matrix",
        value=test_input3,
        idx_true="target",
        idx_pred="prediction",
        category=DataCategory.PUBLIC,
    )

    # Testing un-supported data type
    test_input4 = [1, 2, 3, 4]
    with pytest.raises(PublicRuntimeError):
        log.metric_confusion_matrix(
            name="test_log_confusion_matrix",
            value=test_input4,
            idx_true="target",
            idx_pred="prediction",
            category=DataCategory.PUBLIC,
        )

    # Testing dict
    test_input5 = {
        "target": ["cat", "ant", "cat", "cat", "ant", "bird"],
        "prediction": ["ant", "ant", "cat", "cat", "ant", "cat"],
    }
    test_input6 = {
        "schema_type": "confusion_matrix",
        "schema_version": "1.0.0",
        "data": {
            "class_labels": [1, 2],
            "matrix": [[2, 0], [2, 1]],
        },
    }
    test_input7 = {
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "prediction": [1.1],
    }
    log.metric_confusion_matrix(
        name="test_log_confusion_matrix",
        value=test_input5,
        idx_true="target",
        idx_pred="prediction",
        category=DataCategory.PUBLIC,
    )
    log.metric_confusion_matrix(
        name="test_log_confusion_matrix",
        value=test_input6,
        idx_true="target",
        idx_pred="prediction",
        labels=["cat", "dog"],
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_confusion_matrix(
            name="test_log_confusion_matrix",
            value=test_input7,
            idx_true="target",
            idx_pred="prediction",
            category=DataCategory.PUBLIC,
        )


def test_logging_aml_metric_accuracy_table():
    """Pytest CompliantLogger.metric_accuracy_table"""
    compliant_logging.enable_compliant_logging(use_aml_metrics=True)
    log = logging.getLogger()

    # Testing vaex dataframe
    test_input1 = vaex.from_arrays(x=[0.1, 0.3, 0.7], y=["a", "b", "c"])
    log.metric_accuracy_table(
        name="test_log_accuracy_table",
        value=test_input1,
        col_predict="x",
        col_target="y",
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_accuracy_table(
            name="test_log_accuracy_table",
            value=test_input1,
            category=DataCategory.PUBLIC,
        )
    with stream_handler(log, "") as context:
        log.metric_accuracy_table(
            name="test_log_accuracy_table",
            value=test_input1,
            col_predict="x",
            col_target="y",
        )
        logs = str(context)
    assert "Logging Accuracy Tables to text is not yet implemented" in logs

    # Testing spark dataframe
    test_input2 = (
        SparkSession.builder.appName("SparkUnitTests")
        .getOrCreate()
        .createDataFrame(
            data=[(0.1, "a"), (0.3, "b"), (0.6, "c")],
            schema=StructType(
                [
                    StructField("probability", FloatType(), True),
                    StructField("labels", StringType(), True),
                ]
            ),
        )
    )
    log.metric_accuracy_table(
        name="test_log_accuracy_table",
        value=test_input2,
        col_predict="probability",
        col_target="labels",
        category=DataCategory.PUBLIC,
    )

    # Testing panda dataframe
    test_input3 = pd.DataFrame(
        [[0.1, "a"], [0.3, "b"], [0.6, "c"]],
        columns=["probability", "labels"],
    )
    log.metric_accuracy_table(
        name="test_log_accuracy_table",
        value=test_input3,
        col_predict="probability",
        col_target="labels",
        category=DataCategory.PUBLIC,
    )

    # Testing un-supported data type
    test_input4 = [1, 2, 3, 4]
    with pytest.raises(PublicRuntimeError):
        log.metric_accuracy_table(
            name="test_log_accuracy_table",
            value=test_input4,
            col_predict="probability",
            col_target="labels",
            category=DataCategory.PUBLIC,
        )

    # Testing dict
    test_input5 = {
        "probability": [0.1, 0.3, 0.7],
        "labels": ["a", "b", "c"],
    }
    test_input6 = {
        "schema_type": "accuracy_table",
        "schema_version": "1.0.1",
        "data": {
            "probability_tables": [
                [[1, 2, 0, 0], [0, 2, 0, 1], [0, 1, 1, 1], [0, 0, 2, 1], [0, 0, 2, 1]],
                [[1, 2, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 2, 1], [0, 0, 2, 1]],
                [[1, 2, 0, 0], [1, 1, 1, 0], [1, 0, 2, 0], [0, 0, 2, 1], [0, 0, 2, 1]],
            ],
            "precentile_tables": [
                [[1, 2, 0, 0], [0, 2, 0, 1], [0, 2, 0, 1], [0, 0, 2, 1], [0, 0, 2, 1]],
                [[1, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 2, 1], [0, 0, 2, 1]],
                [[1, 0, 2, 0], [0, 0, 2, 1], [0, 0, 2, 1], [0, 0, 2, 1], [0, 0, 2, 1]],
            ],
            "probability_thresholds": [0.0, 0.25, 0.5, 0.75, 1.0],
            "percentile_thresholds": [0.0, 0.01, 0.24, 0.98, 1.0],
            "class_labels": ["a", "b", "c"],
        },
    }
    test_input7 = {
        "probability": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "labels": [1.1],
    }
    log.metric_accuracy_table(
        name="test_log_accuracy_table",
        value=test_input5,
        col_predict="probability",
        col_target="labels",
        category=DataCategory.PUBLIC,
    )
    log.metric_accuracy_table(
        name="test_log_accuracy_table",
        value=test_input6,
        col_predict="probability",
        col_target="labels",
        category=DataCategory.PUBLIC,
    )
    with pytest.raises(PublicRuntimeError):
        log.metric_accuracy_table(
            name="test_log_accuracy_table",
            value=test_input7,
            col_predict="probability",
            col_target="labels",
            category=DataCategory.PUBLIC,
        )


def test_convert_obj():
    """Pytest CompliantLogger._convert_obj"""
    logger = CompliantLogger(name="")
    assert logger._convert_obj("tests", category=DataCategory.PUBLIC) == "tests"

    assert "Spark DataFrame (Row Count: 5 / Column Count: 2)" in logger._convert_obj(
        (
            SparkSession.builder.appName("SparkUnitTests")
            .getOrCreate()
            .createDataFrame(
                data=[(1, 1), (2, 2), (1, 2), (2, 2), (1, 2)],
                schema=StructType(
                    [
                        StructField("prediction", IntegerType(), True),
                        StructField("target", IntegerType(), True),
                    ]
                ),
            )
        ),
        category=DataCategory.PUBLIC,
    )

    assert "Vaex DataFrame (Row Count: 5 / Column Count: 2)" in logger._convert_obj(
        vaex.from_arrays(x=np.arange(5), y=np.arange(5) ** 2)
    )
    test_df = pd.DataFrame(
        [["Tom", 10], ["Nick", 15], ["John", 14]], columns=["Name", "Age"]
    )
    assert "Pandas DataFrame (Row Count: 3 / Column Count: 2)" in logger._convert_obj(
        test_df
    )

    test_series = pd.Series([1, 2, 3, 4, 5])
    assert "Pandas Series (Row Count: 5)" in logger._convert_obj(test_series)

    test_ndarray = np.array([1, 2, 3, 4, 5])
    assert "Numpy Array (Shape: (5,))" in logger._convert_obj(test_ndarray)

    assert "List (Count: 5) | range(0, 5)" in logger._convert_obj(
        range(5), category=DataCategory.PUBLIC
    )
    assert "List (Count: 10) | range(0, 10)..." in logger._convert_obj(
        range(10), category=DataCategory.PUBLIC
    )
