# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest
from shrike.spark.spark_net import *
from unittest import mock


def test_get_default_spark_session():
    mock_app_name = mock.MagicMock()
    mock_app = mock.MagicMock()

    # https://stackoverflow.com/a/34668809
    mock_app_name.getOrCreate = lambda: mock_app

    with mock.patch(
        "pyspark.sql.SparkSession.builder.appName", return_value=mock_app_name
    ):
        rv = get_default_spark_session()

    assert mock_app == rv


@pytest.mark.parametrize(
    "py_files,py_file,expected",
    [
        ("a/b/c/d.zip,a/b/e.zip", "d.zip", "a/b/c/d.zip"),
        ("a/b.zip/c.zip,a/b.zip", "b.zip", "a/b.zip"),
    ],
)
def test_full_pyfile_path(py_files, py_file, expected):
    spark = mock.MagicMock()
    spark_conf = {"spark.yarn.dist.pyFiles": py_files}

    # https://stackoverflow.com/a/39457691
    spark.sparkContext.getConf = lambda: spark_conf

    rv = full_pyfile_path(spark, py_file)
    assert expected == rv


@pytest.mark.parametrize("args", [[], [1, 2], ["a", "b", "c"]])
def test_java_args(args):
    """
    https://github.com/dkmiller/pyconfigurableml/blob/master/tests/test_entry.py
    """
    spark = mock.MagicMock()
    gateway = mock.MagicMock()
    gateway.new_array = lambda _, n: [None] * n

    with mock.patch("pyspark.SparkContext._gateway", gateway):
        rv = java_args(spark, args)

    assert rv == args


@pytest.mark.parametrize("zip,binary,args", [("foo.zip", "bar.dll", ["a", "b"])])
def test_run_spark_net_from_known_assembly(zip, binary, args):
    spark = mock.MagicMock()
    spark_conf = {"spark.yarn.dist.pyFiles": zip}

    # https://stackoverflow.com/a/39457691
    spark.sparkContext.getConf = lambda: spark_conf

    main = mock.MagicMock()
    spark._jvm.org.apache.spark.deploy.dotnet.DotnetRunner.main = main

    gateway = mock.MagicMock()
    gateway.new_array = lambda _, n: [None] * n

    with mock.patch("pyspark.SparkContext._gateway", gateway):
        run_spark_net_from_known_assembly(spark, zip, binary, args)

    # https://stackoverflow.com/a/21611963
    assert main.call_count == 1
    assert main.call_args[0][0] == [zip, binary] + args


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            ["--zipFile", "foo.zip", "--binaryName", "bar.dll", "--arg1", "val1"],
            ["foo.zip", "bar.dll", "--arg1", "val1"],
        ),
        (
            ["--binaryName", "bar.dll", "--arg1", "val1", "--zipFile", "foo.zip"],
            ["foo.zip", "bar.dll", "--arg1", "val1"],
        ),
    ],
)
def test_run_spark_net(args, expected):
    gateway = mock.MagicMock()
    gateway.new_array = lambda _, n: [None] * n

    spark = mock.MagicMock()
    spark_conf = {"spark.yarn.dist.pyFiles": ",".join(args)}
    spark.sparkContext.getConf = lambda: spark_conf

    main = mock.MagicMock()
    spark._jvm.org.apache.spark.deploy.dotnet.DotnetRunner.main = main

    # https://stackoverflow.com/a/34668809
    mock_app_name = mock.MagicMock()
    mock_app_name.getOrCreate = lambda: spark

    with mock.patch("sys.argv", args):
        with mock.patch("pyspark.SparkContext._gateway", gateway):
            with mock.patch(
                "pyspark.sql.SparkSession.builder.appName", return_value=mock_app_name
            ):
                run_spark_net()

    # https://stackoverflow.com/a/21611963
    assert main.call_count == 1
    assert main.call_args[0][0] == expected
