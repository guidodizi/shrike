# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Namespace containing Spark .NET utilities.

Inspired by the closed-source implementations:

- https://dev.azure.com/msdata/Vienna/_git/aml-ds?version=GC94d20cb3f190e942b016c548308becc107fcede8&path=/recipes/signed-components/canary-hdi/run.py  # noqa: E501
- https://dev.azure.com/eemo/TEE/_git/TEEGit?version=GC8f000c8c61ae67cf1009d7a70753f0175968ef81&path=/Offline/SGI/src/python/src/sparknet/run.py  # noqa: E501
- https://o365exchange.visualstudio.com/O365%20Core/_git/EuclidOdinML?version=GC849128898f37d138725695fbba3992cd5f5f4474&path=/sources/dev/Projects/dotnet/OdinMLDotNet/DotnetSpark-0.4.0.py  # noqa: E501
"""


import argparse
import logging
from shrike.compliant_logging import DataCategory, enable_compliant_logging
from shrike.compliant_logging.exceptions import (
    PublicRuntimeError,
    PublicValueError,
    PublicArgumentError,
)
import sys
from typing import Any, List, Optional
import uuid


log = logging.getLogger(__name__)


try:
    import py4j
    from pyspark.sql import SparkSession
    from pyspark import SparkContext
except ImportError as e:
    log.warning(f"Unable to import py4j or pyspark: {e}")
    py4j = Any
    SparkSession = Any
    SparkContext = Any


def get_default_spark_session() -> SparkSession:
    """
    Resolve a default Spark session for running Spark .NET applications.
    """
    # https://stackoverflow.com/a/534847
    random = str(uuid.uuid4())[:8]
    app_name = f"spark-net-{random}"
    log.info(f"Application name: {app_name}", category=DataCategory.PUBLIC)
    rv = SparkSession.builder.appName(app_name).getOrCreate()
    return rv


def full_pyfile_path(spark: SparkSession, py_file_name: str) -> str:
    """
    Resolve the full HDFS path of a file out of the Spark Session's PyFiles
    object.
    """
    spark_conf = spark.sparkContext.getConf()
    py_files = spark_conf.get("spark.yarn.dist.pyFiles")
    file_names = py_files.split(",")

    log.info(
        f"Searching py_files for {py_file_name}: '{py_files}'",
        category=DataCategory.PUBLIC,
    )

    for file_name in file_names:
        if file_name.split("/")[-1] == py_file_name:
            return file_name

    raise PublicValueError(f"py_files do not contain {py_file_name}: {py_files}")


def java_args(spark: SparkSession, args: List[str]):
    """
    Convert a Python list into the corresponding Java argument array.
    """
    rv = SparkContext._gateway.new_array(spark._jvm.java.lang.String, len(args))

    # https://stackoverflow.com/a/522578
    for index, arg in enumerate(args):
        rv[index] = arg

    return rv


def run_spark_net_from_known_assembly(
    spark: SparkSession, zip_file_name: str, assembly_name: str, args: List[str]
) -> None:
    """
    Invoke the binary `assembly_name` inside `zip_file_name` with the command
    line parameters `args`, using the provided Spark session. Print the Java
    stack trace if the job fails.
    """
    fully_resolved_zip_file_name = full_pyfile_path(spark, zip_file_name)
    dotnet_args = [fully_resolved_zip_file_name, assembly_name] + args

    log.info(
        f"Calling dotnet with arguments: {dotnet_args}", category=DataCategory.PUBLIC
    )
    dotnet_args_java = java_args(spark, dotnet_args)

    message = None

    try:
        spark._jvm.org.apache.spark.deploy.dotnet.DotnetRunner.main(dotnet_args_java)
    except py4j.protocol.Py4JJavaError as err:
        log.error("Dotnet failed", category=DataCategory.PUBLIC)
        for line in err.java_exception.getStackTrace():
            log.error(str(line), category=DataCategory.PUBLIC)

        message = f"{err.errmsg} {err.java_exception}"

    if message:
        # Don't re-raise the existing exception since it's unprintable.
        # https://github.com/bartdag/py4j/issues/306
        raise PublicRuntimeError(message)

    log.info("Done running dotnet", category=DataCategory.PUBLIC)


def run_spark_net(
    zip_file: str = "--zipFile",
    binary_name: str = "--binaryName",
    spark: Optional[SparkSession] = None,
    args: Optional[list] = None,
) -> None:
    """
    Easy entry point to one-line run a Spark .NET application. Simplest sample
    usage is:

    > run_spark_net_with_smart_args()
    """

    if not spark:
        spark = get_default_spark_session()

    if not args:
        args = sys.argv

    enable_compliant_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(zip_file, dest="ZIP_FILE")
    parser.add_argument(binary_name, dest="BINARY_NAME")

    try:
        (known_args, unknown_args) = parser.parse_known_args(args)

        zf = known_args.ZIP_FILE
        bn = known_args.BINARY_NAME
    except BaseException as e:
        raise PublicArgumentError(None, str(e)) from e

    run_spark_net_from_known_assembly(spark, zf, bn, unknown_args)
