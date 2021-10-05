# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Demonstrate advanced use of the Spark .NET runner functionality.
"""

import argparse
from pyspark.sql import SparkSession
from shrike.spark import run_spark_net_from_known_assembly


def main(args):
    spark = SparkSession.builder.appName(args.app_name).getOrCreate()

    run_spark_net_from_known_assembly(
        spark, "dotnet-publish.zip", "assembly-name", ["--input", args.input_path]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name")
    parser.add_argument("--input-path")
    args = parser.parse_args()
    main(args)
