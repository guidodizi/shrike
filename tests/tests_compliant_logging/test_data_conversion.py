# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np
import pandas as pd
import vaex
import platform
from packaging.version import parse
from shrike.compliant_logging.exceptions import PublicRuntimeError
import pytest

from shrike.compliant_logging.data_conversions import (
    collect_pandas_dataframe,
    collect_spark_dataframe,
    collect_vaex_dataframe,
    is_numpy_array,
    is_pandas_dataframe,
    is_pandas_series,
    is_spark_dataframe,
    is_vaex_dataframe,
    get_numpy_array_info,
    get_pandas_dataframe_info,
    get_pandas_series_info,
    get_spark_dataframe_info,
    get_vaex_dataframe_info,
    numpy_array_to_list,
    pandas_series_to_list,
    pandas_dataframe_schema,
    vaex_dataframe_schema,
    spark_dataframe_schema,
)
from shrike.compliant_logging.exceptions import PublicRuntimeError


def test_numpy_import():
    test_ndarray = np.array([1, 2, 3, 4, 5])
    assert is_numpy_array(test_ndarray)
    assert not is_numpy_array([1, 2, 3, 4, 5])
    assert "Numpy Array (Shape: (5,))" in get_numpy_array_info(test_ndarray)
    assert numpy_array_to_list(test_ndarray) == [1, 2, 3, 4, 5]


def test_pandas_import():
    test_series = pd.Series([1, 2, 3, 4, 5])

    # Create the pandas DataFrame
    test_df = pd.DataFrame(
        [["Tom", 10], ["Nick", 15], ["John", 14]], columns=["Name", "Age"]
    )

    assert is_pandas_series(test_series)
    assert not is_pandas_series([1, 2, 3, 4, 5])
    assert is_pandas_dataframe(test_df)
    assert not is_pandas_dataframe(test_series)
    assert "Pandas Series (Row Count: 5)" in get_pandas_series_info(test_series)
    assert (
        "Pandas DataFrame (Row Count: 3 / Column Count: 2)"
        in get_pandas_dataframe_info(test_df)
    )
    assert pandas_series_to_list(test_series) == [1, 2, 3, 4, 5]
    assert pandas_dataframe_schema(test_df) == {"Name": "object", "Age": "int64"}
    assert collect_pandas_dataframe(test_df) == {
        "Name": ["Tom", "Nick", "John"],
        "Age": [10, 15, 14],
    }


def test_vaex_import():
    test_df = vaex.from_arrays(x=np.arange(5), y=np.arange(5) ** 2)
    assert is_vaex_dataframe(test_df)
    assert "Vaex DataFrame (Row Count: 5 / Column Count: 2)" in get_vaex_dataframe_info(
        test_df
    )
    assert "x" in vaex_dataframe_schema(test_df)
    assert "y" in vaex_dataframe_schema(test_df, {"x": "x", "y": "y"})
    assert collect_vaex_dataframe(test_df) == {
        "x": [0, 1, 2, 3, 4],
        "y": [0, 1, 4, 9, 16],
    }
    assert collect_vaex_dataframe([1, 2, 3, 4, 5]) is None


def test_pyspark_import():
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType

    test_data = [
        ("James", "", "Smith", "36636", "M", 3000),
        ("Michael", "Rose", "", "40288", "M", 4000),
        ("Robert", "", "Williams", "42114", "M", 4000),
        ("Maria", "Anne", "Jones", "39192", "F", 4000),
        ("Jen", "Mary", "Brown", "", "F", -1),
    ]

    test_schema = StructType(
        [
            StructField("firstname", StringType(), True),
            StructField("middlename", StringType(), True),
            StructField("lastname", StringType(), True),
            StructField("id", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("salary", IntegerType(), True),
        ]
    )
    test_df = (
        SparkSession.builder.appName("SparkUnitTests")
        .getOrCreate()
        .createDataFrame(data=test_data, schema=test_schema)
    )

    assert is_spark_dataframe(test_df)
    assert not is_spark_dataframe([1, 2, 3, 4, 5])

    # get_spark_dataframe_info() may fail if JAVA version is not expected
    get_spark_dataframe_info(test_df)

    assert spark_dataframe_schema(test_df) == {
        "firstname": "string",
        "middlename": "string",
        "lastname": "string",
        "id": "string",
        "gender": "string",
        "salary": "int",
    }

    collect_spark_dataframe(test_df)
