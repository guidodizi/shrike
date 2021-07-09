# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provides a list of conversion functions based
on which libraries are available for import
"""

from typing import Any, Optional
from .exceptions import PublicRuntimeError


# spark functions
def is_spark_dataframe(obj: Any) -> bool:  # type: ignore
    """
    Checks if the given object is a spark dataframe.

    Args:
        obj (Any): The object to check

    Returns:
        bool: True if a dataframe, otherwise False
    """
    try:
        from pyspark.sql import DataFrame  # noqa
    except Exception:
        return False
    return isinstance(obj, DataFrame)


def get_spark_dataframe_info(df: Any) -> str:  # type: ignore
    """
    Provides information about the spark dataframe

    Args:
        obj (Any): The dataframe to provide information for

    Returns:
        str: info string about the dataframe
    """
    try:
        from pyspark.sql import DataFrame  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Spark DataFrame is not supported in current environment"
        )

    try:
        # df.count() may fail if JAVA version is not expected
        return "Spark DataFrame (Row Count: {} / Column Count: {})".format(
            df.count(), len(df.columns)
        )
    except Exception:
        return "Failed to extract Spark DataFrame info"


def spark_dataframe_schema(df: Any, schema_map: dict = None) -> dict:  # type: ignore
    """
    Retrieves the database schema in a dict as format `name: datatype` (both str)

    Args:
        df (Any): DataFrame to analyze
        schema_map (dict, optional): Optional renaming of column names.
            Defaults to None.

    Returns:
        dict: Str to Str dict that contains column names and datatypes
    """
    try:
        from pyspark.sql import DataFrame  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Spark DataFrame is not supported in current environment"
        )

    if schema_map:
        return dict([(schema_map.get(n, n), str(t)) for n, t in df.dtypes])
    else:
        return dict([(n, str(t)) for n, t in df.dtypes])


def collect_spark_dataframe(df: Any) -> Optional[dict]:  # type: ignore
    """
    Collects the spark dataframe as a dictionary,
    where keys are column names and values are a list of values

    Args:
        df (Any): The dataframe to collect

    Returns:
        dict: The dict with column names as keys and list of column values.
            Returns None if conversion fails
    """
    try:
        from pyspark.sql import DataFrame  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Spark DataFrame is not supported in current environment"
        )

    try:
        return df.toPandas().to_dict("list")
    except Exception:
        return None


# vaex functions


def is_vaex_dataframe(obj: Any) -> bool:  # type: ignore
    """
    Checks if the given object is a vaex dataframe

    Args:
        obj (Any): Object to check

    Returns:
        bool: True if a DataFrame otherwise False
    """
    try:
        from vaex.dataframe import DataFrame as VaexDataFrame  # noqa
    except Exception:
        return False
    return isinstance(obj, VaexDataFrame)


def get_vaex_dataframe_info(df: Any) -> str:  # type: ignore
    """
    Retrieves a general info string to the given Vaex DataFrame

    Args:
        df (Any): DataFrame to check for

    Returns:
        str: Info string
    """
    try:
        from vaex.dataframe import DataFrame as VaexDataFrame  # noqa
    except Exception:
        raise PublicRuntimeError("Vaex DataFrame not supported in current environment")

    return "Vaex DataFrame (Row Count: {} / Column Count: {})".format(
        df.count(), len(df.get_column_names())
    )


def vaex_dataframe_schema(  # type: ignore
    df: Any, schema_map: dict = None
) -> dict:  # type: ignore
    """
    Retrieves the schema of the given DataFrame

    Args:
        df (Any): DataFrame to provide the schema for
        schema_map (dict, optional): Mapping of the schema to

    Returns:
        dict: str to str dict that contains column names and datatypes
    """
    try:
        from vaex.dataframe import DataFrame as VaexDataFrame  # noqa
    except Exception:
        raise PublicRuntimeError("Vaex DataFrame not supported in current environment")

    if schema_map:
        return {
            schema_map.get(col, col): df[col].dtype.name
            for col in df.get_column_names()
        }
    else:
        return {
            column_name: df.data_type(column_name)
            for column_name in df.get_column_names()
        }


def collect_vaex_dataframe(df: Any) -> Optional[dict]:  # type: ignore
    """
    Collects the vaex dataframe as a dictionary,
    where keys are column names and values are a list of values

    Args:
        df (Any): The dataframe to collect

    Returns:
        dict: The dictionary with column names as keys and list of column values.
            Returns None if conversion fails
    """
    try:
        from vaex.dataframe import DataFrame as VaexDataFrame  # noqa
    except Exception:
        raise PublicRuntimeError("Vaex DataFrame not supported in current environment")

    try:
        return df.to_pandas_df().to_dict("list")
    except Exception:
        return None


# numpy functions
def is_numpy_array(obj: Any) -> bool:  # type: ignore
    """
    Checks if the given object is a numpy array

    Args:
        obj (Any): The object to check

    Returns:
        bool: True if the object is a numpy array otherwise False
    """
    try:
        import numpy as np  # noqa
    except Exception:
        return False
    return isinstance(obj, np.ndarray)


def get_numpy_array_info(arr: Any) -> str:  # type: ignore
    """
    Retrieves an info string from the given numpy array

    Args:
        arr (Any): The array to generate string from

    Returns:
        str: Info String
    """
    try:
        import numpy as np  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Numpy Array not supported in the current environment."
        )
    return "Numpy Array (Shape: {})".format(arr.shape)


def numpy_array_to_list(arr: Any) -> list:  # type: ignore
    """
    Converts the given numpy array into a python list

    Args:
        arr (Any): Array to convert

    Returns:
        list: Converted (potentially multi-dimensional) list
    """
    try:
        import numpy as np  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Numpy Array not supported in the current environment."
        )
    return arr.tolist()


# pandas functions
def is_pandas_series(obj: Any) -> bool:  # type: ignore
    """
    Checks if the given object is a pandas series

    Args:
        obj (Any): Object to check

    Returns:
        bool: True if object is a series otherwise False
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        return False
    return isinstance(obj, pd.Series)


def is_pandas_dataframe(obj: Any) -> bool:  # type: ignore
    """
    Checks if the given object is a pandas DataFrame

    Args:
        obj (Any): Object to check

    Returns:
        bool: True if object is a DataFrame otherwise False
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        return False
    return isinstance(obj, pd.DataFrame)


def get_pandas_series_info(series: Any) -> str:  # type: ignore
    """
    Retrieves an info string from the given pandas series

    Args:
        series (Any): Series to convert

    Returns:
        str: Info String containing the Row Count
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Pandas Series not supported in the current environment."
        )
    return "Pandas Series (Row Count: {})".format(series.count())


def get_pandas_dataframe_info(df: Any) -> str:  # type: ignore
    """
    Retrieves an info string from the given pandas DataFrame

    Args:
        df (Any): DataFrame to convert

    Returns:
        str: Info String containing Row and Column Count
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Pandas Series not supported in the current environment."
        )
    return "Pandas DataFrame (Row Count: {} / Column Count: {})".format(
        df.shape[0], df.shape[1]
    )


def pandas_series_to_list(series: Any) -> list:  # type: ignore
    """
    Converts the given Pandas Series to a list

    Args:
        series (Any): Series to convert

    Returns:
        list: List output
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Pandas Series not supported in the current environment."
        )
    return series.tolist()


def pandas_dataframe_schema(df: Any) -> dict:  # type: ignore
    """
    Retrieves the schema of the dataframe as dict in name to datatype format

    Args:
        df (Any): DataFrame to convert

    Returns:
        dict: str-to-str Dict that contains column names and datatypes
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Pandas Series not supported in the current environment."
        )
    return dict([(col, df[col].dtype.name) for col in df.columns])


def collect_pandas_dataframe(df: Any) -> dict:  # type: ignore
    """
    Converts the given pandas DataFrame to a list

    Args:
        df (Any): DataFrame to convert

    Returns:
        list: Converted Dataframe as multi-dimensional list
    """
    try:
        import pandas as pd  # noqa
    except Exception:
        raise PublicRuntimeError(
            "Pandas Series not supported in the current environment."
        )
    return df.to_dict("list")
