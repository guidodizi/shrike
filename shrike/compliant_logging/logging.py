# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utilities around logging data which may or may not contain private content.
"""


from shrike.compliant_logging.exceptions import PublicRuntimeError
from typing import Optional, Iterable
from shrike.compliant_logging.constants import DataCategory
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
)
from datetime import datetime
import logging
import sys
from threading import Lock


_LOCK = Lock()
_PREFIX = None
_AML_RUN = None


def set_prefix(prefix: str) -> None:
    """
    Set the global prefix to use when logging public (non-private) data.

    This method is thread-safe.
    """
    with _LOCK:
        global _PREFIX
        _PREFIX = prefix


def get_prefix() -> Optional[str]:
    """
    Obtain the current global prefix to use when logging public (non-private)
    data.
    """
    return _PREFIX


def set_aml_context() -> None:
    """
    Retrieves the AML Context, should be bundled in a try-catch.
    """
    global _AML_RUN
    from azureml.core.run import Run

    _AML_RUN = Run.get_context()


def get_aml_context():
    """
    Obtains the AML Context
    """
    return _AML_RUN


def floating_range(buckets):
    """
    Computes a equal distributed list of bucket thresholds

    Args:
        buckets (int): Number of buckets

    Returns:
        List: List of bucket thresholds of length buckets
    """
    return [x / 100 for x in list(range(0, 100, int(100 / (buckets - 1)))) + [100]]


class CompliantLogger(logging.getLoggerClass()):  # type: ignore
    """
    Subclass of the default logging class with an explicit `category` parameter
    on all logging methods. It will pass an `extra` param with `prefix` key
    (value depending on whether `category` is public or private) to the
    handlers.

    The default value for data `category` is `PRIVATE` for all methods.

    Implementation is inspired by:
    https://github.com/python/cpython/blob/3.8/Lib/logging/__init__.py
    """

    def __init__(self, name: str, use_aml_metrics: bool = False):
        super().__init__(name)  # type: ignore
        self.start_time = datetime.now()
        self.metric_count = 1
        # number of iterable items that are logged
        self.max_iter_items = 10

        # check for azure context
        if use_aml_metrics:
            run = get_aml_context()
            if run is None:
                try:
                    set_aml_context()
                    self.info(
                        "AML Metrics:"
                        + f"{get_aml_context().get_status()}",  # type: ignore
                        category=DataCategory.PUBLIC,
                    )
                except Exception:
                    self.warning(
                        "AML writer failed to initialize.", category=DataCategory.PUBLIC
                    )

    def _convert_obj(self, obj, category=DataCategory.PRIVATE):
        """
        Converts the given object into a string type.

        Args:
            obj ([type]): Object to convert

        Returns:
            str: Info String about the object
        """
        # pass through for strings
        if isinstance(obj, str):
            return obj

        # check through different types of objects
        try:
            if is_spark_dataframe(obj):
                return get_spark_dataframe_info(obj)

            if is_vaex_dataframe(obj):
                return get_vaex_dataframe_info(obj)

            if is_pandas_dataframe(obj):
                return get_pandas_dataframe_info(obj)

            if is_pandas_series(obj):
                return get_pandas_series_info(obj)

            if is_numpy_array(obj):
                return get_numpy_array_info(obj)

            if isinstance(obj, Iterable):
                list_str = f"List (Count: {len(obj)})"  # type: ignore
                if category == DataCategory.PUBLIC:
                    list_str += " | "
                    if len(obj) < self.max_iter_items:  # type: ignore
                        list_str += str(obj)
                    else:
                        list_str += (
                            str(obj[: self.max_iter_items]) + "..."  # type: ignore
                        )
                return list_str

            if isinstance(obj, Exception):
                exc_type, exc_obj, exc_tb = sys.exc_info()
                obj_str = str(type(obj))
                if category == DataCategory.PUBLIC:
                    obj_str = str(obj)
                return f"Exception ({str(exc_type)}) for object {obj_str}"

        except Exception as ex:
            return f"Unkown DataType ({obj} > {type(ex)})"

        # default add the type of object
        if category == DataCategory.PUBLIC:
            return f"{type(obj)} | {str(obj)}"
        return str(type(obj))

    def _get_aml_context(self):
        """
        Obtains the AML Run Context and log warning messages
        if unable to retrieve AML Run Context
        """
        run = get_aml_context()
        if run is None:
            self.warning(
                "Unable to retrieve AML Run Context, will print to logs",
                category=DataCategory.PUBLIC,
            )
        return run

    def _log(
        self,
        level,
        msg,
        args=None,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        items=None,
        category=DataCategory.PRIVATE,
    ):
        p = ""
        if category == DataCategory.PUBLIC:
            p = get_prefix()
        if extra:
            extra.update({"prefix": p})
        else:
            extra = {"prefix": p}

        # update message accordingly to items
        if items is not None:
            if not isinstance(items, list):
                items = [items]
            msg += " | " + " | ".join(
                [self._convert_obj(item, category) for item in items]
            )

        if sys.version_info[1] <= 7:
            super(CompliantLogger, self)._log(
                level=level,
                msg=msg,
                args=args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
            )
        else:
            super(CompliantLogger, self)._log(
                level=level,
                msg=msg,
                args=args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,  # type: ignore
            )

    def metric(
        self,
        value,
        step=None,
        name=None,
        description=None,
        max_rows=250,
        category=DataCategory.PRIVATE,
    ):
        """
        Converts most datatypes into a metric and logs them to AML Metric
        for the RunContext (if available) or directly to log

        Note: Private Data will not be send to metrics!

        Args:
            value (Any): The value to log
                (can be vaex/pandas/spark dataframe, numpy array, list, dict, int/float)
            step (str | int, optional): Step value used for single value items.
                Defaults to None.
            name (str, optional): Name under which the metric should be logged.
                Defaults to None.
            description (str, optional): Description for the metric provided
                to the run context. Defaults to None.
            max_rows (int, optional): Defines the number of rows to batch table metrics
                (only required for table metrics).
                Defaults to 250.
            category (DataCategory, optional): Category of the data
                (logging to AML requires this to be set to PUBLIC explicitly).
                Defaults to DataCategory.PRIVATE.
        """
        # check for name
        if name is None:
            name = f"metric_{self.metric_count}"
            self.metric_count += 1
        # check for description
        if description is None:
            description = ""

        # retrieve AML Context
        run = self._get_aml_context()

        # check if value provided
        if value is None:
            self.error(
                f"Value provided for metric {name} is None, skipping (step: {step})"
            )
            return

        # check different data-types
        if isinstance(value, (float, int)):
            # log the data
            if run is not None and category == DataCategory.PUBLIC:
                if step:
                    run.log(name=name, value=value, description=description, step=step)
                else:
                    run.log(name=name, value=value, description=description)
            else:
                self.info(
                    f"NumbericMetric  | {name}:{step} | {value}",
                    category=category,
                )

            return

        # collect dataframes
        if is_vaex_dataframe(value):
            value = collect_vaex_dataframe(value)
        elif is_spark_dataframe(value):
            value = collect_spark_dataframe(value)
        elif is_pandas_dataframe(value):
            value = collect_pandas_dataframe(value)

        # log dictionary data
        if isinstance(value, dict):
            # check if values are present
            if len(value) == 0:
                self.warning(f"Dictionary Value for Metric {name} is empty. Skipping.")
                return

            # check the value types of the dict
            type_set = list(set([type(v) for v in value.values()]))

            # check for mixed types
            if len(type_set) > 1:
                pass
            else:
                type_set = type_set[0]

            # check types
            if type_set == list:
                if run is not None and category == DataCategory.PUBLIC:
                    run.log_table(name, value, description)
                else:
                    # log the matrix manually
                    col_names = " | ".join(
                        [f"{('' if col is None else col):15}" for col in value.keys()]
                    )
                    header = f"TableMetric     | Index | {col_names} |"
                    self.info(f"TableMetric     | {name}", category=category)
                    self.info(header, category=category)
                    self.info("-" * len(header), category=category)

                    # generate the rows
                    max_rows = max([len(value[col]) for col in value])
                    for i in range(max_rows):
                        row_str = f"TableMetric     | {i:05}"
                        for key in value:
                            col = value[key]
                            col = col[i] if i < len(col) and col[i] else ""
                            row_str += f" | {str(col):15}"
                        self.info(row_str, category=category)
            elif type_set in [int, float]:
                for key, val in value.items():
                    key = name + "/" + key
                    self.metric(val, step, key, description, category)
            else:
                self.warning(
                    (
                        "The provided dictionary for metric"
                        f" {name} appears to be unstructured!"
                    ),
                    category=category,
                )

            return

        # collect list wise datatypes
        if is_numpy_array(value):
            value = numpy_array_to_list(value)
        if is_pandas_series(value):
            value = pandas_series_to_list(value)

        # log list data
        if isinstance(value, (list, tuple)):
            value = list(value)

            # check if values are present
            if len(value) == 0:
                self.warning(f"List Value for Metric {name} is empty. Skipping.")
                return

            # log data to run context
            if run is not None and category == DataCategory.PUBLIC:
                run.log_list(name=name, value=value, description=description)
            else:
                self.info(f"ListMetric      | {name} | {value}")

            return

        self.warning(
            f"Value {value} of the provided metric {name} has an unkown type",
            category=category,
        )

    def metric_value(
        self, name, value, description=None, step=None, category=DataCategory.PRIVATE
    ):
        """
        Equivalent to the `Run.log` function.
        Logs a single value to a metric

        Note: Private Data will not be send to metrics!

        Args:
            name (str): name of the metric
            value (Any): value to log
            description (str, optional): Description of the metric. Defaults to None.
            step (int, optional): Step of the current metric. Defaults to None.
            category (DataCategory, optional): Data category to make sure no data leaks.
                Defaults to DataCategory.PRIVATE.
        """
        self.metric(value, step, name, description, category=category)

    def metric_image(
        self,
        name=None,
        plot=None,
        path=None,
        description=None,
        category=DataCategory.PRIVATE,
    ):
        """
        Logs an image to the AML Metrics.
        Note that this is only possible for public data when
        AML Run context is available

        Note: Private Data will not be send to metrics!

        Args:
            plot (pyplot.Plot): The plot that should be logger
            path (str, optional): Optional Path to the image. Defaults to None.
            name (str, optional): Name of the image. Defaults to None.
            description (str, optional): Description of the metric. Defaults to None.
            category (DataCategory, optional): Category under which this image is logged
                Defaults to DataCategory.PRIVATE.
        """
        # retrieve the run context
        run = self._get_aml_context()

        # check if parameters are correct
        if category != DataCategory.PUBLIC:
            self.warning(f"Unable to log image metric {name} as private, skipping.")
            return

        # check for name
        if name is None:
            name = f"metric_{self.metric_count}"
            self.metric_count += 1
        if description is None:
            description = ""

        # log the image
        run.log_image(  # type: ignore
            name=name, path=path, plot=plot, description=description
        )

    def metric_list(self, name, value, description=None, category=DataCategory.PRIVATE):
        """
        Equivalent to the `Run.log_list`.
        Logs a list of values for a single metric.

        Note: Private Data will not be send to metrics!

        Args:
            value (list): List values to log
            name (str, optional): Name of the metric. Defaults to None.
            description (str, optional): Description of the metric. Defaults to None.
            category (DataCategory, optional): DataCategory to log the data as.
                Defaults to DataCategory.PRIVATE.
        """
        self.metric(value, name=name, description=description, category=category)

    def _compute_truth_matrix(self, predict, target, class_id, thresholds):
        """
        Computes the truth matrix for the given class in a one-vs-rest fashion.

        Args:
            predict (pd.Series): Series of probability values for the target class
            target (pd.Series): Series of target classes
            class_id (int): id of the current class
            thresholds (list): List of thresholds of length P to be used for the
                computation

        Returns:
            Matrix: A truth matrix in format [P, 4] that contains tp, fp, tn, fn for
                each of the thresholds
        """
        cl_table = []
        for thres in thresholds:
            cl_res = [
                (predict >= thres) & (target == class_id),
                (predict >= thres) & (target != class_id),
                (predict < thres) & (target != class_id),
                (predict < thres) & (target == class_id),
            ]
            cl_table.append([x.sum() for x in cl_res])
        return cl_table

    def metric_accuracy_table(
        self,
        name,
        value,
        description=None,
        col_predict=None,
        col_target=None,
        probability_thresholds=5,
        percentile_thresholds=[0.0, 0.01, 0.24, 0.98, 1.0],
        class_labels=None,
        category=DataCategory.PRIVATE,
    ):
        """
        Equivalent of the `Run.log_accuracy_table` function.
        Logs the data for an accuracy table to the metrics.

        In the dataframe case, the `col_predict` value has to contain the prediction
        probabilities for the **target** class!

        Note: Private Data will not be send to metrics!

        Args:
            value (dict | table): Either dicationary in AML defined format
                or table that provides accuracy values.
            name (str, optional): Name of the metric. Defaults to None.
            description (str, optional): Description of the metric. Defaults to None.
            col_predict (str | int, optional): Name or Id of the predicted probabilities
                for the target class. This is only required if DataFrame is passed.
                Defaults to None.
            col_target (str | int, optional): Name or id of the target value column.
                This is only required if DataFrame is passed. Defaults to None.
            probability_thresholds (list | int, optional): Either a list of thresholds
                or a number of evenly spaced threshold points. Defaults to 5.
            percentile_thresholds (list | int, optional): Either a list of thresholds
                or a number of evenly spaced threshold points. Defaults to a list.
            category (DataCategory, optional): Classification of the data category.
                Defaults to DataCategory.PRIVATE.
        """
        # retrieve the context
        run = self._get_aml_context()

        # convert data if not already pre-computed
        if not isinstance(value, dict) or "schema_type" not in value:
            # check the data
            if is_vaex_dataframe(value):
                value = collect_vaex_dataframe(value)
            if is_spark_dataframe(value):
                value = collect_spark_dataframe(value)
            if is_pandas_dataframe(value):
                value = collect_pandas_dataframe(value)

            # check if datatype matches
            if not isinstance(value, dict):
                raise PublicRuntimeError("Unkown value-type passed to accuracy_table!")

            # convert the data
            try:
                import pandas as pd

                # create the dataframe
                df = pd.DataFrame.from_dict(value)

                # column checks
                if None in [col_predict, col_target]:
                    raise PublicRuntimeError(
                        "If table is passed to accuracy_table it requires all "
                        + "columns to be present!"
                    )

                # check the class list (sort to make sure it is aligned)
                class_list = list(df[col_target].unique())
                class_list.sort()
                if class_labels is None:
                    class_labels = class_list

                # compute ranges
                if isinstance(probability_thresholds, int):
                    probability_thresholds = floating_range(probability_thresholds)
                if isinstance(percentile_thresholds, int):
                    percentile_thresholds = floating_range(percentile_thresholds)

                # compute one-vs-rest labels for the class
                prob_tables = []
                perc_tables = []
                for cl in class_list:
                    # compute the thresholds
                    prob_tables.append(
                        self._compute_truth_matrix(
                            df[col_predict], df[col_target], cl, probability_thresholds
                        )
                    )

                    # compute per class percentiles
                    cl_proba = (df[col_predict] * (df[col_target] == cl)) + (
                        (1 - df[col_predict]) * (df[col_target] != cl)
                    )
                    cl_percentile = list(cl_proba.quantile(percentile_thresholds))
                    perc_tables.append(
                        self._compute_truth_matrix(
                            df[col_predict], df[col_target], cl, cl_percentile
                        )
                    )

                # generate data
                value = {
                    "schema_type": "accuracy_table",
                    "schema_version": "1.0.1",
                    "data": {
                        "probability_tables": prob_tables,
                        "precentile_tables": perc_tables,
                        "probability_thresholds": probability_thresholds,
                        "percentile_thresholds": percentile_thresholds,
                        "class_labels": class_labels,
                    },
                }
            except Exception:
                raise PublicRuntimeError(
                    "Unable to import pandas and parse the given data table! "
                    + "Make sure that libraries are available and "
                    + "correct data is passed."
                )

        # log the data
        if category == DataCategory.PUBLIC and run is not None:
            run.log_accuracy_table(name, value, description)
        else:
            self.warning("Logging Accuracy Tables to text is not yet implemented")

    def metric_confusion_matrix(
        self,
        name,
        value,
        idx_true=None,
        idx_pred=None,
        labels=None,
        description=None,
        category=DataCategory.PRIVATE,
    ):
        """
        Equivalent of the `Run.log_confusion_matrix` function.
        Logs or generates a confusion matrix to the AML logs.

        Note: Private Data will not be send to metrics!

        Args:
            value (dict | DataFrame): Data to be used for the confusion_matrix
            idx_true (int | str, optional): Name or id of the target column.
                Defaults to None.
            idx_pred (int | str, optional): Name or id of the prediction column.
                Defaults to None.
            labels (list, optional): List of labels used for the rows. Defaults to None.
            name (str, optional): Name of the metric. Defaults to None.
            description (str, optional): Description of the metric. Defaults to None.
            category (DataCategory, optional): Classification of the data.
                Defaults to DataCategory.PRIVATE.

        Raises:
            PublicRuntimeError: [description]
            PublicRuntimeError: [description]
        """
        # retrieve the context
        run = self._get_aml_context()

        # convert data if not already pre-computed
        if (
            not isinstance(value, dict)
            or "schema_type" not in value
            or "schema_version" not in value
        ):
            # check the data
            if is_vaex_dataframe(value):
                value = collect_vaex_dataframe(value)
            if is_spark_dataframe(value):
                value = collect_spark_dataframe(value)
            if is_pandas_dataframe(value):
                value = collect_pandas_dataframe(value)

            # check if datatype matches
            if not isinstance(value, dict):
                raise PublicRuntimeError(
                    "Unkown value-type passed to Run.log_confusion_matrix!"
                )

            # convert the data
            try:
                # try to import libs
                import numpy as np
                from sklearn.metrics import confusion_matrix

                # update row names
                if isinstance(idx_true, str):
                    idx_true = list(value.keys()).index(idx_true)
                if isinstance(idx_pred, str):
                    idx_pred = list(value.keys()).index(idx_pred)

                # retrieve left right
                val_true, val_pred = None, None
                value = np.array(list(value.values()))
                val_true = value[idx_true]
                val_pred = value[idx_pred]

                # compute matrix
                mat = confusion_matrix(val_true, val_pred)

                # generate labels as distincts
                if labels is None:
                    labels = np.unique(val_true)

                # generate the dict
                value = {
                    "schema_type": "confusion_matrix",
                    "schema_version": "1.0.0",
                    "data": {"class_labels": labels, "matrix": mat},
                }
            except Exception:
                raise PublicRuntimeError(
                    "Unable to import numpy & scikit and parse the given data table! "
                    + "Make sure that libraries are available and correct "
                    + "data is passed."
                )

        # log the data
        if category == DataCategory.PUBLIC and run is not None:
            run.log_confusion_matrix(name, value, description)
        else:
            self.warning("Logging Confusion Matrices to text is not yet implemented")

    def metric_predictions(
        self,
        name,
        value,
        description=None,
        col_predict=None,
        col_target=None,
        bin_edges=5,
        category=DataCategory.PRIVATE,
    ):
        """
        Equivalent of `Run.log_predictions` function.
        This will log regression prediction histogram from dict or dataframe.

        Note: Private Data will not be send to metrics!

        For the dataframe case the prediction error is computed as the absolute
        difference between prediction and target.

        Args:
            name (str): Name of the metric
            value (dict | DataFrame): The data to log
            description (str, optional): Description of the metric. Defaults to ''.
            col_predict (str | int, optional): Id or Name of the target column.
                Defaults to None.
            bin_edges (list, optional): List of edge boundaries for logging.
                Defaults to None.
            category (DataCategory, optional): Privacy Classification of the data.
                Defaults to DataCategory.PRIVATE.

        Raises:
            PublicRuntimeError: If the data is not in the right format or required
                parameters are not passed.
        """
        # retrieve the context
        run = self._get_aml_context()

        # convert data if not already pre-computed
        if (
            not isinstance(value, dict)
            or "schema_type" not in value
            or "schema_version" not in value
        ):
            # check the data
            if is_vaex_dataframe(value):
                value = collect_vaex_dataframe(value)
            if is_spark_dataframe(value):
                value = collect_spark_dataframe(value)
            if is_pandas_dataframe(value):
                value = collect_pandas_dataframe(value)

            # check if datatype matches
            if not isinstance(value, dict):
                raise PublicRuntimeError("Unkown value-type passed to predictions!")

            # convert the data
            try:
                import pandas as pd

                # create the dataframe
                df = pd.DataFrame.from_dict(value)

                # column checks
                if None in [col_predict, col_target]:
                    raise PublicRuntimeError(
                        "The col_predict and col_target columns are both required."
                    )

                # compute edges automatically
                if isinstance(bin_edges, int):
                    bin_edges = floating_range(bin_edges)

                # compute groupings in bins
                df["bin"] = pd.cut(df[col_target], bin_edges)
                df["error"] = (df[col_predict] - df[col_target]).abs()

                # generate data
                value = {
                    "schema_type": "predictions",
                    "schema_version": "1.0.0",
                    "data": {
                        "bin_averages": list(df.groupby("bin")[col_target].mean()),
                        "bin_errors": list(df.groupby("bin")["error"].sum()),
                        "bin_counts": list(df.groupby("bin")[col_target].count()),
                        "bin_edges": bin_edges,
                    },
                }
            except Exception:
                raise PublicRuntimeError(
                    "Unable to import pandas and parse the given data! "
                    + "Make sure that libraries are available and correct "
                    + "data is passed."
                )

        # log the data
        if category == DataCategory.PUBLIC and run is not None:
            run.log_predictions(name, value, description)
        else:
            self.warning("Logging Predictions to text is not yet implemented")

    def metric_residual(
        self,
        name,
        value,
        description=None,
        col_predict=None,
        col_target=None,
        bin_edges=5,
        category=DataCategory.PRIVATE,
    ):
        """
        Equivalent on the `Run.log_residuals` functions.
        Logs residual values for a list of edges

        Note: Private Data will not be send to metrics!

        Args:
            name (str): Name of the metric
            value (dict | DataFrame): Values to contain the residuals
            description (str, optional): Description of the dataframe. Defaults to ''.
            col_target (str, optional): Name of the target column (if value is a df).
                Defaults to None.
            bin_edges (list | int, optional): List of edges towards the bins.
                Defaults to 5.
            category (DataCategory, optional): Privacy Classification of the data.
                Defaults to DataCategory.PRIVATE.

        Raises:
            PublicRuntimeError: Thrown when data is in unkown format or required params
                not provided
        """
        # retrieve the context
        run = self._get_aml_context()

        # convert data if not already pre-computed
        if (
            not isinstance(value, dict)
            or "schema_type" not in value
            or "schema_version" not in value
        ):
            # check the data
            if is_vaex_dataframe(value):
                value = collect_vaex_dataframe(value)
            if is_spark_dataframe(value):
                value = collect_spark_dataframe(value)
            if is_pandas_dataframe(value):
                value = collect_pandas_dataframe(value)

            # check if datatype matches
            if not isinstance(value, dict):
                raise PublicRuntimeError(
                    "Unkown value-type passed to Run.log_residuals()!"
                )

            # convert the data
            try:
                import pandas as pd

                # create the dataframe
                df = pd.DataFrame.from_dict(value)

                # column checks
                if None in [col_predict, col_target]:
                    raise PublicRuntimeError(
                        "The col_predict and col_target columns are both required."
                    )

                # check if bins should be generated automatically
                if isinstance(bin_edges, int):
                    bin_edges = floating_range(bin_edges)

                # compute the values
                df["residual"] = df[col_predict] - df[col_target]
                df["bin"] = pd.cut(df[col_target], bin_edges)

                # generate data
                value = {
                    "schema_type": "residuals",
                    "schema_version": "1.0.0",
                    "data": {
                        "bin_edges": bin_edges,
                        "bin_counts": list(df.groupby("bin")["residual"].sum()),
                    },
                }
            except Exception:
                raise PublicRuntimeError(
                    "Unable to import pandas and parse the given data! "
                    + "Make sure that libraries are available and correct "
                    + "data is passed."
                )

        # log the data
        if category == DataCategory.PUBLIC and run is not None:
            run.log_residuals(name, value, description)
        else:
            self.warning("Logging Residuals to text is not yet implemented")

    def metric_row(
        self, name, description=None, category=DataCategory.PRIVATE, **kwargs
    ):
        """
        Equivalent of the `Run.log_row` function.
        Logs a single row of a table to the metrics.

        Note: Private Data will not be send to metrics!

        Args:
            name (str): Name of the metric.
            description (str): Description of the metric.
            category (DataCategory, optional): Classification of the data.
                Defaults to DataCategory.PRIVATE.
        """
        # check run context
        run = self._get_aml_context()

        # log the data
        if category == DataCategory.PUBLIC and run is not None:
            run.log_row(name=name, description=description, **kwargs)
        else:
            row_str = f"RowMetric      | {name} | "
            row_str += " | ".join([f"{r}:{c}" for r, c in kwargs.items()])
            self.info(row_str, category=category)

    def metric_table(
        self, name, value, description=None, category=DataCategory.PRIVATE
    ):
        """
        Equivalent to the `Run.log_table` function.
        Logs a table in dict format {rows: [values]} to metrics.

        Note: Private Data will not be send to metrics!

        Args:
            name (str): Name of the metric.
            value (dict): Dictionary representation of the table.
            description (str, optional): Description of the metric. Defaults to None.
            category (DataCategory, optional): Category to log the data.
                Default to DataCategory.PRIVATE.
        """
        self.metric(value=value, name=name, description=description, category=category)


_logging_basic_config_set_warning = """
********************************************************************************
The root logger already has handlers set! As a result, the behavior of this
library is undefined. If running in Python >= 3.8, this library will attempt to
call logging.basicConfig(force=True), which will remove all existing root
handlers. See https://stackoverflow.com/q/20240464 and
https://github.com/Azure/confidential-ml-utils/issues/33 for more information.
********************************************************************************
"""


def enable_compliant_logging(
    prefix: str = "SystemLog:", use_aml_metrics: bool = False, **kwargs
) -> None:
    """
    The default format is `logging.BASIC_FORMAT` (`%(levelname)s:%(name)s:%(message)s`).
    All other kwargs are passed to `logging.basicConfig`. Sets the default
    logger class and root logger to be compliant. This means the format
    string `%(prefix)` will work.

    Set the format using the `format` kwarg.

    If running in Python >= 3.8, will attempt to add `force=True` to the kwargs
    for logging.basicConfig.

    After calling this method, use the kwarg `category` to pass in a value of
    `DataCategory` to denote data category. The default is `PRIVATE`. That is,
    if no changes are made to an existing set of log statements, the log output
    should be the same.

    The standard implementation of the logging API is a good reference:
    https://github.com/python/cpython/blob/3.9/Lib/logging/__init__.py
    """
    set_prefix(prefix)

    if "format" not in kwargs:
        kwargs["format"] = f"%(prefix)s{logging.BASIC_FORMAT}"

    # Ensure that all loggers created via `logging.getLogger` are instances of
    # the `CompliantLogger` class.
    logging.setLoggerClass(CompliantLogger)

    if len(logging.root.handlers) > 0:
        p = get_prefix()
        for line in _logging_basic_config_set_warning.splitlines():
            print(f"{p}{line}", file=sys.stderr)

    if "force" not in kwargs and sys.version_info >= (3, 8):
        kwargs["force"] = True

    old_root = logging.root

    root = CompliantLogger(logging.root.name, use_aml_metrics)
    root.handlers = old_root.handlers

    logging.root = root
    logging.Logger.root = root  # type: ignore
    logging.Logger.manager = logging.Manager(root)  # type: ignore

    # https://github.com/kivy/kivy/issues/6733
    logging.basicConfig(**kwargs)


def enable_confidential_logging(
    prefix: str = "SystemLog:", use_aml_metrics: bool = False, **kwargs
) -> None:
    """
    This function is a duplicate of the function `enable_compliant_logging`.
    We encourage users to use `enable_compliant_logging`.
    """
    print(
        f"{prefix} The function enable_confidential_logging() is on the way"
        " to deprecation. Please use enable_compliant_logging() instead.",
        file=sys.stderr,
    )
    enable_compliant_logging(prefix, use_aml_metrics, **kwargs)
