# Logging metrics in Azure ML Portal using shrike

Logging real-time metrics and sending them to Azure Machine Learning (ML) workspace portal
is supported by `shrike >= 1.7.0`. This page is on how to use the `shrike`
library to log various types of metrics in AML. For the information on how to use the
Azure ML Python SDK for metrics logging, please check out this [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics).

This metrics logging feature in shrike is supported in both eyes-on and eyes-off
environments, and it also works in the offline runs (e.g., your local laptop or any
non-AML virtual machines). For jobs in detonation chambers, please
check out this internal [note](https://dev.azure.com/msdata/Vienna/_wiki/wikis/aml-1p-onboarding/22174/Metric-filtering-for-detonation-chamber).

Before logging any metrics, call `shrike.compliant_logging.enable_compliant_logging`
with the argument `use_aml_metrics=True` and `category=DataCategory.PUBLIC` to connect with
the workspace portal and set up data-category-aware logging.
Then continue to use the standard Python logging functionality as before.
There are various metric-logging functions to match various metric types.
The API references on all metric-logging functions are availabe
on this [page](https://azure.github.io/shrike/compliant_logging/logging/#shrike.compliant_logging.logging.CompliantLogger.metric).

Use the following methods in the logging APIs for different scenarios & metric types.

|Logged Value|Example Code| Supported Types|
|----|----|----|
|Log image|`log.metric_image(name='food', path='./breadpudding.jpg', plot=None, description='desert', category=DataCategory.PUBLIC)`| string, matplotlib.pyplot.plot|
|Log an array of numeric values|`log.metric_list(name="Fibonacci", value=[0, 1, 1, 2, 3, 5, 8], category=DataCategory.PUBLIC)`| list, tuple|
|Log a single value |`log.metric_value(name='metric_value', value=1, step=NA, category=DataCategory.PUBLIC))`|scalar|
|Log a row with 2 numerical columns |`log.metric_row(name='Cosine Wave', angle=0, cos=1, category=DataCategory.PUBLIC))`|scalar|
|Log a table |`log.metric_table(name="students", value={"name": ["James", "Robert", "Michael"], "number": [2, 3, 1, 5]}, category=DataCategory.PUBLIC)`|dict|
|Log residuals |`log.metric_residual(name="ml_residual", value=panda.DataFrame([[1.0, 1.1], [2.0, 2.0], [3.0, 3.1]],columns=["pred", "targ"]), col_predict="pred", col_target="targ", category=DataCategory.PUBLIC)`|dict, pandas.DataFrame, vaex.dataframe, spark dataframe|
|Log confusion matrix |`log.metric_confusion_matrix(name="animal_classification", value=vaex.from_arrays(x=["cat", "ant", "cat", "cat", "ant", "bird"], y=["ant", "ant", "cat", "cat", "ant", "cat"]), idx_true="x", idx_pred="y",category=DataCategory.PUBLI)`|dict, pandas.DataFrame, vaex.dataframe, spark dataframe|
|Log accuracy table |`log.metric_confusion_matrix(name="accuracy_table", value=vaex.from_arrays(x=[0.1, 0.3, 0.7], y=["a", "b", "c"]), idx_true="x", idx_pred="y",category=DataCategory.PUBLI)`|dict, pandas.DataFrame, vaex.dataframe, spark dataframe|

Here is a full-fledged example:

```python
{!docs/compliant_logging/metrics-logging.py!}
```