# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Canary helper code
"""


from azureml.core import Dataset
from azureml.data.datapath import DataPath
import logging


log = logging.getLogger(__name__)


def get_repo_info():
    """[EXPERIMENTAL] Obtains info on the current repo the code is in.

    Returns:
        dict: git meta data"""
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        branch = repo.active_branch
        head = repo.head
        return {
            "git": repo.remotes.origin.url,
            "branch": branch.name,
            "commit": head.commit.hexsha,
            "last_known_author": head.commit.author.name,
        }
    except:
        return {"git": "n/a"}


def test_pipeline_step_metrics(pipeline_run, expected_metrics):
    """Tests a pipeline run against a set of expected metrics.

    Args:
        pipeline_run (PipelineRun): the AzureML pipeline run
        expected_metrics (dict): defines the tests to execute

    Returns:
        List: errors collected during tests

    Notes:
        example entries in expected_metrics

        "SelectJsonField" : [{"row" : {"name" : "output", "key" : "size", "value" : 369559}}],
        tests module "SelectJsonField" for a metric row named "output", checks key "size" must have value 369559

        "tokenizerparallel" : [{"metric" : {"key" : "Failed Items", "value" : 0}}],
        tests module "tokenizerparallel" for a metric named "Failed Items", value must be 0
    """
    errors = []

    log.info("Looping through PipelineRun steps to test metrics...")
    for step in pipeline_run.get_steps():
        log.info(f"Checking status of step {step.name}...")

        observed_metrics = step.get_metrics()
        log.info(f"Step Metrics: {observed_metrics}")

        status = step.get_status()
        if status != "Finished":
            errors.append(f"Pipeline step {step.name} status is {status} != Finished")

        if step.name in expected_metrics:
            for expected_metric_test in expected_metrics[step.name]:
                if "row" in expected_metric_test:
                    log.info(f"Checking metrics, looking for {expected_metric_test}")
                    row_key = expected_metric_test["row"]["name"]
                    metric_key = expected_metric_test["row"]["key"]
                    expected_value = expected_metric_test["row"]["value"]
                    if row_key not in observed_metrics:
                        errors.append(
                            f"Step {step.name} metric row '{row_key}' not available in observed metrics {observed_metrics}"
                        )
                    elif metric_key not in observed_metrics[row_key]:
                        errors.append(
                            f"Step {step.name} metric row '{row_key}' does not have a metric '{metric_key}' in observed metrics {observed_metrics[row_key]}"
                        )
                    elif observed_metrics[row_key][metric_key] != expected_value:
                        errors.append(
                            f"Step {step.name} metric row '{row_key}' - metric '{metric_key}' - does not have expected value {expected_value} in observed metrics {observed_metrics[row_key]}"
                        )
                if "metric" in expected_metric_test:
                    log.info(f"Checking metrics, looking for {expected_metric_test}")
                    metric_key = expected_metric_test["metric"]["key"]
                    expected_value = expected_metric_test["metric"]["value"]
                    if metric_key not in observed_metrics:
                        errors.append(
                            f"Step {step.name} metric '{metric_key}' not available in observed metrics {observed_metrics}"
                        )
                    elif observed_metrics[metric_key] != expected_value:
                        errors.append(
                            f"Step {step.name} metric row '{metric_key}' does not have expected value {expected_value} in observed metrics {observed_metrics[metric_key]}"
                        )

    return errors


def test_pipeline_step_output(pipeline_run, step_name, output_name, **kwargs):
    """Verify a given pipeline output for some basic checks.

    Args:
        pipeline_run (PipelineRun): the pipeline run
        step_name (str): name of the step to check
        output_name (str): name of the output to check
        **kwargs: Arbitrary keyword arguments defining the test

    Kwargs:
        length (int) : to verify the length

    Returns:
        dict: results
    """
    pipeline_step = pipeline_run.find_step_run(step_name)
    results = {"errors": []}

    if not pipeline_step:
        results[
            "exception"
        ] = f"Could not find step {step_name} in pipeline {pipeline_run._run_id}."
        return results

    output_port = pipeline_step[0].get_output_data(output_name)
    if not output_port:
        results[
            "exception"
        ] = f"Could not find output {output_name} in step {step_name} in pipeline {pipeline_run._run_id}."
        return results

    data_reference = output_port._data_reference

    data_path = DataPath(
        datastore=data_reference.datastore,
        path_on_datastore=data_reference.path_on_datastore,
        name=data_reference.data_reference_name,
    )

    if kwargs.get("length"):
        expected_length = kwargs.get("length")
        log.info(
            f"Checking count={expected_length} of files for step {step_name} output {output_name}..."
        )
        data_set = Dataset.File.from_files(data_path)

        files_list = data_set.to_path()

        if expected_length < 0:
            # test any length > 0
            results["length"] = {"expected": ">0", "observed": len(files_list)}
            if results["length"]["observed"] == 0:
                message = """Length mismatch in output {output_name} in step {step_name} in pipeline {run_id}. Expected len {a} found {b}.""".format(
                    output_name=output_name,
                    step_name=step_name,
                    run_id=pipeline_run._run_id,
                    b=results["length"]["observed"],
                    a=results["length"]["expected"],
                )
                # logging.error(message)
                results["errors"].append(message)
        else:
            results["length"] = {
                "expected": expected_length,
                "observed": len(files_list),
            }

            if results["length"]["observed"] != results["length"]["expected"]:
                message = """Length mismatch in output {output_name} in step {step_name} in pipeline {run_id}. Expected len {a} found {b}.""".format(
                    output_name=output_name,
                    step_name=step_name,
                    run_id=pipeline_run._run_id,
                    b=results["length"]["observed"],
                    a=results["length"]["expected"],
                )
                # logging.error(message)
                results["errors"].append(message)

    return results
