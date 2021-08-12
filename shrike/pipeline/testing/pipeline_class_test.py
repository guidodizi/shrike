# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for testing all module specification:
"""
import json
import os
import logging
import traceback

from shrike.pipeline import AMLPipelineHelper


log = logging.getLogger(__name__)


def get_config_class(pipeline_class):
    """Test if the get_arg_parser() method is in there and behaves correctly"""

    try:
        config_class = pipeline_class.get_config_class()
    except:
        assert (
            False
        ), "getting config class for pipeline class {} resulted in an exception: {}".format(
            pipeline_class.__name__, traceback.format_exc()
        )


def pipeline_required_modules(pipeline_class):
    """Test if the required_modules() returns the right list of modules with all required keys"""
    modules_manifest = pipeline_class.required_modules()

    assert isinstance(
        modules_manifest, dict
    ), "required_modules() must return a dictionary."

    error_log = []
    for module_key, module_description in modules_manifest.items():
        if not isinstance(module_description, dict):
            error_log.append(
                f"values in dictionary returned by required_modules() must be dictionaries (under key={module_key}, found value of type={module_description.__name__})"
            )
            continue
        if "yaml_spec" not in module_description:
            error_log.append(
                f"In pipeline class module {pipeline_class.__name__},"
                + f" module under key={module_key} (in required_modules() function)"
                + " does not provide any yaml_spec key."
                + " You need to give such a yaml_spec path before creating your pull request"
                + " so that we're able to consume this module when running pre-merge tests (detonation chamber)"
            )
        if "remote_module_name" not in module_description:
            error_log.append(
                f"In pipeline class module {pipeline_class.__name__},"
                + f" module under key={module_key} (in required_modules() function)"
                + " does not provide any remote_module_name."
                + " You need to give such a name before creating your pull request"
                + " so that we're able to consume this module when running in production."
            )
        # if "namespace" not in module_description:
        #    error_log.append(
        #        f"In pipeline class module {pipeline_class.__name__},"
        #        + f" module under key={module_key} (in required_modules() function)"
        #        + " does not provide any namespace."
        #    )
        # TODO : verify if the version exists or is in the yaml spec?

    assert not (error_log), (
        f"In pipeline class module {pipeline_class.__name__}, validation of the dictionary returned by required_modules() method shows errors:\n"
        + "\n".join(error_log)
    )


def pipeline_required_subgraphs(pipeline_class):
    """Tests if the required_subgraphs() returns the right list of modules with all requires keys"""
    subgraphs_manifest = pipeline_class.required_subgraphs()

    assert isinstance(
        subgraphs_manifest, dict
    ), "required_subgraphs() must return a dictionary."

    error_log = []
    for subgraph_key, subgraph_class in subgraphs_manifest.items():
        if not issubclass(subgraph_class, AMLPipelineHelper):
            error_log.append(
                f"In pipeline class module {pipeline_class.__name__}, values in dictionary returned by required_subgraphs() must be subclass of AMLPipelineHelper (under key={subgraph_key}, found object {subgraph_class.__name__})"
            )
            continue

    assert not (error_log), (
        f"In pipeline class module {pipeline_class.__name__}, validation of the dictionary returned by required_subgraphs() shows errors: "
        + "\n".join(error_log)
    )


def deeptest_graph_comparison(pipeline_export_file, pipeline_definition_file):
    """Compare a pipeline object to a serialized definition [EXPERIMENTAL]

    Args:
        pipeline_export_file (str): path to pipeline exported file
        pipeline_definition_file (str): path to reference file

    Returns:
        None
    """
    # checks the exported file in temp dir
    assert os.path.isfile(
        pipeline_export_file
    ), f"deeptest_graph_comparison() expects a file as first argument but {pipeline_export_file} does not exist."
    assert os.path.isfile(
        pipeline_definition_file
    ), f"deeptest_graph_comparison() expects a file as second argument but {pipeline_definition_file} does not exist."

    # read the exported graph
    with open(pipeline_export_file, "r") as export_file:
        pipeline = json.loads(export_file.read())
    assert (
        pipeline is not None
    ), f"deeptest_graph_comparison() expects first argument to be a parsable json, instead it found None"

    with open(pipeline_definition_file, "r") as definition_file:
        definition = json.loads(definition_file.read())
    assert (
        definition is not None
    ), f"deeptest_graph_comparison() expects first argument to be a parsable json, instead it found None"

    deeptest_graph(pipeline, definition)


def deeptest_graph(pipeline, definition, path="ROOT"):
    """Recursively compare a pipeline object to a serialized definition [EXPERIMENTAL]

    Args:
        pipeline (json): source for the comparison
        definition (json): target/reference for the comparison
        path (str): current path of the comparison (in the json tree)

    Returns:
        None
    """
    if definition is None:
        # no definition provided, let's stop inspection at this path
        log.info(f"deeptest_graph @ {path}: nop, definition is None")
        return

    # is inspecting a dictionary structure, iterate on keys
    if isinstance(pipeline, dict) and isinstance(definition, dict):
        log.info(f"deeptest_graph @ {path}: checking dictionary")
        for key in definition:
            assert (
                key in pipeline
            ), f"pipeline graph does not have key {key} at level @ {path}"

            # ignoring all ids
            if key in {"id", "node_id", "module_id", "dataset_id"}:
                log.info(f"deeptest_graph @ {path}: ignore id key {key}")
                return

            if (
                key in {"run_settings", "compute_run_settings"}
                and definition[key] is not None
            ):
                # this is a specific kind of key containing a list we're transforming into a dict
                log.info(f"deeptest_graph @ {path}: refactoring key {key} as dict")
                pipeline_run_settings = dict(
                    [(entry["name"], entry) for entry in pipeline[key]]
                )
                definition_run_settings = dict(
                    [(entry["name"], entry) for entry in definition[key]]
                )
                deeptest_graph(
                    pipeline_run_settings,
                    definition_run_settings,
                    path + ".(runsettings)" + key,
                )
            else:
                deeptest_graph(pipeline[key], definition[key], path + "." + key)
        return

    # is inspecting a list structure, each element MUST passed
    # NOTE: this should be improved in case list can be shuffled ?
    if isinstance(pipeline, list) and isinstance(definition, list):
        log.info(f"deeptest_graph @ {path}: checking list")
        for key, entry in enumerate(definition):
            deeptest_graph(pipeline[key], entry, path + "[" + str(key) + "]")
        return

    # if anything else (int, str, unknown), just test plain equality
    log.info(f"deeptest_graph @ {path}: checking equality {pipeline} == {definition}")
    assert pipeline == definition, f"values mismatch @ {path}"
