# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for testing if run.py is aligned with module specification:

> Status: this code relates to the _recipe_ and is a _proposition_
"""
import argparse
import logging
import os
import sys
import pytest
import traceback

from ruamel import yaml

from azure.ml.component._core._component_definition import (
    ComponentDefinition,
    ComponentType,
)

from shrike.pipeline.testing.importer import (
    import_and_test_class,
    dynamic_import_module,
)


log = logging.getLogger(__name__)


def component_spec_yaml_exists_and_is_parsable(component_spec_path):
    """Checks component spec file"""
    assert os.path.isfile(
        component_spec_path
    ), f"Component spec file under path {component_spec_path} could not be found"

    # opens file for testing schema
    with open(component_spec_path, "r") as ifile:
        component_spec_content = ifile.read()
        if "$schema: http://azureml/" in component_spec_content:
            use_component_sdk = True
        else:
            use_component_sdk = False

    # Block unit tests from working with module sdk if not enabled
    if not os.environ.get("MODULE_SDK_ENABLE"):
        assert (
            use_component_sdk
        ), "These unit tests are intentionnally blocked from support Module SDK, which is DEPRECATED. To bypass, create env variable MODULE_SDK_ENABLE."

    if use_component_sdk:
        try:
            definition = ComponentDefinition.load(component_spec_path)
        except BaseException as e:
            assert (
                False
            ), "Failed: failed to load (sdk 2.0) component yaml %r, exception=%r" % (
                component_spec_path,
                e,
            )
    else:
        try:
            with open(component_spec_path, "r") as ifile:
                definition = yaml.safe_load(ifile)
        except BaseException as e:
            assert (
                False
            ), "Failed: failed to load (old style) module yaml %r, exception=%r" % (
                component_spec_path,
                e,
            )

    return definition, use_component_sdk


### TESTING PRIVATE ACR IS IN POLYMERPROD ###


def component_uses_private_acr_modulesdk(component_spec_path, definition, acr_url):
    """Tests base image in private ACR"""
    try:
        base_image_url = definition["implementation"]["container"]["amlEnvironment"][
            "docker"
        ]["baseImage"]
    except KeyError:
        base_image_url = None
        pass

    if base_image_url is not None:
        assert base_image_url.startswith(
            acr_url
        ), "Component(1.5) {} baseImage should be drawn from polymerprod, instead found url {}".format(
            component_spec_path, base_image_url
        )


def component_uses_private_acr_componentsdk(component_spec_path, definition, acr_url):
    """Tests base image in private ACR"""
    definition_type = definition.type
    if definition_type in [
        ComponentType.HDInsightComponent,
        ComponentType.ScopeComponent,
        ComponentType.DataTransferComponent,
    ]:
        return

    try:
        base_image_url = definition.environment.docker.image
    except KeyError:
        base_image_url = None
        pass

    if base_image_url is not None:
        assert base_image_url.startswith(
            acr_url
        ), "Component {} baseImage should be drawn from polymerprod, instead found url {}".format(
            component_spec_path, base_image_url
        )


def component_uses_private_acr(component_spec_path, acr_url):
    """Tests base image in private ACR"""
    definition, use_component_sdk = component_spec_yaml_exists_and_is_parsable(
        component_spec_path
    )

    if use_component_sdk:
        component_uses_private_acr_componentsdk(
            component_spec_path, definition, acr_url
        )
    else:
        component_uses_private_acr_modulesdk(component_spec_path, definition, acr_url)


### TESTING PRIVATE FEED ###


def component_uses_private_python_feed(component_spec_path, feed_url):
    """Tests private python feed referenced in conda"""
    definition, use_component_sdk = component_spec_yaml_exists_and_is_parsable(
        component_spec_path
    )

    if use_component_sdk:
        if definition.type in [
            ComponentType.HDInsightComponent,
            ComponentType.ScopeComponent,
            ComponentType.DataTransferComponent,
        ]:
            return

        try:
            conda_deps_yaml = definition.environment.conda.conda_dependencies._to_dict()
        except KeyError:
            conda_deps_yaml = None
            pass
    else:
        job_type = str(definition["jobType"]).lower()
        if job_type in ["hdinsight", "scopecomponent", "datatransfercomponent"]:
            # hdi/scope/datatransfer jobs don't have python feed
            return
        if job_type == "parallel":
            try:
                conda_deps_path = definition["implementation"]["parallel"][
                    "amlEnvironment"
                ]["python"]["condaDependenciesFile"]
            except KeyError:
                conda_deps_path = None
                pass
        else:
            try:
                conda_deps_path = definition["implementation"]["container"][
                    "amlEnvironment"
                ]["python"]["condaDependenciesFile"]
            except KeyError:
                conda_deps_path = None
                pass

        if conda_deps_path is None:
            # no conda yaml provided, nothing to do here
            return

        conda_deps_abspath = os.path.join(
            os.path.dirname(component_spec_path), conda_deps_path
        )
        assert os.path.isfile(
            conda_deps_abspath
        ), "Component {} specified a conda_dependencies_file {} that cannot be found (abspath: {})".format(
            component_spec_path, conda_deps_path, conda_deps_abspath
        )

        try:
            with open(conda_deps_abspath, "r") as ifile:
                conda_deps_yaml = yaml.safe_load(ifile)
        except:
            assert (
                False
            ), "Component {} conda_dependencies_file under path {} should be yaml parsable, but loading it raised an exception: {}".format(
                component_spec_path, conda_deps_abspath, traceback.format_exc()
            )

    if conda_deps_yaml is None:
        # no conda yaml provided, nothing to do here
        return

    if "channels" in conda_deps_yaml:
        assert conda_deps_yaml["channels"] == [
            "."
        ], "In conda deps, no channels must be specified, or use . as channel"

    if "dependencies" in conda_deps_yaml:
        for entry in conda_deps_yaml["dependencies"]:
            if "pip" in entry and isinstance(entry, dict):
                assert (
                    f"--index-url {feed_url}" in entry["pip"]
                ), "conda deps must reference private python feed under pip dependencies."


### TEST IMPORTING RUN PY ###


def find_run_py_in_command(definition, use_component_sdk):
    """Finds runnable python script in command"""
    run_py_command, definition_command = None, None
    if use_component_sdk:
        definition_type = definition.type
        if definition_type == ComponentType.HDInsightComponent:
            run_py_command = definition.file
            definition_command = definition.args
        elif definition_type == ComponentType.DistributedComponent:
            # run_py_command not provided, we need to find it
            definition_command = definition.launcher.additional_arguments
        elif definition_type == ComponentType.ParallelComponent:
            run_py_command = definition.entry
            definition_command = definition.args
        elif definition_type == ComponentType.CommandComponent:
            # run_py_command not provided, we need to find it
            definition_command = definition.command
        elif definition_type not in [
            ComponentType.ScopeComponent,
            ComponentType.DataTransferComponent,
        ]:
            raise Exception(
                f"Component type {definition_type} is not supported in the helper code unit tests (yet)."
            )

        if (
            run_py_command is None
            and definition.type != ComponentType.ScopeComponent
            and definition.type != ComponentType.DataTransferComponent
        ):
            # search for python script
            for entry in definition_command.split(" "):
                if entry.endswith(".py"):
                    run_py_command = entry
                    break
            else:
                assert (
                    False
                ), "Could not find any script name like *.py in component command {}".format(
                    definition_command
                )
    else:
        job_type = str(definition["jobType"]).lower()
        if job_type == "hdinsight":
            run_py_command = definition["implementation"]["hdinsight"]["file"]
            definition_command = run_py_command
        elif job_type == "parallel":
            run_py_command = definition["implementation"]["parallel"]["entry"]
            definition_command = run_py_command
        elif job_type not in ["scopecomponent", "datatransfercomponent"]:
            definition_command = definition["implementation"]["container"]["command"]
            for entry in definition_command:
                if entry.endswith(".py"):
                    run_py_command = entry
                    break
            else:
                assert (
                    False
                ), "Could not find any script name like *.py  in component command {}".format(
                    definition_command.split(" ")
                )

    return run_py_command, definition_command


def component_run_py_import(component_spec_path):
    """Try importing run.py, just to check if basic script passes syntax/imports checks"""
    definition, use_component_sdk = component_spec_yaml_exists_and_is_parsable(
        component_spec_path
    )

    run_py_command, definition_command = find_run_py_in_command(
        definition, use_component_sdk
    )

    component_import_path = os.path.dirname(component_spec_path)
    run_py_absdir = os.path.join(component_import_path, run_py_command)
    assert os.path.isfile(
        run_py_absdir
    ), "Component {} has command {} using a python script {} that cannot be found".format(
        component_spec_path, definition_command, run_py_command
    )

    if component_import_path not in sys.path:
        sys.path.insert(0, component_import_path)

    try:
        spec, mod = dynamic_import_module(run_py_absdir)
    except:
        assert False, "importing {} resulted in an exception: {}".format(
            run_py_absdir, traceback.format_exc()
        )


def component_run_get_arg_parser(component_spec_path):
    """Tests is module run.py has function get_arg_parser(parser)"""
    definition, use_component_sdk = component_spec_yaml_exists_and_is_parsable(
        component_spec_path
    )

    run_py_command, definition_command = find_run_py_in_command(
        definition, use_component_sdk
    )

    component_import_path = os.path.dirname(component_spec_path)
    run_py_absdir = os.path.join(component_import_path, run_py_command)
    assert os.path.isfile(
        run_py_absdir
    ), "Component {} has command {} using a python script {} that cannot be found".format(
        component_spec_path, definition_command, run_py_command
    )

    if component_import_path not in sys.path:
        sys.path.insert(0, component_import_path)

    try:
        assert os.path.isfile(
            run_py_absdir
        ), f"module command {run_py_absdir} should exist"
        get_arg_parser_func = import_and_test_class(run_py_absdir, "get_arg_parser")
    except:
        assert (
            False
        ), "importing {} function get_arg_parser() resulted in an exception: {}".format(
            run_py_absdir, traceback.format_exc()
        )

    try:
        returned_parser = get_arg_parser_func()
    except:
        assert (
            False
        ), "Component script {}.get_arg_parser() should be able to run on argparse.ArgumentParser, but raised an exception: {}".format(
            run_py_absdir, traceback.format_exc()
        )

    assert (
        returned_parser is not None
    ), "component script {}.get_arg_parser() is supposed to return a parser when provided with None, please add 'return parser' at the end of the function.".format(
        run_py_absdir
    )

    try:
        parser = argparse.ArgumentParser()
        returned_parser = get_arg_parser_func(parser)
    except:
        assert (
            False
        ), "Component script {}.get_arg_parser() should be able to run on argparse.ArgumentParser, but raised an exception: {}".format(
            run_py_absdir, traceback.format_exc()
        )

    assert (
        returned_parser is not None
    ), "Component script {}.get_arg_parser() is not supposed to return None, please add 'return parser' at the end of the function.".format(
        run_py_absdir
    )

    # test object equality
    assert (
        returned_parser is parser
    ), "Component script {}.get_arg_parser() is supposed to return the parser it was provided, please do not create a new instance if provided with a parser.".format(
        run_py_absdir
    )

    return parser


def _generate_fake_input_arg_componentsdk(arg_spec):
    """Generate a fake argument value for inputs of module spec

    Args:
        arg_spec (dict) : argument specification from yaml module spec

    Returns:
        object: sample fake value

    Raises:
        NotImplementedError: if arg type is not implemented
    """
    if "AzureMLDataset" in arg_spec.type:
        return "/mnt/fakeinputdatasetpath"
    if "AnyDirectory" in arg_spec.type:
        return "/mnt/fakeinputdirectorypath"
    if "AnyFile" in arg_spec.type:
        return "/mnt/fakeinputfilepath/file.txt"
    if arg_spec.default:
        return arg_spec.default
    if "String" in arg_spec.type:
        return "0"
    if "Integer" in arg_spec.type:
        return arg_spec.min or arg_spec.max or "0"
    if "Boolean" in arg_spec.type:
        return False
    if "Float" in arg_spec.type:
        return arg_spec.min or arg_spec.max or "0.32"
    if "Enum" in arg_spec.type:
        return arg_spec.enum[0]
    raise NotImplementedError(
        "input type {} is not implemented in our test suite yet".format(arg_spec.type)
    )


def generate_component_arguments_componentsdk(
    component_spec, arg, output_script_arguments
):
    """Recursively generate fake arguments to test script argparse.

    Args:
        component_spec (dict): module specification in yaml
        arg (list or str or dict) : argument specification
        output_script_arguments (list) : output

    Returns:
        list: output_script_arguments
    """
    log.info(f"generate_component_arguments(spec, {arg}, ...)")
    if isinstance(arg, list):  # optional argument or root list
        for entry in arg:
            generate_component_arguments_componentsdk(
                component_spec, entry, output_script_arguments
            )
    elif isinstance(arg, str) and arg.startswith("{"):
        io_key = arg.lstrip("{").rstrip("}")
        if io_key.startswith("inputs."):
            input_key = io_key[7:]
            log.info("inputs keys: " + " ".join([key for key in component_spec.inputs]))
            log.info(
                "parameter keys: "
                + " ".join([key for key in component_spec.parameters])
            )
            if input_key in component_spec.inputs:
                output_script_arguments.append(
                    str(
                        _generate_fake_input_arg_componentsdk(
                            component_spec.inputs[input_key]
                        )
                    )
                )
            elif input_key in component_spec.parameters:
                output_script_arguments.append(
                    str(
                        _generate_fake_input_arg_componentsdk(
                            component_spec.parameters[input_key]
                        )
                    )
                )
            else:
                raise Exception(
                    f"Input key {input_key} is neither an input or a parameter"
                )
        elif io_key.startswith("outputs."):
            output_key = io_key[8:]
            log.info(
                "outputs keys: " + " ".join([key for key in component_spec.outputs])
            )
            output_script_arguments.append(
                str(
                    _generate_fake_input_arg_componentsdk(
                        component_spec.outputs[output_key]
                    )
                )
            )
        else:
            raise NotImplementedError(
                "In argument spec {}, I/O key arg spec is not supported {}".format(
                    arg, io_key
                )
            )
    elif isinstance(arg, str):
        output_script_arguments.append(arg)
    elif isinstance(arg, dict):  # for old module def
        if "inputValue" in arg:
            # find in inputs
            for i_spec in component_spec.inputs:
                if i_spec["name"] == arg["inputValue"]:
                    output_script_arguments.append(
                        str(_generate_fake_input_arg_componentsdk(i_spec))
                    )
        elif "inputPath" in arg:
            # find in inputs
            for i_spec in component_spec.inputs:
                if i_spec["name"] == arg["inputPath"]:
                    output_script_arguments.append(
                        str(_generate_fake_input_arg_componentsdk(i_spec))
                    )
        elif "outputPath" in arg:
            # find in outputs
            output_script_arguments.append("/mnt/fakeoutputpath")

    return output_script_arguments


def _generate_fake_input_arg_modulesdk(arg_spec):
    """Generate a fake argument value for inputs of module spec

    Args:
        arg_spec (dict) : argument specification from yaml module spec

    Returns:
        object: sample fake value

    Raises:
        NotImplementedError: if arg type is not implemented
    """
    if arg_spec["type"] == "AzureMLDataset":
        return "/mnt/fakeinputdatasetpath"
    if arg_spec["type"] == "AnyDirectory":
        return "/mnt/fakeinputdirectorypath"
    if arg_spec["type"] == "AnyFile":
        return "/mnt/fakeinputfilepath/file.txt"
    if "default" in arg_spec:
        return arg_spec["default"]
    if arg_spec["type"] == "String":
        return "0"
    if arg_spec["type"] == "Integer":
        return "0"
    if arg_spec["type"] == "Boolean":
        return False
    if arg_spec["type"] == "Float":
        return "0.32"
    if arg_spec["type"] == "Enum":
        return arg_spec["options"][0]
    raise NotImplementedError(
        "input type {} is not implemented in our test suite yet".format(
            arg_spec["type"]
        )
    )


def generate_component_arguments_modulesdk(module_spec, arg, output_script_arguments):
    """Recursively generate fake arguments to test script argparse.

    Args:
        component_spec (dict): module specification in yaml
        arg (list or str or dict) : argument specification
        output_script_arguments (list) : output

    Returns:
        list: output_script_arguments
    """
    if isinstance(arg, list):  # optional argument or root list
        for entry in arg:
            generate_component_arguments_modulesdk(
                module_spec, entry, output_script_arguments
            )
    elif isinstance(arg, str):
        output_script_arguments.append(arg)
    elif isinstance(arg, dict):
        if "inputValue" in arg:
            # find in inputs
            for i_spec in module_spec["inputs"]:
                if i_spec["name"] == arg["inputValue"]:
                    output_script_arguments.append(
                        str(_generate_fake_input_arg_modulesdk(i_spec))
                    )
        elif "inputPath" in arg:
            # find in inputs
            for i_spec in module_spec["inputs"]:
                if i_spec["name"] == arg["inputPath"]:
                    output_script_arguments.append(
                        str(_generate_fake_input_arg_modulesdk(i_spec))
                    )
        elif "outputPath" in arg:
            # find in outputs
            output_script_arguments.append("/mnt/fakeoutputpath")

    return output_script_arguments


def if_arguments_from_component_spec_match_script_argparse(component_spec_path):
    """Tests alignment between spec arguments and script parser arguments"""
    # assuming we have a yaml spec file that is loadable
    definition, use_component_sdk = component_spec_yaml_exists_and_is_parsable(
        component_spec_path
    )

    # assuming we can import the get_arg_parser() function
    parser = component_run_get_arg_parser(component_spec_path)

    run_py_command, definition_command = find_run_py_in_command(
        definition, use_component_sdk
    )

    if use_component_sdk:
        arguments_spec = [
            entry.lstrip("[").rstrip("]") for entry in definition_command.split(" ")
        ]

        if arguments_spec[0].startswith("python"):
            arguments_spec.pop(0)
        if arguments_spec[0].endswith(".py"):
            arguments_spec.pop(0)

        script_arguments = []
        generate_component_arguments_componentsdk(
            definition, arguments_spec, script_arguments
        )
    else:
        job_type = str(definition["jobType"]).lower()
        if job_type == "hdinsight":
            arguments_spec = definition["implementation"]["hdinsight"]["args"]
        elif job_type == "parallel":
            arguments_spec = definition["implementation"]["parallel"]["args"]
        elif job_type not in ["scopecomponent", "datatransfercomponent"]:
            arguments_spec = definition["implementation"]["container"]["args"]

        script_arguments = []
        generate_component_arguments_modulesdk(
            definition, arguments_spec, script_arguments
        )

    try:
        _, unknown_args = parser.parse_known_args(script_arguments)
    except:
        assert (
            False
        ), "Component {}, in run.py, parse_known_args() should be able to parse {}, instead raised an exception: {}".format(
            component_spec_path, script_arguments, traceback.format_exc()
        )
    assert (
        len(unknown_args) == 0
    ), "Component {}, while calling run.py with args {}, parsing arguments from module spec should not return unknown args, instead we observed unknown args : {}".format(
        component_spec_path, script_arguments, unknown_args
    )


def script_main_with_synthetic_arguments(module, mocker):
    """Try importing run.py, just to check if basic script passes syntax/imports checks"""
    paths = _get_module_paths(module)

    # assuming we have a yaml spec file that is loadable
    module_spec = module_spec_yaml_exists_and_is_parsable(module)

    # import module to get main() function
    if paths.module_spec_absdir not in sys.path:
        sys.path.insert(0, paths.module_spec_absdir)

    try:
        spec, mod = dynamic_import_module(paths.module_import_path)
    except:
        assert False, "importing {} resulted in an exception: {}".format(
            paths.module_import_path, traceback.format_exc()
        )

    if module_spec["jobType"].lower() == "hdinsight":
        arguments_spec = module_spec["implementation"]["hdinsight"]["args"]
    elif (
        module_spec["jobType"].lower() != "scopecomponent"
        and module_spec["jobType"].lower() != "datatransfercomponent"
    ):
        arguments_spec = module_spec["implementation"]["container"]["args"]
    script_arguments = []
    generate_argument(module_spec, arguments_spec, script_arguments)

    log.info(script_arguments)
    # https://medium.com/python-pandemonium/testing-sys-exit-with-pytest-10c6e5f7726f
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        mod.main(script_arguments + ["-h"])

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
