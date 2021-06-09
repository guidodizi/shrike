# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This runnable graph can be used to onvert a tsv to cosmos ss.
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys
from dataclasses import dataclass
from typing import Optional

from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper

# NOTE: if you need to import from pipelines.*
SMARTCOMPOSE_ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if SMARTCOMPOSE_ROOT_PATH not in sys.path:
    print(f"Adding to path: {SMARTCOMPOSE_ROOT_PATH}")
    sys.path.append(str(SMARTCOMPOSE_ROOT_PATH))


class ConvertTsvToSSPipeline(AMLPipelineHelper):
    """Runnable/reusable pipeline helper class

    This class inherits from AMLPipelineHelper which provides
    helper functions to create reusable production pipelines for SmartCompose.
    """

    @classmethod
    def get_config_class(cls):
        """Returns the config object (dataclass) for this runnable script.

        Returns:
            dataclass: class for configuring this runnable pipeline.
        """

        @dataclass
        class converttsvtossexample:  # pylint: disable=invalid-name
            """Config object constructed as a dataclass.

            NOTE: the name of this class will be used as namespace in your config yaml file.
            See conf/reference/lstmtraining.yaml for an example.
            """

            # input datasets names (and versions if necessary)
            TextData: str = "dummy_data"
            TextData_version: Optional[str] = "latest"

        # return the dataclass itself
        # for helper class to construct config file
        return converttsvtossexample

    def build(self, config):
        """Builds a pipeline function for this pipeline using AzureML SDK (dsl.pipeline).

        This method should build your graph using the provided config object.
        Your pipeline config will be under config.CONFIGNAME.*
        where CONFIGNAME is the name of the dataclass returned by get_config_class()

        This method returns a constructed pipeline function (decorated with @dsl.pipeline).

        Args:
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            dsl.pipeline: the function to create your pipeline
        """
        # helper function below loads the module from registered or local version
        # depending on your config run.use_local
        convert2ss = self.module_load("convert2ss")

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(
            name="Convert TSV to SS",
            description="Convert TSV to SS",
            default_datastore=config.compute.compliant_datastore,
        )
        def training_pipeline_function(TextData):
            """Pipeline function for this graph.

            Args:
                TextData: input tsv
                ExtractionClause: str following pattern 'Label:string, Id:string'

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            # general syntax:
            # module_instance = module_class(input=data, param=value)
            # or
            # subgraph_instance = subgraph_function(input=data, param=value)

            convert2ss_step = convert2ss(
                TextData=TextData,
                ExtractionClause="",
            )
            # each module should be followed by this call
            # in enabled the helper class to fill-up all the settings for this module instance
            self.apply_smart_runsettings(
                convert2ss_step,
                scope=True,
                adla_account_name="",  # TODO: config.
                custom_job_name_suffix="test",
                scope_param="-tokens 50",
            )
            convert2ss_step.outputs.SSPath.configure(
                path_on_datastore="/testScopeScript/{run-id}/outputdata.ss"
            )  # TODO: config. for datastore=config.compute.compliant_datastore,

            # return {key: output}
            return {"output_ss": convert2ss_step.outputs.SSPath}

        # finally return the function itself to be built by helper code
        return training_pipeline_function

    def pipeline_instance(self, pipeline_function, config):
        """Given a pipeline function, creates a runnable instance based on provided config.

        This is used only when calling this as a runnable pipeline using .main() function (see below).
        The goal of this function is to map the config to the pipeline_function inputs and params.

        Args:
            pipeline_function (function): the pipeline function obtained from self.build()
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            azureml.core.Pipeline: the instance constructed with its inputs and params.
        """
        # NOTE: self.dataset_load() helps to load the dataset based on its name and version
        TextData = self.dataset_load(
            name=config.converttsvtossexample.TextData,
            version=config.converttsvtossexample.TextData_version,
        )

        # when all inputs are obtained, we call the pipeline function
        experiment_pipeline = pipeline_function(TextData)

        # and we return that function so that helper can run it.
        return experiment_pipeline


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    ConvertTsvToSSPipeline.main()
