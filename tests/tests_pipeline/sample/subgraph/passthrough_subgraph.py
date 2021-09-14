# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Constructor for a test pipeline for SmartCompose.
"""
import os
import sys
import argparse

from azure.ml.component import dsl
from azureml.core import Dataset
from shrike.pipeline.pipeline_helper import AMLPipelineHelper

from dataclasses import dataclass
from typing import Dict
from omegaconf import MISSING


class PassthroughSubgraph(AMLPipelineHelper):
    """sample pipeline builder class"""

    @classmethod
    def get_config_class(cls):
        @dataclass
        class passthrough:  # pylint: disable=invalid-name
            input_dataset: str = "dummy_dataset"

        return passthrough

    def build(self, config):
        """Builds a pipeline function for this pipeline.

        Args:
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            pipeline_function: the function to create your pipeline
        """
        # this function just loads modules either locally or remotely depending on use_local list
        stats_passthrough = self.module_load("stats_passthrough")
        stats_passthrough_windows = self.module_load("stats_passthrough_windows")

        # create an instance of a pipeline function using parameters
        @dsl.pipeline(
            name="TutorialPipeline",
            description="Just runs the Probe module in AML",
            default_datastore=config.compute.compliant_datastore,
        )
        def tutorial_pipeline_function(input_dataset):
            """Constructs a sequence of steps to ...

            Args:
                None

            Returns:
                dict: output_path
            """
            stats_step = stats_passthrough(input_path=input_dataset)
            self.apply_recommended_runsettings("stats_passthrough", stats_step)

            stats_step_2 = stats_passthrough_windows(
                input_path=stats_step.outputs.output_path
            )
            self.apply_recommended_runsettings(
                "stats_passthrough_windows", stats_step_2
            )

            return {"output_path": stats_step_2.outputs.output_path}

        return tutorial_pipeline_function

    def pipeline_instance(self, pipeline_function, config):
        """Creates an instance of the pipeline using arguments.

        Args:
            pipeline_function (function): the pipeline function obtained from self.build()
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            pipeline: the instance constructed using build() function
        """
        pipeline_input_dataset = self.dataset_load(config.passthrough.input_dataset)

        runnable_pipeline = pipeline_function(pipeline_input_dataset)
        return runnable_pipeline


if __name__ == "__main__":
    PassthroughSubgraph.main()
