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
from omegaconf import MISSING, OmegaConf

# NOTE: if you need to import from pipelines.*
SAMPLE_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SAMPLE_ROOT_PATH not in sys.path:
    print(f"Adding to path: {SAMPLE_ROOT_PATH}")
    sys.path.append(str(SAMPLE_ROOT_PATH))

from subgraph.build_time_conditional_passthrough_subgraph import (
    BuidTimeConditionalPassthroughSubgraph,
)
from subgraph.passthrough_subgraph import PassthroughSubgraph


class CanaryPipelineStatsPassthrough(AMLPipelineHelper):
    """sample pipeline builder class"""

    @classmethod
    def get_config_class(cls):
        @dataclass
        class passthrough:  # pylint: disable=invalid-name
            input_dataset: str = "dummy_dataset"

        return passthrough

    @classmethod
    def required_subgraphs(cls):
        return {
            "PassthroughSubgraph": PassthroughSubgraph,
            "BuidTimeConditionalPassthrough": BuidTimeConditionalPassthroughSubgraph,
        }

    def build(self, config):
        """Builds a pipeline function for this pipeline.

        Args:
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            pipeline_function: the function to create your pipeline
        """
        # load subgrqaph without custom paramters outside pipeline definition
        passthrough_func = self.subgraph_load("PassthroughSubgraph")
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

            passthrough_step = passthrough_func(input_dataset)

            # load subgrqaph without custom paramters inside pipeline definition
            conditional_passthrough_1_func = self.subgraph_load(
                "BuidTimeConditionalPassthrough",
                custom_config=OmegaConf.create(
                    {"subgraph": {"passthrough_windows": True}}
                ),
            )
            conditional_passthrough_1_step = conditional_passthrough_1_func(
                input_dataset
            )

            conditional_passthrough_2_func = self.subgraph_load(
                "BuidTimeConditionalPassthrough",
                custom_config=OmegaConf.create(
                    {"subgraph": {"passthrough_windows": False}}
                ),
            )
            conditional_passthrough_2_step = conditional_passthrough_2_func(
                input_dataset
            )

            return {"output_path": conditional_passthrough_1_step.outputs.output_path}

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
    CanaryPipelineStatsPassthrough.main()
