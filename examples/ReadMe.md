# Some minimal examples of using shrike to submit a graph pipeline

## Setup

- Clone the current repository and set `examples` as your working directory.
- Set up and activate a new Conda environment:
  `conda create --name shrike-examples-env python=3.7 -y`,
  `conda activate shrike-examples-env`.
- Install the `shrike` dependencies:
  `pip install -r requirements.txt`

## How to run the examples

You will find 2 example experiments using shrike.

1. [demo_component_with_parameter.py](./pipelines/experiments/demo_component_with_parameter.py) is for running a [basic experiment](https://ml.azure.com/runs/2ba28186-b358-4991-877c-a3cc374bb945?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47) demonstrating how to consume a parameter value and operate on it.
2. [demo_count_rows_and_log.py](./pipelines/experiments/demo_count_rows_and_log.py) is for running a [basic experiment](https://ml.azure.com/runs/d39074ac-09ec-4b74-b5c1-8a911dd1dfed?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47) demonstrating how to use shrike for logging, and how to consume a dataset.

To try and submit these experiments, use the command given in the comments at the top of each python file.

## Further resources

If you want to learn more about how to use shrike, please consider checking out the [problem set](https://github.com/Azure/azure-ml-problem-sets/blob/main/shrike-examples/ReadMe.md).