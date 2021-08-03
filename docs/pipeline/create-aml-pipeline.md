# Instructions for creating a reusable AML pipeline using shrike.pipeline


To enjoy this doc, you need to:

1. have already setup your python environment with the AML SDK following [these instructions](https://aka.ms/aml/pythonenvsetup) and cloned the accelerator repository as described in the "Set up" section [here](https://dev.azure.com/msdata/Vienna/_wiki/wikis/aml-1p-onboarding/20452/Run-your-first-AML-experiments?anchor=set-up);
2. have access to an AML workspace

## Motivation
The Azure ML pipeline helper class `AMLPipelineHelper` in the `shrike` package was developed with the goal of helping data scientists to more easily create reusable pipelines. These instructions explain how to use the Azure ML pipeline helper class.

## 1. Review an existing  Azure ML pipeline created using the Azure ML pipeline helper class
The [accelerator repository](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation&version=GBmain&_a=contents) already has examples of pipelines created using the pipeline helper class. We will now have an overview of the structure of the two most important directories (`components` and `pipelines`, under `aml-ds/recipes/compliant-experimentation`) and go over the key files defining these pipelines.

### 1.1 "components" directory
This is where the components are defined, one folder per component. Each folder contains the following files:

- `component_spec.yaml`: this is where the component's inputs, outputs and parameters are defined. This is the Azure ML equivalent to the component manifest in &AElig;ther. 
- `component_env.yaml`: this is where the component dependencies are listed (not required for HDI components).
- `run.py`: this is the python file actually run in Azure ML; in most cases, it is just importing a python file from elsewhere in the repo.

Further reading on components is available [here](https://aka.ms/aml/creatingnewmodules).

### 1.2 "pipelines" directory

This is where the graphs, a.k.a. pipelines, are defined. Here is what you will find in its subdirectories:

- The `config` directory contains the config files which contain the parameter values, organized in four sub-folders: `experiments` which contains the overall graph configs, then `aml` and `compute` which contain auxiliary config files referred to in the graph configs. The `modules` folder hosts the file where the components are defined (by their key, name, default version, and location of the component specification file). _Once you have created new components, you will need to add them to that file._
- The `subgraphs` directory contain python files that define graphs that are not meant to be used on their own but as part of larger graphs. There is a demo subgraph available there, which consists of 2 `probe` components chained after each other.
- The `experiments` directory contain the python files whichactually define the graphs.

Now let's take a closer look at the definition of a graph in python. We will stick with the demo graph for eyes-off and open the `demograph_eyesoff.py` file in the `experiments` folder. The key parts are listed below.
- The `required_subgraphs()` function ([line 37](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyesoff.py&version=GBmain&line=37&lineEnd=47&lineStartColumn=5&lineEndColumn=46&lineStyle=plain&_a=contents), also shown below) defines the subgraphs that are used in the graph.

```python
# line 37
def required_subgraphs(cls):
    """Declare dependencies on other subgraphs to allow AMLPipelineHelper to build them for you.

    This method should return a dictionary:
    - Keys will be used in self.subgraph_load(key) to build each required subgraph.
    - Values are classes inheriting from AMLPipelineHelper

    Returns:
        dict[str->AMLPipelineHelper]: dictionary of subgraphs used for building this one.
    """
    return {"DemoSubgraph": DemoSubgraph}
```

- The `build()` function, well,  builds the graph.

    - First, the required subgraph is loaded in [line 62](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyesoff.py&version=GBmain&line=62&lineEnd=62&lineStartColumn=9&lineEndColumn=60&lineStyle=plain&_a=contents): 
    ```python
    probe_subgraph = self.subgraph_load("DemoSubgraph")
    ```
    - Then we define a pipeline function for the graph starting [line 70](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyesoff.py&version=GBmain&line=70&lineEnd=70&lineStartColumn=9&lineEndColumn=50&lineStyle=plain&_a=contents). This is where all the components and subgraphs are given their parameters and inputs. Note how the parameter values are read from the config files. To see how the outputs of some components can be used as inputs of the following components see [here in the subgraph python file](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fsubgraphs%2Fdemosubgraph.py&version=GBmain&line=79&lineEnd=128&lineStartColumn=9&lineEndColumn=80&lineStyle=plain&_a=contents).
    - For the time being, we have to manually apply run settings to every component. In the future, this will not be necessary anymore. For the current example, it is also done in the subgraph python file, by calling the [`apply_recommended_runsettings()` function](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fsubgraphs%2Fdemosubgraph.py&version=GBmain&line=110&lineEnd=112&lineStartColumn=13&lineEndColumn=14&lineStyle=plain&_a=contents).

- The `pipeline_instance()` function creates a runnable instance of the pipeline.

    - The input dataset is defined in [lines 104-107](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyesoff.py&version=GBmain&line=104&lineEnd=107&lineStartColumn=9&lineEndColumn=10&lineStyle=plain&_a=contents), by calling the `dataset_load()` function with the _name_ and _version_ values provided in the config file.
    - The pipeline function is then called with the input data as argument.

Next, let's open the [`demograph_eyesoff.yaml` config file](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fexperiments%2Fdemograph_eyesoff.yaml&version=GBmain&_a=contents) under the `pipelines/config/experiments` directory, and note how the other config files are referenced, and how the parameters are organized in sections. We also explain config files in more details in this page: [Configure your pipeline](https://aka.ms/aml/configpipeline).

Finally, below is the command to run this existing pipeline (a very basic demo pipeline):

`python pipelines/experiments/demograph_eyesoff.py --config-dir pipelines/config --config-name experiments/demograph_eyesoff run.submit=True`

## 2. Create your own simple Azure ML pipeline using the pipeline helper class and an already existing component

In this section, We will create a pipeline graph consisting of a single component called `probe`, which is readily available in the accelerator repository. We will pass the parameters through a config file.

**Procedure:**

- [1] For creating your own pipeline, we invite you to start from an already existing pipeline definition such as [`demograph_eyeson.py`](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyeson.py&version=GBmain&_a=contents) and build from there. Just copy `demograph_eyeson.py`, rename it as `demograph_workshop.py`, update the contents accordingly, and put it under the same directory (i.e., `pipelines/experiments`).
The important parts to modify for this file are those listed in the section on key files above: `build()`, and `pipeline_instance()` (since we won't be using a subgraph, we don't need to worry about the `required_subgraphs` part).
- [2] To prepare the YAML config file, start from an existing example, such as [`demograph_eyeson.yaml`](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fexperiments%2Fdemograph_eyeson.yaml&version=GBmain&_a=contents). Just copy `demograph_eyeson.yaml`, rename it as `demograph_eyeson_workshop.yaml`, update the contents accordingly, and put it under the same directory (i.e., `pipelines/config/experiments`). 
The important parts are defining the component parameter values, and declaring that we want to use the local version of the component (argument `use_local`) for `probe`. 
    > Note: you will also need to update two auxiliary config files (`eyesoff.yaml`/`eyeson.yaml` file under directory `pipelines/config/aml` and `eyesoff.yaml`/`eyeson.yaml` under directory `pipelines/config/compute`), referenced by this main config file `demograph_eyeson.yaml`, to point to the Azure ML workspace and compute targets to which you have access.

And now you should be able to run your pipeline using the following command:

`python pipelines/experiments/demograph_eyesoff.py --config-dir pipelines/config --config-name experiments/demograph_eyesoff run.submit=True`

If you are using an eyes-on workspace, you will also need to update the base image info in `component_spec.yaml` since only eyes-off workspaces can connect to the polymer prod ACR which hosts the base image.

When a parameter is not specified in the config file, you need to use + when overriding directly from command line. Otherwise there'll be errors. For example, if `run.submit` is not in the config file, you need to use
`python pipelines/experiments/demograph_eyesoff.py --config-dir pipelines/config --config-name experiments/demograph_eyesoff +run.submit=True`. 
Please refer to [Hydra override syntax](https://hydra.cc/docs/next/advanced/override_grammar/basic/) for more info.
