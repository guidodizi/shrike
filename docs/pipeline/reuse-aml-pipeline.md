# General process for reusing and configuring an existing AML experiment

## Motivations

Essentially, an experiment is made of:
- a runnable python script,
- a configuration file.

They are checked in a repository, making it reusable and shareable in your team. In many cases, it will be easy to copy an existing experiment to make it your own by just modifying the configuration file. In some instances, you'll want to modify the python script as well to have a more in-depth change on the structure of the graph.

## 1. Branch from an existing experiment

In this section, we'll assume you have identified an existing runnable script (_e.g._: `demograph_eyesoff.py` [here](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fexperiments%2Fdemograph_eyesoff.py) in the accelerator repo) and a configuration file (_e.g._ `demograph_eyesoff.yaml` [there](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fexperiments%2Fdemograph_eyesoff.yaml) in the accelerator repo)
1. Create a new branch in your team's repository (to avoid conflicts).

2. In the `config/experiments/` [folder](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fexperiments), identify a config file you want to start from.

3. Copy this file into a new configuration file of your own.

4. If this configuration file has a module manifest specified in the defaults section, as an example:

    ```yaml
    defaults:
      - aml: eyesoff # default aml references
      - compute: eyesoff # default compute target names
      - modules: module_defaults # list of modules + versions, see config/modules/
    ```

    Copy the file under `config/experiments/demograph_eyesoff.yaml` to a file of your own (ex: `demograph_eyesoff_modelX.yaml`), and rename the name under `defaults` in your experiment config accordingly.

    ```yaml
    defaults:
      - aml: eyesoff # default aml references
      - compute: eyesoff # default compute target names
      - modules: modules_modelX # <<< modify here
    ```

    In the following sections, you will be able to pin each component in this manifest to a particular version number, different from your colleagues version numbers.

4. In the `run` section, modify your experiment name:

    ```yaml
    run: # params for running pipeline
      experiment_name: "modelX" # <<< modify here
    ```

    > Note: we recommend to use a different experiment for each project you work on. Experiments are essentially just a folder to put all your runs in order.

5. That's it, at this point we invite you to try running the script for validating the graph (see below as an example):

    ```powershell
    python pipelines/experiments/demograph_eyesoff.py --config-dir pipelines/config --config-name experiments/demograph_eyesoff_modelX
    ```

    This will try to build the graph, but will not submit it as an experiment. You're ready to modify the experiment now.

# 2. Modifying the code (best practices)

Here's a couple recommendations depending on what you want to do.

## Your experiment consists in modifying components only

If your experiment consists in modifying one or several components only (not the structure of the graph).

- work in a branch containing your component code
- use local code to experiment
- create a specific configuration file for your experiment that shows which component versions to use
- when satisfied, merge in to main/master and register the new versions for your team to share.

## Your experiment consists in modifying the graph structure

- work in a branch containing your graph
- identify conflicts of versions with your team and either create a new graph or modify the existing one with options to switch on/off your changes
- create a specific configuration file for your experiment that shows which component versions to use
- when satisfied, merge in to main/master so that the team can reuse the new graph

# 3. Experiment with local code

If you want to modify a particular component for your experiment, we recommend to iterate on the component code using detonation chamber.

**IMPORTANT**: this is NOT possible for HDI components. If you want to modify HDI components, we recommend to test those first in eyes-on, or to register new versions of those HDI components in parallel of your experiment branch (create a specific branch for your new component versions).

To use the local code for a given component:

## Identify the component key

Identify the component **key**, this is the key used to map to the component in the graphs/subgraphs code.

1. Go to the graph or subgraph you want to modify, check in the `build()` function to identify the component load key used. For instance below we want to modify `VocabGenerator`:

    ```python
    def build(self, config):
        # ...
        vocab_generator = self.module_load("VocabGenerator")
        # ...
    ```

2. If your pipeline uses the `required_modules()` method, this key will match with an entry in the required modules dictionary:

    ```python
    @classmethod
    def required_modules(cls):
        return {
            "VocabGenerator":{
                # references of remote module
                "remote_module_name":"SparkVocabGenerator", "namespace":"microsoft.com/office/smartcompose", "version":None,
                # references of local module
                "yaml_spec":"spark_vocab_generator/module_spec.yaml"
            },
    ```

    The key here is `VocabGenerator`, which is not the component name, but its key in the required modules dictionary.

3. If your pipeline uses a module manifest in YAML (recommended!), this key will map to an entry in the modules manifest file `config/modules/modules_modelX.yaml`:

    ```yaml
    manifest:
      # ...
      - key: "VocabGenerator"
        name: "microsoft.com/office/smartcompose://SparkVocabGenerator"
        version: null
        yaml: "spark_vocab_generator/module_spec.yaml"
      # ...
    ```

    The key here is `VocabGenerator`, which is not the component name, but its key in the required modules dictionary.

    > Note: if no `key` is specified, the `name` is used as key.

## Use component key to run this component locally

1. Use the `use_local` command with that key. You can either add it to the command line:

    ```powershell
    python pipelines/experiments/demograph_eyesoff.py --config-dir pipelines/config --config-name experiments/demograph_eyesoff_modelX module_loader.use_local="VocabGenerator"
    ```

    Or you can write it in your configuration file:
    ```yaml
    # module_loader 
    module_loader: # module loading params
      # IMPORTANT: if you want to modify a given module, add its key here
      # see the code for identifying the module key
      # use comma separation in this string to use multiple local modules
      use_local: "VocabGenerator"
    ```

    > Note: this is example for illustrative purposesonly; the accelerator repo and the demo eyes-off graph do not really have a `VocabGenerator` component.

    From `shrike v1.4.0`, you could also specify "all local except for a few", using the syntax:
    ```yaml
    # module_loader
    module_loader:
      use_local="!KEY1, !KEY2"
    ```
    or from command line override:
    ```powershell
    python <your_pipeline_and_config> module_loader.use_local="'!KEY1, !KEY2'"
    ```
    where only components `KEY1` and `KEY2` are remote, and all the other components are local.

    In summary, four patterns are allowed 
    - use_local = "", all components are remote
    - use_local = "*", all components are local
    - use_local = "KEY1, KEY2", components `KEY1` and `KEY2` are local
    - use_local = "!KEY1, !KEY2", all components except for `KEY1` and `KEY2` are local

    :warning:  **"Mix" (e.g., use_local="KEY1, !KEY2") is not allowed.**
2. When running the experiment, watch-out in the logs for a line that will indicate this component has loaded from local code:

    ```
    Building subgraph [Encoding as EncodingHdiPipeline]...
    --- Building module from local code at spark_vocab_generator/module_spec.yaml
    ```

# 4. Experiment with different versions of (registered) components

The way the helper code decides which version to use for a given component M is (in order):

- if `module_loader.force_all_module_version` is set, use it as version for component M (and all others)
- if a version is set under component M in `modules.manifest`, use it
- if a version is hardcoded in `required_modules()` for component M, use it
- if `module_loader.force_default_module_version` is set, use it as version for component M (and all others non specified versions)
- else, use default version registered in Azure ML (usually, the latest).

Version management for your experiment components can have multiple use cases.

## Use a specific version for all unspecified (`force_default_module_version`)

If all your components versions are synchronized in the registration build, you can use this to use a single version number accross all the graph for all components that have unspecified versions (`version:null`). If you want to pin down a specific version number outside of this, you can add a specific version in your module manifest, or in the `required_modules()` method.

## Use a specific version for all components

If all your components versions are synchronized in the registration build, you can use this to use a single version number accross all the graph for all components. This will give you an exact replication of the components at a particular point in time. This will override all other version settings.

## Use specific versions for some components

If you want to pin down a specific version number for some particular components, specify this version in the module manifest:

```yaml
# @package _group_
manifest:
  # ...
  - key: "LMPytorchTrainer"
    name: "[SC] [AML] PyTorch LM Trainer"
    version: null # <<< HERE
    yaml: "pytorch_trainer/module_spec.yaml"
  # ...
```

or hardcode it (not recommended) in your `required_modules()` method:

```python
   @classmethod
    def required_modules(cls):
        return {
            "LMPytorchTrainer":{
                "remote_module_name":"[SC] [AML] PyTorch LM Trainer",
                "namespace":"microsoft.com/office/smartcompose",
                "version":None, # <<< HERE
                "yaml_spec":"pytorch_trainer/module_spec.yaml"
            }
        }
```
