# API design for Federated Learning
To facilitate federated learning, we propose some new APIs for shrike.

## Code structure
The code structure for a federated learning experiment is consistent with the existing one:
```
repo
├── components  # contains subfolder corresponds to an AML component
│   ├──────── component1
│   └──────── component2
└── pipelines
    ├──────── config
    │            ├──────── aml
    │            │           └─── eyesoff.yaml  # workspace info: subscription_id, resource_group, etc.
    │            ├──────── compute
    │            │           └─── eyesoff.yaml  # see example below; info on compute targets, data stores, silos, etc.
    │            ├──────── experiments
    │            │           └─── submit.yaml  # see example below; info on experiment name, hdi_conf, datasets, etc.
    │            └──────── modules  
    │            │           └─── module_defaults.yaml  # contains some yaml files describing which components in `components` or remote copies will be used
    ├──────── experiments
    │            └── submit.py  # see example below; main script for defining and submiting the experiment
    └──────── subgraphs
                 └─────── pipeline_definition.py  # see example below; subgraphs used in the main script submit.py
```

## Federated learning workflow
We consider four fundamental building blocks: preprocess pipeline, midprocess pipeline, postprocess pipeline, training pipeline. The workflow is shown as in the diagram:
```
preprocess (orchestrator) -> training (silos) -> midprocess (orchestrator) -> training (silos) -> midprocess (orchestrator) ... many rounds -> training (silos) -> postprocess (orchestrator)

```
![image](img/fl-diagram.png)

In Azure ML, each "job" can be implemented as a subgraph, or a simple component. On web UI, it is expected to show as a single experiment, with multiple steps and subgraphs, just like the diagram above.

Note that this diagram assumes the strategy will be always synchronous and centralized. See "open questions" for scenarios that may not be fitted into this digram.

## Main submission script: submit.py
We propose two designs. In the first design, Shrike provides a class `FederatedPipelineBase` with built-in methods for the user to inherit and override. The second design requires the user to explictly write the loop, and define the hook connecting upstream and downstream pipelines. While the second design provides more flexibility, the first design serves a similar role as [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) by provoding an easy structured way to do federated learning.

### Design 1: Shrike handles the loop
Shrike will implement a new `FederatedPipelineBase` class which extends `AMLPipelineHelper`. The `FederatedPipelineBase` class provides the structure for federated learning, through four methods `preprocess()`, `midprocess()`, `training()`, and `postprocess()`. Each method returns the output data that will be fed into the next pipeline. Shrike will automatically validate, anonymize, and transfer the outputs from upstream pipelines to downstream pipelines (through calling `apply_federated_runsettings` from the backend). User can override the inherited methods. In this design, we will avoid many user errors that might happen if they write the loop explicitly. However, it could be tricky to define how the pipelines are connected. 
```python
from shrike.pipeline import FederatedPipelineBase
from .pipeline_definition import TrainingPipeline, PreprocessPipeline, MidProcessPipeline, PostProcessPipeline

class MyCoolPipeline(FederatedPipelineBase):
    # TODO: what would this look like?
    # define each pipeline
    # define parameters (groups)


    @classmethod
    def required_subgraphs(cls):
        # implement this new method
        # need a better name
        return {"TrainingPipeline": TrainingPipeline, "PreprocessPipeline": PreprocessPipeline, "MidProcessPipeline": MidProcessPipeline, "PostProcessPipeline": PostProcessPipeline}

    # each function returns the output that will pass on to the next pipeline
    def preprocess(self, config):
        # required function, will run inside orchestrator
        preprocess_pipeline = self.subgraph_load("PreprocessPipeline")  # also allowing loading a single component
        preprocess_pipeline_step = preprocess_pipeline(input = config.federation.preprocess.input, param = config.federation.preprocess.param)
        return preprocess.outputs.output1
    
    def midprocess(self, config):
        # required function, will run inside orchestrator
        midprocess_pipeline = self.subgraph_load("MidProcessPipeline")
        # outputs from all silos will be moved into a shared folder, and midprocess will take this folder as input
        midprocess_pipeline_step = midprocess_pipeline(input = training.outputs.output1)
        return midprocess_pipeline_step.outputs.output1
    
    def postprocess(self, config):
        # required function, will run inside orchestrator
        postprocess_pipeline = self.subgraph_load("PostProcessPipeline")
        postprocess_pipeline_step = postprocess_pipeline(input = training.outputs.output1, param_1 = self.config.federation.postprocess.param_1, param_2 = self.config.federation.postprocess.param_2)
        return postprocess_pipeline_step.outputs.output1

    def training(self, config, input):
        # required function, will run in each silo
        # Shrike handles this: input = preprocess.outputs.output1 if self.first_epoch else midprocess.outputs.output1
        training_pipeline = self.subgraph_load("TrainingPipeline")
        training_pipeline_step = training_pipeline(input = config.federation.training.input1, 
                                                    weights = input, 
                                                    param1 = self.config.federation.training.param1, 
                                                    param2 = self.config.federation.training.param2)
        return training_pipeline_step.outputs.output1
    
# if __name__ == "__main__":
#     MyCoolPipeline()
```
### Design 2: explicitly writing the loop by user
As the main pipeline script, here we need to define the subgraphs and how they are connected. This solution requires the user to explicity write the loop and the data movement. We provide `apply_federated_runsettings(pipeline_instance, silos=*, validation=True, secure_aggregation=True)` method and the `pipeline_instance` will be running in all `silos` configured in the yaml file. With `secure_aggregation=True`, Shrike will anonymize the data and make it ready for transfer.
```python
from shrike.pipeline import AMLPipelineHelper
from .pipeline_definition import TrainingPipeline, PreprocessPipeline, MidProcessPipeline, PostProcessPipeline

class MyCoolPipeline(AMLPipelineHelper):
    # TODO: what would this look like?
    # define each pipeline
    # define parameters (groups)


    @classmethod
    def required_subgraphs(cls):
        # implement this new method
        # need a better name
        return {"TrainingPipeline": TrainingPipeline, "PreprocessPipeline": PreprocessPipeline, "MidProcessPipeline": MidProcessPipeline, "PostProcessPipeline": PostProcessPipeline}

    def build(self, config):

        preprocess_pipeline = self.subgraph_load("PreprocessPipeline")  # also allowing loading a single component
        midprocess_pipeline = self.subgraph_load("MidProcessPipeline")
        postprocess_pipeline = self.subgraph_load("PostProcessPipeline")
        training_pipeline = self.subgraph_load("TrainingPipeline")
        data_transfer = self.component_load("DataTransfer")

        @dsl.pipeline(
            name="federated-learning-pipeline",
            description="",
            default_datastore=config.compute.compliant_datastore,
        )
        def federated_pipeline_function():
            # initialization     
            preprocess_pipeline_step = preprocess_pipeline(input = config.federation.preprocess.input, param = config.federation.preprocess.param)
            # self.apply_smart_runsettings(preprocess_pipeline_step)    # required if preprocess is a component, instead of a subgraph
            iter = 0
            self.apply_federated_runsettings(preprocess_pipeline_step)
            data_transfer_step = data_transfer(input = preprocess_pipeline_step.outputs.output1, fanout=True)

            # iterations
            while iter < config.federation.max_iterations:
                training_pipeline_step = training_pipeline(input = config.federation.training.input1,
                                                           weights = data_transfer_step.outputs.output1, 
                                                           param2 = config.federation.training.param)
                self.apply_federated_runsettings(training_pipeline_step)    # apply_federated_runsettings(pipeline_instance, silos = config.federation.silos)
                data_transfer_step = data_transfer(input = training_pipeline_step.outputs.output1, fanout=False)
                self.apply_federated_runsettings(data_transfer_step)
                midprocess_pipeline_step = midprocess_pipeline(input = data_transfer_step.outputs.output1
                                                               param = config.federation.midprocess.param)
                # self.apply_smart_runsettings(midprocess_pipeline_step)
                data_transfer_step = data_transfer(input = midprocess_pipeline_ste, fanout=True)

            # finalization
            postprocess_pipeline_step = postprocess_pipeline(input = data_transfer_step.outputs.output1, 
                                                             param_1 = config.federation.postprocess.param_1, 
                                                             param_2 = config.federation.postprocess.param_2)
            # self.apply_smart_runsettings(postprocess_pipeline_step)
        return federated_pipeline_function
    
    def pipeline_instance(self, pipeline_function, config):
        my_cool_pipeline = pipeline_function()
        return my_cool_pipeline

if __name__ == "__main__":
    MyCoolPipeline()
```

## `apply_federated_runsettings()` method
We will implement the following method so that the user can assign pipeline to silos in a simple way. Also, with `validation=True` and `secure_aggregation=True`, Shrike will validate and anonymize the data, thus making it ready for transfer. Secure aggregation is not required in the "fan-out" stage, while validation is required before any cross-silo movement. The arg `silos` defines in which silos the `pipeline_instance` will be performing in.
```python
def apply_federated_runsettings(pipeline_instance, silos=*, validation=True, secure_aggregation=True):
    ...
```
This method should be called after `apply_smart_runsettings()` by copying the `pipeline_instance` to multiple silos and ensuring the outputs are safe for transfer, and should be able to override settings already configured in `apply_smart_runsettings()`.

## Compute settings eyesoff.yaml
We add computes and storage info for each silo, also notice the datasets that live in each silo are also specified here, e.g. `input1`.
```yaml
# name of default target
default_compute_target: "cpu-cluster-0-dc"
# where intermediary output is written
compliant_datastore: "heron_sandbox_storage"
noncompliant_datastore: "cosmos14_office_adhoc"

# Linux targets
linux_cpu_dc_target: "cpu-cluster-0-dc"
linux_cpu_prod_target: "cpu-cluster-0"
linux_gpu_dc_target: "gpu-cluster-dc"
linux_gpu_prod_target: "gpu-cluster-0"

# data I/O for linux modules
linux_input_mode: "mount"
linux_output_mode: "mount"

# Windows targets
windows_cpu_prod_target: "cpu-cluster-win"
windows_cpu_dc_target: "cpu-cluster-win"

# data I/O for windows modules
windows_input_mode: "download"
windows_output_mode: "upload"

# hdi cluster
hdi_prod_target: "hdi-cluster"

# data transfer cluster
datatransfer_target: "data-factory"

## new!
silos:
    # override priority: default_config < customized shared config < per-silo setting
    default_config:
        compute: gpu-cluster-1  # by default, all silos will inherit this default_config
        param1: 0
    foo_config: # customized config that multiple silos could share
        compute: gpu-cluster-foo
    bar_config:
        datatransfer_target: "data-factory-bar"
    silo1:
        compute: gpu-cluster-2
        compliant_datastore: heron_sandbox_storage
        datatransfer_target: "data-factory"
        input1: dummy_dataset_on_silo1
    silo2:
        inherit: foo_config, bar_config # merge default_config, foo_config, and bar_config (highest order)
        ...
        input1: dummy_dataset_on_silo2
        param1: 1
    silo3:
        inherit: foo_config
        ...
        input1: dummy_dataset_on_silo3
        param1: 2
```
## Configuration file: submit.yaml
The user needs to add an additional section `federation` with required `max_iterations` and `silos`. Also the user can specify the input datasets and parameters for pipelines running inside the orchestrator and silos here. For input dataset in silos, the syntax is `<pipeline_input_name>: silos.<dataset_name>`, where `<dataset_name>` is specified for each silo in the compute settings `eyesoff.yaml`.
```yaml
defaults:
    ...
run:
    experiment_name: "sample_federated_pipeline"
    ...
# new!
federation:
    max_iterations = 100
    silos = "silo1, silo2, silo3"   # each silo corresponds to a section in compute/eyesoff.yaml
    # could also use: `silos = *`, `silos = "!silo1"`
    preprocess:
        input: dummy_dataset
        param: 1
    midprocess:
        param: "a"
    postprocess:
        param_1: 1
        param_2: True
    training:
        input: silos.input1    # per silo input dataset
        param1: silos.param1    # per silo parameter
        param2: 1    # shared parameters/datasets across all silos
```

## Define a subgraph as usual: pipeline_definition.py
As usual, the user can define each "job" as a subgraph or a single component, and import using `load_component()` or `load_subgraph()` in the pipeline script. Here we
```python
# skeleton code for defining a pipeline
from azure.ml.component import dsl
from shrike.pipeline import AMLPipelineHelper

class TrainingPipeline(AMLPipelineHealer):
    def build(self, config):
        # load components
        my_component = self.component_load("my-component")

        # create an instantce of a pipeline function
        @dsl.pipeline(
            name="A-simple-pipeline",
            description="",
            default_datastore=config.compute.compliant_datastore,
        )
        def my_pipeline_function(InputData, Weights, Param):
            # assume my_component takes three arguments, an input dataset, an epoch that should increments each run, and an parameter that can be dynamically changed in each run
            step = my_component(input = InputData, weights = Weights, param = Param)
            self.apply_smart_runsettings(step)
        
        return my_pipeline_function

class PreprocessPipeline(AMLPipelineHealer):
    ...

class MidprocessPipeline(AMLPipelineHealer):
    ...

class PostprocessPipeline(AMLPipelineHealer):
    ...    
```
## Flexibility
### Per-silo training
We should allow some per-silo variability, which should be handled through `apply_federated_runsettings()`.
For design 1, by default, the pipeline defined in `def training()` will be applied to all silos with an implicit call `apply_federated_runsettings(pipeline_instance, silos=*)`. When this function returns more than one pipeline, then the user could specify the mapping, e.g. `per_silo_training_1` is applied to silo 1 and 2, and `per_silo_training_2` is applied to all other silos,
```python
def training(self, config, input):
    per_silo_training_1 = self.subgraph_load("TrainingPipeline")
    per_silo_training_1_step = per_silo_training_1(input = config.federation.training.input1, 
                                                weights = input, 
                                                param1 = self.config.federation.training.param1, 
                                                param2 = self.config.federation.training.param2)
    self.apply_federated_runsettings(per_silo_training_1_step, silos = "silo1, silo2")

    per_silo_training_2 = self.subgraph_load("TrainingPipeline2")
    per_silo_training_2_step = per_silo_training_2(input = config.federation.training.input1, 
                                                weights = input, 
                                                param1 = self.config.federation.training.param1, 
                                                param2 = self.config.federation.training.param2)
    self.apply_federated_runsettings(per_silo_training_2_step, silos = "*silo1, *silo2")

    return [per_silo_training_1_step.outputs.output1, per_silo_training_2_step.outputs.output1]
```

For design 2, it is more straightforward and the user can just define the per-silo training directly.
```python
while iter < config.federation.max_iterations:
  per_silo_training_1_step = ...
  self.apply_federated_runsettings(per_silo_training_1_step, silos = "silo1, silo2")

  per_silo_training_2_step = ...
  self.apply_federated_runsettings(per_silo_training_2_step, silos = "!silo1, !silo2")
```

### Output anything
There is no limitation that what the four methods shouls output, i.e. we are not limited to the scenario where we are just outputting and aggregating the model weights. A use case is federated distillation where the server asks for predictions over some public dataset, instead of models or gradients.

### Debug mode
The library will also have "debug" mode where it turns off all data transfer and assumes no approval endpoint. This way, data scientists could quickly debug with "dummy" (artificial) silos in eyes-on or reserach context.

## Open questions
1. How are the model weights aggregated after each "training" round?
    - User provides a weight in config file.
    - Shrike can configure the output path to `/output_epoch_<n>/silo<x>`, and the downstream pipeline will read from `/output_epoch_<n>`.
    - Platform ask: `args` (N input datasets) as input to a single component. Worst-case scenario, we can do some kind of "merge folders" hack in advance.
2. What parameters need to be changed dynamically, i.e. not-fixed for each iteration, other than "input data set"?
    - We should cache things like optimizer state (learning rate, etc.), previous models/gradients on disk so that they can be restored in the next round.
3. What other parameters are required in `federation` section in `submit.yaml`?
4. How to handle regional outages? Even though silos are reliable Azure-hosted clusters, they'll occasionally go down. Hopefully, AML pipelines can, as a general feature, allow configurable per-step retry.
5. How to handle early stopping? Once conditional components land, those should be configurable as well (maybe after aggregation, if configured, in each round?).
6. The design also assumes the orchestrator has significant power over the silos- whoever runs code in the orchestrator can effectively inject any code they want to run in each silo. Maybe the design should allow room for the individual silos to control their own internal train() pipeline and this method is more of an RPC into the silo to execute code owned by the silo.
7. How to handle the drop-out or failure of some training processes?
8. Does the federated learning pipeline always look like as the diagram shown above? Does the upstream pipeline just generate one output (e.g. weights) to feed into the downstream pipeline?
    - Scenario : asynchronicity, e.g. client A is taking a very long time to do a round, and in the meantime all the other clients can do 5 rounds. In our current design, the downstream pipeline need to wait until all clients finish their work.
9. How much flexibility do we need to provide to each silo, e.g. `PerSiloPipeline`?
    - See section Flexibility/Per-silo training.
10. [Need discussion] Do we support subgraph + component as a "pipeline", or just subgraph?
11. What is a good naming for the four methods (preprocess/training/midprocess/postprocess)?

## Scenarios
1. Alice creates a FL project with 30 silos, and submits a debugging pipeline to 3 silos.
    - She can specify in `submit.yaml`: `silos = "silo1, silo2, silo3"`
2. Alice wants to extend the pipeline to 27 silos.
    - She can specify in `submit.yaml`: `silos = "!silo1, !silo2, !silo3"`
3. Alice wants a different training pipeline for silo X.
    - See section Flexibility/Per-silo training.
5. The previous experiment fails after 10 epochs, and Alice wants to restart from there.
