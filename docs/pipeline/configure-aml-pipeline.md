# How to use pipeline config files when using shrike.pipeline

## Structure of config files

In this page, we perform a detailed review of an [example standard config file](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fexperiments%2Fdemograph_eyeson.yaml&version=GBmain). This should give you a good idea of how to use config files properly based on your scenarios.

For a pipeline, we set up 4 config files under the [`config` directory](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig&version=GBmain&_a=contents), which includes 4 sub-folders: `experiments`, `modules`, `aml`, and `compute`. The `demograph_eyeson.yaml` file linked above lives in the `experiments` folder; it is the main config file which specifies the overall pipeline configuration. This main config file refers to three other config files under the `config` directory:

1. a [config file under the `aml` folder](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Faml%2Feyeson.yaml&version=GBmain)  which lets you point at your Azure ML workspace by specifying subscription_id, resource_group, workspace_name, tenant and auth;
2. a [config file under the `compute` folder](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fcompute%2Feyeson.yaml&version=GBmain)  which specifies configurations such as the compliant data store name, compute targets names, data I/O methods, etc;
3. a [config file under the `modules` folder](https://dev.azure.com/msdata/Vienna/_git/aml-ds?path=%2Frecipes%2Fcompliant-experimentation%2Fpipelines%2Fconfig%2Fmodules%2Fmodule_defaults.yaml&version=GBmain), which lists all the available components with their properties (key, name, default version, and location of the component specification file).

Now we will go through the config file linked above and explain each section.

### 1. Brief summary section

At the beginning of the config file, it is suggested to provide a brief comment explaining which pipeline this config file is used for, and also provide an example command to run the pipeline with this config file. See below for an example:

```yaml
# This YAML file configures the accelerator tutorial pipeline for eyes-on

# command for running the pipeline:
# python pipelines/experiments/demograph_eyeson.py --config-dir pipelines/config --config-name experiments/demograph_eyeson run.submit=True
```

### 2. `defaults` section

The `defaults` section contains references of the aml resources, pointing to two other config files under the `aml` and `compute` folders. It also points to the file listing all available components, which is located in the `modules` folder. This section looks like below.

```yaml
defaults:
  - aml: eyeson # default aml references
  - compute: eyeson # default compute target names
  - modules: module_defaults # list of modules + versions
```

See below for the contents of the `aml` config file. You will need to update the info based on your own aml resources. To find your workspace name, subscription Id, and resource group, go to your Azure ML workspace, then click the "change subscription" icon in the top right (between the settings and question mark), then "Download config file". You will find the 3 values in this file. The Torus TenantId for _eyes-off_ workspaces is `cdc5aeea-15c5-4db6-b079-fcadd2505dc2`, whereas the `72f988bf-86f1-41af-91ab-2d7cd011db47` used below is the Microsoft TenantId that you will use for personal workspaces.

```yaml
# @package _group_
subscription_id: 48bbc269-ce89-4f6f-9a12-c6f91fcb772d
resource_group: aml1p-rg
workspace_name: aml1p-ml-wus2
tenant: 72f988bf-86f1-41af-91ab-2d7cd011db47
auth: "interactive"
```



See below for the contents of the `compute` config file (update the info based on your own aml resources).

```yaml
# @package _group_
# name of default target
default_compute_target: "cpu-cluster"
# where intermediary output is written
compliant_datastore: "workspaceblobstore"

# Linux targets
linux_cpu_dc_target: "cpu-cluster"
linux_cpu_prod_target: "cpu-cluster"
linux_gpu_dc_target: "gpu-cluster"
linux_gpu_prod_target: "gpu-cluster"

# data I/O for linux modules
linux_input_mode: "download"
linux_output_mode: "upload"

# Windows targets
windows_cpu_prod_target: "cpu-cluster"

# data I/O for windows modules
windows_input_mode: "download"
windows_output_mode: "upload"

# hdi cluster
hdi_prod_target: "hdi-cluster"

# data transfer cluster
datatransfer_target: "data-factory"
```

### 3. `run` section

In this section, you configure the parameters controlling how to run your experiment. Update the info based on your own pipeline. Parameter names should be self-explanatory.

```yaml
# run parameters are command line arguments for running your experiment
run: # params for running pipeline
  experiment_name: "demo_graph_eyeson" # IMPORTANT
  regenerate_outputs: false
  continue_on_failure: false
  verbose: false
  submit: false
  resume: false
  canary: false
  silent: false
  wait: false
```

### 4. `module_loader` section

This section includes 4 arguments: `use_local`, `force_default_module_version`, `force_all_module_version`, and `local_steps_folder`.

- The `use_local` parameter specifies which components of the pipeline you would like to build from your local code (rather than consuming the remote registered component). Use a comma-separated string to specify the list of components from your local code. If you use "*", all components are loaded from local code. For more information, please check out [Use component key to run this component locally](./reuse-aml-pipeline.md).
- The `force_default_module_version` argument enables you to change the default version of the component in your branch (the default version is the latest version, but this argument allows you to pin it to a given release version if you prefer).
- The `force_all_module_version` argument enables you to force all components to consume a fixed version, even if the version is specified otherwise in the pipeline code.
- The argument `local_steps_folder` should be clear and self-explanatory: this is the directory where all the component folders are located.

```yaml
# module_loader
module_loader: # module loading params
  # IMPORTANT: if you want to modify a given module, add its key here
  # see the code for identifying the module key
  # use comma separation in this string to use multiple local modules
  use_local: "DemoComponent"

  # fix the version of modules in all subgraphs (if left unspecified)
  # NOTE: use the latest release version to "fix" your branch to a given release
  # see https://eemo.visualstudio.com/TEE/_release?_a=releases&view=mine&definitionId=76
  force_default_module_version: null

  # forces ALL module versions to this unique value (even if specified otherwise in code)
  force_all_module_version: null

  # path to the steps folder, don't modify this one
  # NOTE: we're working on deprecating this one
  local_steps_folder: "../../../components" # NOTE: run scripts from accelerator-repo

```

### 5. Other sections

The sections above only defined _overall_ pipeline parameters, not _component_ parameters. We recommend gathering the component parameters into distinct sections, one per component. The example for the eyes-on demo experiment is shown below.

```yaml
# DemoComponent config
democomponent:
  input_data: irisdata # the data we'll be working on
  input_data_version: "latest" # use this to pin a specific version
  message: "Hello, World!"
  value: 1000 # the size of the sample to analyze
```
