# Submission-Time Image Override

| Owner | Approvers | Participants |
| - | - | - |
| [Fuhui Fang](mailto:fufang@microsoft.com) | [Daniel Miller](mailto:danmill@microsoft.com) | AML DS Team |

To allow components to operate across multiple workspaces (eyes-off vs. eyes-on), the image and base url of Python Package Index need to be adjusted accordingly. For example, given a signed component in Heron PROD environment, if the user wants to run it in MyData, then he/she need to change the base image from `polymerprod` to `mcr` or `polymerdev`. Currently, this change must be manually done by the user, which is not an optimal Data Science experience.

This RFC (**R**equest **F**or **C**omment) proposes the code patterns and expected behavior of automatic image override at submission time. It begins by giving a concrete proposal with sample code, followed by concerns and known risks of the proposal. A summary of the proposed workflow is also presented at the end.

The following topics are out of scope for this RFC:
- Switching between remote copies of a component: this should be handled through `shrike.build`. Here we only discuss how to switch from a remote copy to a local copy


## Proposal
The `shrike.pipeline` library should **automatically detect the tenant id** and decide if image override is required. If required, it will **modify `component_spec.yaml` for all components listed in `use_local` using the mapping defined, and revert the changes after submission**. Note that in addition to base image, tags, and other fields should be adjusted accordingly.

From a user perspective, the following change is required: in `pipeline.yaml`, adding a section for `tenant_overrides`:
```yaml
defaults:
- aml: eyesoff
- compute: eyesoff
- modules: module_defaults

run:
  experiment_name: test
  tags:
    accelerator_repo: test

module_loader:
  local_steps_folder: ../../../components

## Adding the following section for this new feature
tenant_overrides:
    # MSIT tenant
    72f988bf-86f1-41af-91ab-2d7cd011db47:
      environment.docker.image:
        polymerprod.azurecr.io/polymercd/prod_official/azureml_base_gpu_openmpi312cuda101cudnn7: mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
        polymerprod.azurecr.io/polymercd/prod_official/fake_image: mcr.microsoft.com/azureml/fake_image
      tags:
        eyesoff: eyeson
    
    fake_tenant_id_using_dev:
      environment.docker.image:
        polymerprod.azurecr.io/polymercd/prod_official/azureml_base_gpu_openmpi312cuda101cudnn7: polymerdev.azurecr.io/polymercd/dev_official/azureml_base_gpu_openmpi312cuda101cudnn7
      tags:
        eyesoff: dev
    
    personal: # using file name to specify tenant instead of id
      environment.docker.image:
        'polymerprod.azurecr.io/polymercd/prod_official/(.+)': 'polymerdev.azurecr.io/polymercd/dev_official/$1'

```
, with an example component `spec.yaml` defined as 
```yaml
# yaml-language-server: $schema=https://componentsdk.blob.core.windows.net/jsonschema/DistributedComponent.json

$schema: http://azureml/sdk-1-5/DistributedComponent.json

name: canary-gpu
version: 0.0.2

tags: eyesoff

type: DistributedComponent
is_deterministic: False

description: test
inputs:
  args:
    type: String
    optional: true

environment:
  docker:
    image: polymerprod.azurecr.io/polymercd/prod_official/azureml_base_gpu_openmpi312cuda101cudnn7
  conda:
    conda_dependencies:
      dependencies:
      - python=3.7
      - pip:
        - azureml-core==1.27.0
        - --index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/

launcher:
  type: mpi
  additional_arguments: >-
    python run.py
    [ {inputs.args} ]
```
When a user submits the experiment to an eyes-on workspace under MSIT using command:
```powershell
python pipelines/experiments/test.py --config-dir pipelines/config --config-name experiments/test aml=eyeson compute=eyeson
```
The `spec.yaml` would be modified for this submission as follows:
```yaml
$schema: http://azureml/sdk-1-5/DistributedComponent.json

name: canary-gpu
version: 0.0.2

tags: eyeson    # Note change of tags

type: DistributedComponent
is_deterministic: False

description: test
inputs:
  args:
    type: String
    optional: true

environment:
  docker:
    image: mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04   # Note change of image
  conda:
    conda_dependencies:
      dependencies:
      - python=3.7
      - pip:
        - azureml-core==1.27.0
        - --index-url https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/  # This will be kept, as MyData/eyes-on workspaces have a connection to this feed.
launcher:
  type: mpi
  additional_arguments: >-
    python run.py
    [ {inputs.args} ]
```
Similarly, if the detected tenant is `fake_tenant_id_using_dev`, the base image will be pulled from `polymerdev.azurecr.io` and tag will be updated as `dev`, as defined.

The following fields are changed after switching the tenant:
- `environment.docker.image`
- `tags`

The added code should be interpreted as a JSONPath expression, and both "naive" string substitution and regex-based substitution are allowed, e.g.:
```yaml
'polymerprod.azurecr.io/polymercd/prod_official/(.+)': 'polymerdev.azurecr.io/polymercd/dev_official/$1'
```

**Question:** what else should we allow override?

## Concerns and Risks
- This would edit the users' YAML files. Potential for concern with registered components.
  - Solution 1: spec modification + revert in try/except block
  - Solution 2: create a copy of spec and the delete
- We would then have to support editing all other fields (tags, etc.)
- <s>MCR and Polymer have different image names, both of which may change over time. There's no clear 1:1 mapping.</s> 
  - Fortunately, `polymerdev` is in MSIT and you can already get that attached to MyData. Also, clear 1:1 mapping between that and *prod.
- (add-on feature) In addition to explicitly specify `tenant_id`, user might also want to use a more readable keyword, e.g. the file name for workspace configs `eyes-off`, `personal`, etc.. There might be some issues, and we will investigate as we implement the feature. 

## Summary
The proposed workflow can be summarized as:

0. Signed components designed for eyes-off
1. User adds `tenant_overrides` to their pipeline yaml file, using "naive" string substitution and/or regex-based substitution
2. User runs the command to submit the experiment to an eyes-on workspace 
3. The library detects the change-of-tenant, and modifies all `component_spec.yaml` to address image, PyPI, tags, etc. change
4. `component_spec.yaml` reverted after experiment submitted