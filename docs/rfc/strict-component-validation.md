# Strict Validation of Azure ML Components

| Owners | Approvers | Participants |
| - | - | - |
| [Haozhe Zhang](mailto:haoz@microsoft.com) | [Daniel Miller](mailto:danmill@microsoft.com) | [AML DS 1P enablers](mailto:aml-ds-1p-enablers@microsoft.com) |


This document presents a proposal on how to enable **strict** validation
of Azure ML components in the signing and registering builds by using
the `shrike` library.

## Motivation
At the `prepare` step of a signing and registering build,
the `shrike.build.command.prepare.validate_all_components()` function
executes an azure cli command 
`az ml component validate --file ${component_spec_path}` to validate
whether the given component spec YAML file has any syntax errors or matches
the strucuture of the pre-defined schema. The validation procedure, which
is critical to satisfactory user experience, could greatly improve the code 
quality of user codebase and eliminate user errors before runtime.

Nevertheless, different teams may have different "rules" for component
validation. Apart from basic syntax checking and schema structure matching,
our customers may have "stricter" validation requirements, to list a few,

- enforcing naming conventions(e.g., office.smartcompose.*)
- requiring email address in the component description
- forcing all input parameters to have non-empty descriptions

To address this feature request, we plan to leverage (1) **regular expression 
matching** and (2) **JSONPath expression** in the `shrike` library.

## Proposal

We will introduce two parameters - `enable_component_validation` 
(type: `boolean`, default: `False`) and `component_validation`
(type: `dict`, default: `None`)
into the [configuration](https://github.com/Azure/shrike/blob/308c2566753470239136c54406969f4762fa2537/shrike/build/core/configuration.py#L17)
dataclass in `shrike.build.core`. The logic for performing
strict component validation will be

- If `config.enable_component_validation is True`:
   - if `len(config.component_validation) == 0`:
      - run the compliance validation only
   - else:
      - run the compliance validation, then 
      run the user-provided customized validation,
      which is written as JSONPath expression and 
      regular expression in `config.component_validation`.
- else
   - Pass

### Compliance validation

Compliance validation refers to checking whether a given 
component spec YAML file meets all the requirements for
running in the compliant AML. Currently, we have three requirements
to enforce, which are

- Check whether the image URL is compliant
- Check whether the pip index-url is 
`https://o365exchange.pkgs.visualstudio.com/_packaging/PolymerPythonPackages/pypi/simple/`
- Check that "default" is only Conda feed

In the future, if we find more requirements to obey, we will
add them into compliance validation.

### Customized validation

First, we ask users to write [JSONPath expressions](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html)
to query Azure ML component spec YAML elements. For example,
the path of component name is `$.name`, while the path of
image is `$.environment.docker.image`.

Second, users are expected to translate their specific "strict" validation rules to 
regular expression patterns. For example, enforcing the component
name to start with "office.smartcompose." could be translated to
a string pattern `^office.smartcompose.[A-Za-z0-9-_.]+$`. 

Then, the JSONPath expressions and corresponding regular expressions
will be combined into a dict and assigned to `config.component_validation`
in the configuration file.

Assuming we enforce three "strict" validation requirements on
the command component: (1)
the component name starts with `office.smartcompose.`, (2)
the image URL is empty or starts with `polymerprod.azurecr.io`, 
and (3) all the input parameter descriptions start with a
capital letter. Below is an example of the configuration file that specifies
the above three validation requirements.

```yaml
activation_method: all
compliant_branch: ^refs/heads/develop$
component_specification_glob: 'steps/**/module_spec.yaml'
log_format: '[%(name)s][%(levelname)s] - %(message)s'
signing_mode: aml
workspaces:
- /subscriptions/2dea9532-70d2-472d-8ebd-6f53149ad551/resourcegroups/MOP.HERON.PROD.9e0f4782-7fd1-463a-8bcf-46afb7eeaca1/providers/Microsoft.MachineLearningServices/workspaces/amlworkspacedxb7igyvbjdhc
allow_duplicate_versions: True
use_build_number: True

# strict component validation
enable_component_validation: True
component_validation:
  '$.name': '^office.smartcompose.[A-Za-z0-9-_.]+$'
  '$.environment.docker.image': '^$|^polymerprod.azurecr.io*$'
  '$.inputs..description': '^[A-Z].*'
```

The `jsonpath-ng` library is a JSONPath implementation for Python.
We will leverage it to parse JSONPath in the `shrike` library, an
example of which is shown below.

```python
>>> import yaml
>>> from jsonpath_ng import parse
>>>
>>> with open('tests/tests_pipeline/sample/steps/multinode_trainer/module_spec.yaml', 'r') as file:
...     spec = yaml.load(file, Loader=yaml.FullLoader)
...
>>> jsonpath_expr = parse('$.name')
>>> jsonpath_expr.find(spec)[0].value
'microsoft.com.amlds.multinodetrainer'
>>>
>>> jsonpath_expr = parse('$.environment.docker.image')
>>> jsonpath_expr.find(spec)[0].value
'polymerprod.azurecr.io/training/pytorch:scpilot-rc2'
>>>
>>> jsonpath_expr = parse('$.inputs..description')
>>> [match.value for match in jsonpath_expr.find(spec)]
['Vocabulary file used to encode data, can be a single file or a directory with a single file', 'File used for training, can be raw text or already encoded using vocab', 'File used for validation, can be raw text or already encoded using vocab']
```