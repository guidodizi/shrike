# Shrike: Compliant Azure ML Utilities

[![CodeQL](https://github.com/Azure/shrike/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Azure/shrike/actions/workflows/codeql-analysis.yml)
[![docs](https://github.com/Azure/shrike/actions/workflows/docs.yml/badge.svg)](https://github.com/Azure/shrike/actions/workflows/docs.yml)
[![python](https://github.com/Azure/shrike/actions/workflows/python.yml/badge.svg)](https://github.com/Azure/shrike/actions/workflows/python.yml)
[![Component Governance](https://dev.azure.com/msdata/Vienna/_apis/build/status/aml-ds/Azure.shrike%20Component%20Governance?branchName=main)](https://dev.azure.com/msdata/Vienna/_build/latest?definitionId=16088&branchName=main)
[![Python versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/Azure/shrike/branch/main/graph/badge.svg?token=sSq0BKlfTu)](https://codecov.io/gh/Azure/shrike)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/shrike)](https://pypi.org/project/shrike/)
[![PyPI version](https://badge.fury.io/py/shrike.svg)](https://badge.fury.io/py/shrike)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Compliant Machine Learning is the practice of training, validating and deploying
machine learning models without seeing the private data. It is needed in many
enterprises to satisfy the strict compliance and privacy guarantees that 
they provide to their customers.

The `shrike` library is a set of Python utilities for compliant machine
learning, with a special emphasis on running experiments in the 
[Azure Machine Learning](https://github.com/Azure/azureml-examples) platform. This
library mainly contains three components, that are

-  `shrike.compliant_logging`: utlities for compliant logging and 
exception handling;
-  `shrike.pipeline`: helper code for managing, validating and submitting Azure
Machine Learning pipelines based on 
[azure-ml-component](https://aka.ms/azure-ml-component-reference);
-  `shrike.build`: helper code for packaging, building, validating, signing and
registering Azure Machine Learning components.

## Documentation
For the full documentation of `shrike` with detailed examples and API reference, 
please see the [docs page](http://azure.github.io/shrike).

## Installation

The `shrike` library is publicly available in PyPi. There are three optional extra dependencies - `pipeline`, `build` and `dev`, 
among which  `pipeline` is for submitting Azure Machine Learning pipelines, `build` is for signing and registering components, 
and `dev` is for the development environment of `shrike`.

- If only the compliant-logging feature would be used, please pip install without any extras:
```pwsh
pip install shrike
```
- If it will be used for signing and registering components, please type with `[build]`:
```pwsh
pip install shrike[build]
```
- If it will be used for submitting Azure Machine Learning pipelines, please type with `[pipeline]`:
```pwsh
pip install shrike[pipeline]
```
- If you would like to contribute to the source code, please install with all the dependencies:
```pwsh
pip install shrike[pipeline,build,dev]
```

## Migration from `aml-build-tooling`, `aml-ds-pipeline-contrib`, and `confidential-ml-utils`
If you have been using the `aml-build-tooling`, `aml-ds-pipeline-contrib`, or `confidential-ml-utils` libraries, 
please use the migration script ([migration.py](https://github.com/Azure/shrike/blob/main/migration.py)) to convert your repo or files and
adopt the `shrike` package with one simple command:
```pwsh
python migraton.py --input_path PATH/TO/YOUR/REPO/OR/FILE
```
:warning: This command will update files **in-place**. Please make a copy of your repo/file if you do not want to do so.

## Need Support?
When you have any feature requests or technical questions or find
any bugs, please don't hesitate to file issues.

- If you are Microsoft employees, please refer to the 
[support page](https://aka.ms/aml/support) for details;
- If you are outside Microsoft, feel free to send an email
to [aml-ds@microsoft.com](mailto:aml-ds@microsoft.com). 


## Contributing

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution.
For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., status check,
comment). Simply follow the instructions provided by the bot. You will only need
to do this once across all repos using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any
additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
