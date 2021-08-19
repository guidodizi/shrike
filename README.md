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

The `shrike` library is a set of Python utilities for running experiments in the 
[Azure Machine Learning](https://github.com/Azure/azureml-examples) platform (_a.k.a._ Azure ML). This
library contains four elements, which are:

-  `shrike.compliant_logging`: utilities for compliant logging and 
exception handling;
-  `shrike.pipeline`: helper code for managing, validating and submitting Azure
ML pipelines based on 
[azure-ml-component](https://aka.ms/azure-ml-component-reference) (_a.k.a._ the Component SDK);
-  `shrike.build`: helper code for packaging, building, validating, signing and
registering Azure ML components.
- `shrike.spark`: utilities for running jobs, especially those leveraging Spark
  .NET, in HDInsight.

## Documentation
For the full documentation of `shrike` with detailed examples and API reference, 
please see the [docs page](http://azure.github.io/shrike).

For a list of problems (along with guidance and solutions) designed specifically
to help you learn how to use shrike, please refer to the information in
[this README file](https://github.com/Azure/azure-ml-problem-sets/blob/main/README.md#azure-ml-problems-aimed-at-learning-shrike)
(located in another GitHub repository).

## Installation

The `shrike` library is publicly available in PyPi. There are three optional extra dependencies: `pipeline`, `build`, and `dev`.
The `pipeline` dependency is for submitting Azure ML pipelines, `build` is for signing and registering components, 
and `dev` is for the development environment of `shrike`.

- If you are only planning on using the compliant-logging feature, please `pip install` without any extras:
```pwsh
pip install shrike
```
- If you are planning on signing and registering components, please `pip install` with `[build]`:
```pwsh
pip install shrike[build]
```
- If you are planning on submitting Azure ML pipelines, please `pip install` with `[pipeline]`:
```pwsh
pip install shrike[pipeline]
```
- If you would like to contribute to the source code, please `pip install` with all the dependencies:
```pwsh
pip install shrike[pipeline,build,dev]
```

Alternatively, for local development, you may use the Conda environment defined
in [environment.yml](./environment.yml). It pins the appropriate versions of
pip, Python, and installs all shrike together with all extras as an editable
package.

## Migration from `aml-build-tooling`, `aml-ds-pipeline-contrib`, and `confidential-ml-utils`
If you have been using the `aml-build-tooling`, `aml-ds-pipeline-contrib`, or `confidential-ml-utils` libraries, 
please use the migration script ([migration.py](https://github.com/Azure/shrike/blob/main/migration.py)) to convert your repo or files and
adopt the `shrike` package with one simple command:
```pwsh
python migraton.py --input_path PATH/TO/YOUR/REPO/OR/FILE
```
:warning: This command will update files **in-place**. Please make a copy of your repo/file if you do not want to do so.

## Need Support?
If you have any feature requests, technical questions, or find
any bugs, please do not hesitate to reach out to us.

- For bug reports and feature requests, you are welcome to open an [issue](https://github.com/Azure/shrike/issues). 
- If you are a Microsoft employee, please refer to the 
[support page](https://aka.ms/aml/support) for details;
- If you are outside Microsoft, please send an email
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
