# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""
Setup script for this Python package.
https://docs.python.org/3/distutils/setupscript.html
"""


import pathlib
from setuptools import setup, find_packages
from shrike import __version__


def versions_in_requirements(file):
    lines = file.read().splitlines()
    versions = [
        line
        for line in lines
        # https://stackoverflow.com/a/2405300
        if not line.isspace() and "--" not in line
    ]
    return list(versions)


HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

with open(HERE / "requirements/requirements-logging.txt") as f:
    required_logging = versions_in_requirements(f)

with open(HERE / "requirements/requirements-pipeline.txt") as f:
    required_pipeline = versions_in_requirements(f)

with open(HERE / "requirements/requirements-build.txt") as f:
    required_build = versions_in_requirements(f)

with open(HERE / "requirements/requirements-dev.txt") as f:
    required_dev = versions_in_requirements(f)

setup(
    name="shrike",
    version=__version__,
    description="Python utilities for compliant Azure machine learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/azure/shrike",
    author="AML Data Science",
    author_email="aml-ds@microsoft.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(include=["shrike*"]),
    include_package_data=True,
    install_requires=required_logging,
    extras_require={
        "pipeline": required_pipeline,
        "build": required_build,
        "dev": required_dev,
    },
    # https://stackoverflow.com/a/48777286
    python_requires="~=3.6",
)
