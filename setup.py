"""Unearthed Submission build."""
from setuptools import find_packages, setup

setup(
    name="unearthed_competition_template",
    py_modules=[
        "preprocess",
        "train",
        "mdl"
        # note predict and score modules are not required to be submitted
        # add any additional modules you want included in your submission here
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version="1.0.0",
    description="Challenge Template",
    author="Unearthed Solutions",
    author_email="info@unearthed.solutions",
)
