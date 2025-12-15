import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 11):
    raise RuntimeError("Python version 3.11 or higher required")

setup(
    name="distilvit",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train=distilvit.train:main",  # "main" is a function in "train_model.py"
        ],
    },
)
