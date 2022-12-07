"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "wandb",
]

# Installation operation
setup(
    name="rover_learning_by_cheating",
    author="Anton Bjørndahl Mortensen",
    version="1.0.0",
    description="Environment for applying learning by cheating to a trained policy",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    zip_safe=False,
)

# EOF
