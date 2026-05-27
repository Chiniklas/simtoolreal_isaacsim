"""SimToolReal Isaac Lab package setuptools."""

from setuptools import find_packages, setup

setup(
    name="simtoolreal_lab",
    version="0.1.0",
    description="Isaac Lab transfer of SimToolReal KUKA-SHARPA environments.",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    package_data={
        "simtoolreal_lab.tasks.simtoolreal_sharpa.agents": ["*.yaml"],
        "simtoolreal_lab.tasks.sharpa_nutscrew_pick_place.agents": ["*.yaml"],
    },
    install_requires=[],
)
