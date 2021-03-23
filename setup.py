
from setuptools import setup, find_packages

setup(
    # Metadata
    name="modular_baselines",
    version="0.1.1",
    author="Tolga Ok",
    author_email="okt@itu.edu.tr",
    url="",
    description="A modular implementatioan of Stable Baselines3",
    long_description=("Reinforcement Research framework based on Pytorch and"
                      " Stable-Baselines3 that aims to provide building blocks"
                      " for RL research. The framework is build distributed "
                      "computing in mind from the very beginning. Alongside "
                      "with the fundamental components, we aim to include "
                      "ipython based visualization and logging tools."),
    license="MIT",

    # Package info
    packages=["modular_baselines"],
    install_requires=[
    ],
    zip_safe=False
)
