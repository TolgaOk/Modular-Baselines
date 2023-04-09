from setuptools import setup, find_packages

setup(
    # Metadata
    name="modular_baselines",
    version="0.2.0-alpha",
    author="Tolga Ok",
    author_email="okt@itu.edu.tr",
    url="",
    description="A modular RL library",
    long_description=("Framework agnostic modular components for RL research written in Numpy, "
                      "Pytorch, and JAX. Includes, examples, logging mechanism, and vega visualizations"),
    license="MIT",

    # Package info
    packages=["modular_baselines"],
    install_requires=[
    ],
    zip_safe=False
)
