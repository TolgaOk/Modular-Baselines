from setuptools import setup, find_packages

setup(
    # Metadata
    name="modular_baselines",
    version="0.2.0",
    author="Tolga Ok",
    author_email="okt@itu.edu.tr",
    url="",
    description="A modular RL library build on Stable-baselines3",
    long_description=("Framework agnostic modular components for RL research with Pytorch and"
                      " JAX policy implementations. Some of the common utilities are imported"
                      " from SB3. Likewise, the structure is quite similar that of SB3 but with"
                      "a modular approach."),
    license="MIT",

    # Package info
    packages=["modular_baselines"],
    install_requires=[
    ],
    zip_safe=False
)
