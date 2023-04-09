# Modular-Baselines

> Under Development (In the current version there is no JAX algorithms and network)

Modular-Baselines is a Reinforcement Learning (RL) library with the objective of providing composable, easy-to-use, general set of components for RL research. These components are framework agnostic in the sense that they do not rely on a specific framework (PyTorch, Tensorflow, Jax). That said, Modular baselines includes both Pytorch and JAX implementations of some of the algorithms. The algorithms are implemented within a single script for readability using the components provided in Modular-Baselines. We extensively use duck-typing instead of inheritance to loose the dependency between components and algorithms. 

| Algorithm |  <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 64px alt="logo"></img> | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width = 50px  height = 50px alt="logo"></img> |
|:-----:|:---------:|:---------:|
|  A2C  |:x:|:heavy_check_mark:|
|  PPO  |:x:|:heavy_check_mark:|

### **Components**
The algorithms in Modular-Baselines are built using the following components:


| Collector |  <img src="https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/logomark/numpylogoicon.svg" width = 64px alt="logo"></img>  |
|:-----:|:---------:|
|  Rollout  |:heavy_check_mark:|
|  Episode  |:x:|
|  Model  |:x:|
|  MCTS  |:x:|

| Buffers |  <img src="https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/logomark/numpylogoicon.svg" width = 64px alt="logo"></img>  |
|:-----:|:---------:|
|  Uniform  |:heavy_check_mark:|
|  Prioritized  |:x:|

| Traces |  <img src="https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/logomark/numpylogoicon.svg" width = 64px alt="logo"></img>  | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width = 50px  height = 50px alt="logo"></img> |
|:-----:|:---------:|:---------:|
|  GAE  |:heavy_check_mark:|:x:|
|  Retrace  |:x:|:x:|
|  Vtrace  |:x:|:x:|



- - -
## Logging

Modular baselines implements a flexible logging mechanism. This includes a set of data loggers, nested log grouping, writers, and MB logger that combines everything. 

<img src="./docs/loggers.svg">

The above logger mechanism explicitly shows the relation between log items and the writers. For example; the ```eps_reward``` log item is a ```QueueDataLog``` which contains the last ```n``` values and its ```mean``` is propagated to the writers in the ```progress``` log group. Hence, readers of the code can easily observe how the reported log records are obtained without diving deep into the source code.

- - -
## Installation

Modular Baselines requires pytorch and gym environments.

We recommend that you install ```pytorch``` and ```mujoco``` separately before installing the requirements. 

- [Mujoco](https://github.com/openai/mujoco-py)
- [Pytorch](https://pytorch.org/get-started/locally/) 

Install the requirement packages by running

```
conda install -c conda-forge swig
conda install nodejs
pip install -r requirements.txt
```

Install the project in development mode

```
pip install -e .
```

- - -
## Vega Visualizations

Modular-Baselines provides vega-lite Json templates for rendering interactive plots for visualizing the logger outputs. The log files of a run can be visualized in Jupyter Notebook via the provided functions in ```visualizers``` folder.

<p float="left">
  <img src="docs/single-seed.svg" width="45%" />
  <img src="docs/multi-seed.svg" width="45%" />
</p>


## Maintainers

Modular Baselines has been developed and maintained by [Tolga Ok](https://tolgaok.github.io./) & [Abdullah Vanlıoğlu](https://github.com/AbdullahVanlioglu) & [Mehmetcan Kaymaz](https://github.com/MehmetcanKaymaz).

## Acknowledgments

 This repository is developed at Data & Decision Lab at [ITÜ Artificial Intelligence and Data Science](https://ai.itu.edu.tr).