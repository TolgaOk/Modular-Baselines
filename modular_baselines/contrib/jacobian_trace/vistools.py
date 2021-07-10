import gym
import numpy as np
import torch
from IPython.display import display
import ipywidgets

class DiscreteLatentPolicyEvaluater():
    
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def evaluate(self):
        pass

    def components(self):
            play = Play(
            min=0,
            max=100,
            step=1,
            interval=200,
            description="Play",
            disabled=False,
        )