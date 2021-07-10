""" Very simple maze environment
"""
import gym
from gym import spaces
import numpy as np
import random
import pycolab
import time
from itertools import product
from pycolab.prefab_parts import sprites
from pycolab.ascii_art import Partial
from pycolab.cropping import ObservationCropper, ScrollingCropper

from gymcolab.colab_env import ColabEnv

WORLDMAP = ["P     @"]


class PlayerSprite(sprites.MazeWalker):
    """ Sprite of the agent that terminates the environment for a counter
    clockwise turn.
    """

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable="#")
        self.inventory = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # del backdrop, layers, actions, things  # Unused
        the_plot.add_reward(0)
        if actions is not None:
            if self.position.col == 0:
                self.inventory = 0 if actions == 0 else 1
            self._east(board, the_plot)


class CashDrape(pycolab.things.Drape):
    """A `Drape` handling all of the coins.
    This Drape detects when a player traverses a coin, removing the coin and
    crediting the player for the collection. Terminates if all coins are gone.
    """

    def __init__(self, curtain, character):
        super().__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_pattern_position = things['P'].position
        if self.curtain[player_pattern_position]:
            if things["P"].inventory == 1:
                the_plot.add_reward(1)
            else:
                the_plot.add_reward(-1)
            the_plot.terminate_episode()


class InitialDecisionEnv(ColabEnv):
    COLORS = {
        "P": "#00B8FA",
        " ": "#DADADA",
        "@": "#DADA22",
    }

    def __init__(self, cell_size=50, colors=None,
                 render_croppers=None, worldmap=None):
        self.world_map = worldmap or WORLDMAP
        super().__init__(cell_size=cell_size,
                         colors=colors or self.COLORS,
                         render_croppers=render_croppers,
                         n_actions=2)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(self.world_map[0]) * 2,))

    def _init_game(self):
        game = pycolab.ascii_art.ascii_art_to_game(
            art=self.world_map,
            what_lies_beneath=" ",
            sprites={"P": Partial(PlayerSprite)},
            drapes={"@": Partial(CashDrape)},
            update_schedule=[["P"], ["@"]],
            z_order="@P"
        )
        return game

    def observation_wrapper(self, observation):
        max_cols = observation.board.shape[1]
        item = self.game.things["P"].inventory
        pos_x = self.game.things["P"].position.col
        observation = np.zeros((max_cols, 2))
        observation[pos_x, item] = 1
        return observation.flatten()

    def markov_matrix(self):
        n_action = self.action_space.n
        n_state = self.observation_space.shape[0]
        matrix = np.zeros((n_action, n_state, n_state))
        self.reset()
        for acts in product(range(n_action), repeat=self.game.cols - 1):
            state = self.reset()
            y_index = state.argmax().item()
            for act in acts:
                x_index, *_ = self.step(act)
                x_index = x_index.argmax().item()
                matrix[act, y_index, x_index] = 1
                y_index = x_index 
        return matrix