""" Very simple maze environment
"""
import gym
from gym import spaces
import numpy as np
import random
import pycolab
import time
from itertools import chain
from itertools import product
from pycolab.prefab_parts import sprites
from pycolab.ascii_art import Partial
from pycolab.cropping import ObservationCropper, ScrollingCropper

from gymcolab.colab_env import ColabEnv

WORLDMAP = [
    "#########",
    "#   # @ #",
    "# b #   #",
    "#   a   #",
    "#   #   #",
    "# P #   #",
    "#########",
]

class PlayerSprite(sprites.MazeWalker):
    """ Sprite of the agent that terminates the environment for a counter
    clockwise turn.
    """

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable="#")
        self.inventory = None

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # del backdrop, layers, actions, things  # Unused
        if actions is None:
            self.inventory = {
                char: [0] * thing.n_items for char, thing in things.items()
                if isinstance(thing, ItemDrape)
            }
            return
        move_act, _ = actions
        if move_act == 0:    # go upward?
            self._north(board, the_plot)
        elif move_act == 1:  # go downward?
            self._south(board, the_plot)
        elif move_act == 2:  # go leftward?
            self._west(board, the_plot)
        elif move_act == 3:  # go rightward?
            self._east(board, the_plot)


class BoxDrape(pycolab.things.Drape):
    """A `Drape` handling all of the coins.
    This Drape detects when a player traverses a coin, removing the coin and
    crediting the player for the collection. Terminates if all coins are gone.
    """

    def __init__(self, curtain, character):
        super().__init__(curtain, character)
        self.env_length = 500


    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_pattern_position = things["P"].position
        if self.curtain[player_pattern_position]:
            the_plot.add_reward(sum(sum(inv) for inv in things["P"].inventory.values()))
            the_plot.terminate_episode()
        else:
            the_plot.add_reward(0)
        self.env_length -= 1
        if self.env_length == 0:
            the_plot.add_reward(-5)
            the_plot.terminate_episode()


class WallDrape(pycolab.things.Drape):
    def __init__(self, curtain, character):
        super().__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del actions, board, layers, backdrop


class ItemDrape(pycolab.things.Drape):
    """A `Drape` handling all of the coins.
    This Drape detects when a player traverses a coin, removing the coin and
    crediting the player for the collection. Terminates if all coins are gone.
    """

    def __init__(self, curtain, character, key_value, n_keys=2):
        super().__init__(curtain, character)
        assert key_value < n_keys
        self.key_value = key_value
        self.n_keys = n_keys

        self.n_items = np.sum(self.curtain)
        # self.keys = np.random.randint(0, n_keys, size=(self.n_items,))
        self.keys = np.ones((self.n_items,)) * self.key_value
        self.is_used = [False] * self.n_items

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None:
            return
        player_pattern_position = things["P"].position
        _, decision = actions
        if self.curtain[player_pattern_position]:
            item_pos = layers["P"][self.curtain].flatten().argmax()
            if self.is_used[item_pos] is False:
                self.is_used[item_pos] = True
                value = 1 if decision == self.keys[item_pos] else -1
                things["P"].inventory[self.character][item_pos] = value


class DecisionMaze(ColabEnv):
    COLORS = {
        "P": "#00B8FA",
        " ": "#DADADA",
        "@": "#DADA22",
        "a": "#33FBFB",
        "b": "#C94959",
    }

    def __init__(self, cell_size=25, colors=None,
                 render_croppers=None, worldmap=None,
                 possible_keys=2):
        self.world_map = worldmap or WORLDMAP
        self.possible_keys = possible_keys
        super().__init__(cell_size=cell_size,
                         colors=colors or self.COLORS,
                         render_croppers=render_croppers,
                         n_actions=4)
        if self.game is None:
            self._renderer = None
        self.game = self._init_game()
        self.game.its_showtime()

        self.n_passable_states = (~self.game._board.layers["#"]).sum()
        self.n_keys = sum(thing.n_items for thing in self.game.things.values()
                          if isinstance(thing, ItemDrape))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1, self.n_keys + 1, self.n_passable_states))
        self.action_space = gym.spaces.MultiDiscrete([4, possible_keys])

    def _init_game(self):
        game = pycolab.ascii_art.ascii_art_to_game(
            art=self.world_map,
            what_lies_beneath=" ",
            sprites={"P": Partial(PlayerSprite)},
            drapes={"@": Partial(BoxDrape),
                    "a": Partial(ItemDrape, 0, n_keys=self.possible_keys),
                    "b": Partial(ItemDrape, 1, n_keys=self.possible_keys),
                    "#": Partial(WallDrape)},
            update_schedule=[["a"], ["b"], ["P"], ["@"], ["#"]],
            z_order="@ab#P"
        )
        return game

    def step(self, actions):
        assert self.game is not None, ("Game is not initialized"
                                       "Call reset function before step")
        assert self._done is False, ("Step can not be called after the game "
                                     "is terminated. Try calling reset first")

        observation, reward, discount = self.game.play(actions)
        done = self.game.game_over
        self.observation = observation
        return self.observation_wrapper(observation), reward, done, {}

    def observation_wrapper(self, observation):
        pos = observation.layers["P"][~observation.layers["#"]].flatten().argmax()
        items = chain(*self.game.things["P"].inventory.values())

        state = np.stack([self.onehot_state(pos), *(self.onehot_state(item + 1) for item in items)])
        return np.expand_dims(state, 0)

    def onehot_state(self, index):
        return (np.arange(self.n_passable_states) == index).astype(np.float32)

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
