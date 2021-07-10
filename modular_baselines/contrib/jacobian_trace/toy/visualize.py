from ipycanvas import Canvas, hold_canvas
import ipywidgets
from ipywidgets.widgets import widget
import matplotlib.pyplot as plt
import matplotlib
import torch
import gym
import numpy as np


class EncodingVisaulizer():

    def __init__(self,
                 n_vector: int,
                 vector_size: int,
                 cell_size: int = 10,
                 border_ratio: float = 0.05,
                 cmap: str = "viridis") -> None:
        self.cmap = plt.get_cmap(cmap)
        self.n_vector = n_vector
        self.vector_size = vector_size
        self.canvas = Canvas(height=n_vector * cell_size, width=vector_size * cell_size)
        self.border_ratio = border_ratio
        self.cell_size = cell_size

    def __call__(self, encoding: torch.Tensor):
        assert encoding.shape[0] == self.n_vector
        assert encoding.shape[1] == self.vector_size
        self.draw(encoding, 0, 0, )

    def draw(self, matrix, y_offset, x_offset):
        border = self.cell_size * self.border_ratio
        canvas = self.canvas
        with hold_canvas(canvas):
            height, width = matrix.shape
            for iy in range(height):
                for ix in range(width):
                    color = self.cmap(matrix[iy, ix])[:3]
                    canvas.fill_style = matplotlib.colors.rgb2hex(color)
                    canvas.fill_rect(x_offset + ix * self.cell_size + border,
                                     y_offset + iy * self.cell_size + border,
                                     self.cell_size - border * 2,
                                     self.cell_size - border * 2)


class GradientVisualizer(EncodingVisaulizer):

    def __init__(self,
                 horizon: int,
                 action_space: gym.spaces,
                 range_slider: ipywidgets.FloatRangeSlider,
                 accumulate_toggle: ipywidgets.ToggleButton,
                 cell_size: int = 10,
                 border_ratio: float = 0.05,
                 cmap: str = "OrRd") -> None:
        self.horizon = horizon
        self.accumulate_toggle = accumulate_toggle
        self.range_slider = range_slider
        if isinstance(action_space, gym.spaces.Discrete):
            height = action_space.n
        elif isinstance(action_space, gym.spaces. MultiDiscrete):
            height = sum(action_space.nvec)
        else:
            raise ValueError("Actoin spaces is unsupported")
        self.prev_grads = None
        super().__init__(height, self.horizon, cell_size, border_ratio, cmap)

    def __call__(self, gradients: torch.Tensor):
        min_val, max_val = self.range_slider.value
        is_acc = self.accumulate_toggle.value

        gradients = gradients[-self.horizon:, :].transpose(1, 0)
        _gradients = gradients.clone()
        if is_acc is False and self.prev_grads is not None:
            _gradients[:, :gradients.shape[1] - 1] -= self.prev_grads[
                :, 0 if self.prev_grads.shape[1] != self.horizon else 1:]
        matrix = torch.zeros((self.n_vector, self.vector_size))
        normalized_grads = _gradients.abs().clamp(min_val, max_val) / (max_val - min_val)
        matrix[:, :min(gradients.shape[1], self.horizon)] = normalized_grads
        self.draw(matrix.numpy(), 0, 0)
        self.prev_grads = gradients
