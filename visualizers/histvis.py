import os
import json
import plotly
import numpy as np
from functools import lru_cache
from itertools import chain


import plotly.graph_objects as go
from IPython import get_ipython
from IPython.display import display
from IPython.core.magic import (register_line_magic,
                                register_cell_magic)
from ipywidgets import (Button,
                        Layout,
                        IntSlider,
                        HBox,
                        VBox,
                        Box,
                        Label,
                        Layout,
                        Image,
                        Output,
                        Dropdown,
                        Play,
                        jslink)


class HistRender():

    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            raise FileNotFoundError("Logging directory does not exist")

        self.hist_dir = os.path.join(log_dir, "hist")
        if not os.path.exists(self.hist_dir):
            raise FileNotFoundError("No hist folder in the logging dir")

        self.hist_file_names = os.listdir(self.hist_dir)
        self.hist_offset = 1
        self.color_scale_name = "OrRd"
        self.color_scale = getattr(plotly.colors.sequential,
                                   self.color_scale_name)
        self.set_components()

    def set_components(self):
        self.file_select = Dropdown(
            options=self.hist_file_names,
            value=None,
            description="file",
            disabled=False,
            # layout=Layout(width="400px")
        )

        self.file_select.observe(self.file_select_callback)

        self.layer_select = Dropdown(
            options=[],
            value=None,
            description="parameter",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.layer_select.observe(self.layer_select_callback)
        self.colorscale_select = Dropdown(
            options=[x for x in dir(
                plotly.colors.sequential) if x[0].isupper()],
            value=self.color_scale_name,
            description="colorscale",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.colorscale_select.observe(self.set_colorscale_callback)
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",)

    def file_select_callback(self, change):
        if (change["new"] == change["old"]) or change["name"] != "index":
            return None
        infos = self._read_file(int(change["new"]))

        if len(infos) == 0:
            return None

        self.data = infos
        self.layer_select.options = sorted(
            set(chain(*(list(info.keys()) for info in self.data))))

    def layer_select_callback(self, change):
        if change["name"] != "value":
            return None
        self.layer_name = change["new"]
        self.render_figure()

    def set_colorscale_callback(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        colorscale_name = change["new"]
        self.color_scale = getattr(plotly.colors.sequential, colorscale_name)

    def render_figure(self):
        self.fig.data = []
        fig_data = []

        for ix, layer_data in enumerate(reversed(self.data)):
            ix = len(self.data) - ix - 1
            for name, data_dict in layer_data.items():
                if name != self.layer_name:
                    continue
                freqs = np.zeros((len(data_dict["freqs"]) + 2,))
                freqs[1:-1] = data_dict["freqs"]
                bins = np.zeros((len(data_dict["bins"]) + 2,))
                bins[1:-1] = data_dict["bins"]
                bins[0] = bins[1]
                bins[-1] = bins[-2]

                color = self.get_color(ix, len(self.data), self.color_scale)

                fig_data.append(go.Scatter(
                    x=bins,
                    y=freqs + (ix * self.hist_offset),
                    mode="lines",
                    showlegend=False,
                    fill="toself",
                    fillcolor=color,
                    line=dict(
                        color="white",
                        width=0.4,
                        shape="spline",
                        smoothing=0.7)))

        self.fig.add_traces(fig_data)
        self.fig.update_layout(
            title=self.layer_name,
            yaxis={
                "title": "Density",
                # "range": [-1, self.hist_offset * len(fig_data) * 1.5],
            },
            xaxis={
                "title": "Values"
            }
        )

    @lru_cache(maxsize=10)
    def _read_file(self, index):
        file_name = self.hist_file_names[index]
        path = os.path.join(self.hist_dir, file_name)
        infos = []
        with open(path, "r") as fobj:
            for jsonstr in fobj.readlines():
                infos.append(json.loads(jsonstr))
        return infos

    def __call__(self):
        out_display = VBox([
            self.file_select,
            self.layer_select,
            self.colorscale_select,
            self.fig,
        ],
            layout=Layout(
                width="46%",
                height="460px",
                display="flex",
                flex_flow="column",
                justify_content="space-around",
                border="solid 2px gray"
        ))
        return out_display

    @staticmethod
    def get_color(index, length, colormap):
        colormap, _ = plotly.colors.convert_colors_to_same_type(colormap)
        delta = length / (len(colormap) - 1)
        lower_index = int(index/delta)
        upper_index = lower_index + 1
        return plotly.colors.find_intermediate_color(
            lowcolor=colormap[upper_index],
            highcolor=colormap[lower_index],
            intermed=(index % delta) / delta,
            colortype="rgb")
