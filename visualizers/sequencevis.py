import os
import json
import plotly
import numpy as np
from functools import lru_cache

import plotly.graph_objects as go
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

class SequenceRender():

    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            raise FileNotFoundError("Logging directory does not exist")

        self.log_dir = log_dir
        self.file_names = os.listdir(self.log_dir)
        self.data = None
        self.time_index = 0
        self.chosen_file_name = None
        self.set_components()

    def set_components(self):
        self.file_select = Dropdown(
            options=self.file_names,
            value=None,
            description="file",
            disabled=False,
            # layout=Layout(width="400px")
        )

        self.file_select.observe(self.file_select_callback)
        self.time_select_slider = IntSlider(
            value=self.time_index,
            min=0,
            max=0,
            step=1,
            description="Iteration",
            # layout=Layout(width="400px"),
        )
        self.time_select_slider.observe(self.set_time_callback)
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",)

    def set_time_callback(self, change):
        if (change["new"] == change["old"] or change["name"] != "value"):
            return None
        self.time_index = int(change["new"])
        if self.data is not None:
            self.render_figure()

    def file_select_callback(self, change):
        if (change["new"] == change["old"]) or change["name"] != "index":
            return None
        infos, file_name = self._read_file(int(change["new"]))
        if len(infos) == 0:
            return None
        self.chosen_file_name = file_name
        self.data = infos
        self.time_select_slider.max = len(self.data) - 1
        self.render_figure()

    def _read_file(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.log_dir, file_name)
        infos = []
        with open(path, "r") as fobj:
            for jsonstr in fobj.readlines():
                infos.append(json.loads(jsonstr))
        return infos, file_name

    def aggregate_statistics(self):
        stacked_values = {step: np.stack(values) for step, values in self.data[self.time_index].items()}
        return {int(step): {"upper_quantile": np.max(values),
                       "lower_quantile": np.min(values),
                       "mean": np.mean(values)} for step, values in stacked_values.items()}

    def render_figure(self):
        self.fig.data = []
        fig_data = []

        color = "#f77f00"
        data = self.aggregate_statistics()
        steps = list(sorted(data.keys()))

        legend_name = f"Time: {self.time_index}"
        fig_data.append(
            go.Scatter(
                x=steps,
                y=[data[step]["upper_quantile"] for step in steps],
                mode="lines",
                legendgroup=legend_name,
                showlegend=False,
                line=dict(
                    color=color,
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        fig_data.append(
            go.Scatter(
                x=steps,
                y=[data[step]["lower_quantile"] for step in steps],
                mode="lines",
                fill="tonexty",
                legendgroup=legend_name,
                showlegend=False,
                line=dict(
                    color=color,
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        fig_data.append(
            go.Scatter(
                x=steps,
                y=[data[step]["mean"] for step in steps],
                mode="lines",
                legendgroup=legend_name,
                name=legend_name,
                showlegend=False,
                line=dict(
                    color=color,
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.add_traces(fig_data)
        self.fig.update_layout(
            title="Sequence Metric",
            yaxis={
                "title": "Value",
            },
            xaxis={
                "title": "Time"
            }
        )

    def __call__(self):
        out_display = VBox([
            self.file_select,
            self.time_select_slider,
            self.fig,
        ],
            layout=Layout(
                height="460px",
                justify_content="space-around",
                border="solid 2px gray",
                margin="10px",
                flex="1 1 46%",
        ))
        return out_display
