import os
import plotly
import numpy as np
from functools import lru_cache
import pandas as pd

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
                        FloatSlider,
                        jslink)


class ScalarRender():

    def __init__(self, log_dir):

        if not os.path.exists(log_dir):
            raise FileNotFoundError("Logging directory does not exist")

        self.progress_path = os.path.join(log_dir, "progress.csv")
        if not os.path.exists(log_dir):
            raise FileNotFoundError(
                "Logging directory does not contain progress.csv")

        self._read_file()
        self.x_name = None
        self.y_name = None
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",)

        self.set_components()

    def _read_file(self):
        self.dataframe = pd.read_csv(self.progress_path)

    def set_components(self):
        names = list(self.dataframe.columns)

        self.select_yaxis = Dropdown(
            options=names,
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)

        monotonic_names = []
        for name in names:
            diff = np.diff(self.dataframe[name].to_numpy())
            diff = diff[~np.isnan(diff)]
            if np.all(diff >= 0):
                monotonic_names.append(name)

        self.select_xaxis = Dropdown(
            options=monotonic_names,
            value=None,
            description="X axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_xaxis.observe(self.set_x_axis)

    def set_x_axis(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.x_name = change["new"]
        self.render_figure()

    def set_y_axis(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.y_name = change["new"]
        self.render_figure()

    def render_figure(self):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []

        self.fig.add_trace(
            go.Scatter(
                x = self.dataframe[self.x_name],
                y = self.dataframe[self.y_name],
                mode="lines",
                line=dict(
                    # color="orange",
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            }
        )

    def __call__(self):
        out_display = VBox([
            self.select_xaxis,
            self.select_yaxis,
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

class MultiScalarRender(ScalarRender):

    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            raise FileNotFoundError("Logging directory does not exist")

        log_dirs = os.listdir(log_dir)
        if len(log_dir) == 0:
            FileNotFoundError("Empty directory")

        self.dataframes = []
        for dir_name in log_dirs:
            log_path = os.path.join(log_dir, dir_name, "progress.csv")
            self.dataframes.append(pd.read_csv(log_path))

        self.x_name = None
        self.y_name = None
        self.quantile_value = 0.25

        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False)
        self.set_components()


    def set_components(self):
        names = tuple(self.dataframes[0].columns)
        for df in self.dataframes:
            if sorted(tuple(df.columns)) != sorted(names):
                raise ValueError("Column names do not match")

        self.select_yaxis = Dropdown(
            options=names,
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)

        intersect_monotonic_names = set(names) 
        for df in self.dataframes:
            monotonic_names = []
            for name in names:
                diff = np.diff(df[name].to_numpy())
                diff = diff[~np.isnan(diff)]
                if np.all(diff >= 0):
                    monotonic_names.append(name)
            monotonic_names = set(monotonic_names)
            intersect_monotonic_names = intersect_monotonic_names & monotonic_names

        self.select_xaxis = Dropdown(
            options=intersect_monotonic_names,
            value=None,
            description="X axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_xaxis.observe(self.set_x_axis)

        self.quantile_slider = FloatSlider(
            value=self.quantile_value,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Quantile",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        )
        self.quantile_slider.observe(self.set_quantile)

    def set_quantile(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.quantile_value = change["new"]
        self.render_figure()

    def render_figure(self):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []

        y_values = np.stack([df[self.y_name].to_numpy() for df in self.dataframes], axis=0)

        median = np.quantile(y_values, 0.5, interpolation="nearest", axis=0)
        upper_quantile = np.quantile(y_values, 0.5 + self.quantile_value, interpolation="nearest", axis=0)
        lower_quantile = np.quantile(y_values, 0.5 - self.quantile_value, interpolation="nearest", axis=0)

        self.fig.add_trace(
            go.Scatter(
                x = self.dataframes[0][self.x_name],
                y = upper_quantile,
                mode="lines",
                line=dict(
                    color="#82c0cc",
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x = self.dataframes[0][self.x_name],
                y = lower_quantile,
                mode="lines",
                fill="tonexty",
                line=dict(
                    color="#82c0cc",
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x = self.dataframes[0][self.x_name],
                y = median,
                mode="lines",
                line=dict(
                    color="#00509d",
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
        )

        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            }
        )

    def __call__(self):
        out_display = VBox([
            self.select_xaxis,
            self.select_yaxis,
            self.quantile_slider,
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