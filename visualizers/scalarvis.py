import os
import plotly
import numpy as np
from functools import lru_cache
import pandas as pd
import re

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
                        jslink,
                        FloatSlider)
import ipywidgets as widgets


class ScalarRender():

    default_colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf"   # blue-teal
    ]

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
                x=self.dataframe[self.x_name],
                y=self.dataframe[self.y_name],
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

        self.dataframes = self.get_dataframes(log_dirs, prefix_dir=log_dir)

        self.setup()
        self.set_figure()
        self.set_components()
        self.fill_dropdowns(self.dataframes)

    def setup(self):
        self.x_name = None
        self.y_name = None
        self.quantile_value = 0.25

    def set_figure(self):
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False)

    def get_dataframes(self, log_dirs, prefix_dir):
        dataframes = []
        for dir_name in log_dirs:
            dir_path = os.path.join(prefix_dir, dir_name)
            if os.path.isdir(dir_path):
                log_path = os.path.join(dir_path, "progress.csv")
                if not os.path.exists(log_path):
                    continue
                try:
                    dataframes.append(pd.read_csv(log_path))
                except pd.errors.EmptyDataError:
                    pass
        return dataframes

    @staticmethod
    def _get_column_names(dataframes):
        names = tuple(dataframes[0].columns)
        for df in dataframes:
            if sorted(tuple(df.columns)) != sorted(names):
                raise ValueError("Column names do not match")

        intersect_monotonic_names = set(names)
        for df in dataframes:
            monotonic_names = []
            for name in names:
                diff = np.diff(df[name].to_numpy())
                diff = diff[~np.isnan(diff)]
                if np.all(diff >= 0):
                    monotonic_names.append(name)
            monotonic_names = set(monotonic_names)
            intersect_monotonic_names = intersect_monotonic_names & monotonic_names

        return set(names), intersect_monotonic_names

    def fill_dropdowns(self, dataframe):
        names, intersect_monotonic_names = self._get_column_names(dataframe)
        self.select_yaxis.options = names
        self.select_xaxis.options = intersect_monotonic_names

    def set_components(self):
        self.select_yaxis = Dropdown(
            options=[],
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)

        self.select_xaxis = Dropdown(
            options=[],
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

    def render_figure(self, color="#82c0cc"):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []
        self._add_traces(self.dataframes, color, "")

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

    def _add_traces(self, dataframe, color, legend_name):
        y_values = np.stack([df[self.y_name].to_numpy()
                             for df in dataframe], axis=0)

        median = np.quantile(y_values, 0.5, interpolation="nearest", axis=0)
        upper_quantile = np.quantile(
            y_values, 0.5 + self.quantile_value, interpolation="nearest", axis=0)
        lower_quantile = np.quantile(
            y_values, 0.5 - self.quantile_value, interpolation="nearest", axis=0)

        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=upper_quantile,
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
        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=lower_quantile,
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
        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=median,
                mode="lines",
                legendgroup=legend_name,
                name=legend_name,
                line=dict(
                    color=color,
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
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


class ComparisonScalarRender(MultiScalarRender):

    def __init__(self, logs_dir):
        if not os.path.exists(logs_dir):
            raise FileNotFoundError("Logs directory does not exist")

        self.log_dirs = [name for name in os.listdir(logs_dir)
                         if os.path.isdir(os.path.join(logs_dir, name))]
        if len(self.log_dirs) == 0:
            FileNotFoundError("Empty directory")

        self.dataframe_dict = {
            dir_name: self.get_dataframes(
                os.listdir(os.path.join(logs_dir, dir_name)),
                prefix_dir=os.path.join(logs_dir, dir_name))
            for dir_name in self.log_dirs
        }

        self.setup()
        self.set_figure()
        self.set_components()
        self.default_check_boxes()

    def setup(self):
        super().setup()
        self.selected_frames = {}

    def default_check_boxes(self):
        for row_select, folder_name in zip(self.selector_widgets, self.log_dirs):
            if len(self.dataframe_dict[folder_name]) == 0:
                row_select.children[0].disabled = True
                continue
            lengths = np.array([len(df)
                                for df in self.dataframe_dict[folder_name]])
            if not np.all(lengths == lengths[0]):
                row_select.children[0].disabled = True

    def checkbox_handler(self, change):
        folder_name = change["owner"].description
        if change["new"] is True:
            self.selected_frames[folder_name] = self.dataframe_dict[folder_name]

        if change["new"] is False:
            if folder_name in self.selected_frames.keys():
                self.selected_frames.pop(folder_name)

        ynames, xnames = self._all_column_names()
        self.select_yaxis.options = ynames
        self.select_xaxis.options = xnames

    def _all_column_names(self):
        if len(self.selected_frames) == 0:
            return [], []

        intersect_ynames, intersect_xnames = None, None
        for frames in self.selected_frames.values():
            ynames, xnames = self._get_column_names(frames)
            if intersect_xnames is None:
                intersect_xnames = xnames
                intersect_ynames = ynames
            else:
                intersect_xnames = intersect_xnames & xnames
                intersect_ynames = intersect_ynames & ynames

        return intersect_ynames, intersect_xnames

    def set_components(self):
        row_layout = widgets.Layout(
            grid_template_columns="200px 200px 50px 100px",
            grid_gap="20px")

        self.selector_widgets = [
            widgets.GridBox([
                widgets.Checkbox(
                    value=False,
                    description=folder_name,
                    disabled=False,
                    indent=False
                ),
                widgets.Text(
                    "",
                    placeholder="Legend name",
                    layout=widgets.Layout(width="200px")),
                widgets.Label(str(len(self.dataframe_dict[folder_name]))),
                widgets.Text(
                    "",
                    placeholder="Color",
                    layout=widgets.Layout(width="100px")),
            ], layout=row_layout)
            for folder_name in self.log_dirs]

        for row_widget in self.selector_widgets:
            row_widget.children[0].observe(self.checkbox_handler, "value")

        header = widgets.GridBox([
            widgets.Label("Experiment Name"),
            widgets.Label("Legend Name"),
            widgets.Label("Count"),
            widgets.Label("Color")
        ],
            layout=row_layout)
        title_selector = widgets.Text(
            "",
            placeholder="Title",
            layout=widgets.Layout(width="200px"))
        title_selector.observe(self.set_title, "value")

        self.selector = widgets.VBox([title_selector, header] + self.selector_widgets)
        super().set_components()

    def set_title(self, change):
        self.fig.update_layout(title=change["new"])

    def _get_legend_name_nd_color(self, folder_name):
        for row_widget in self.selector_widgets:
            if row_widget.children[0].description == folder_name:
                return (row_widget.children[1].value, row_widget.children[3].value)

    def render_figure(self):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []

        for ix, (folder_name, dataframe) in enumerate(self.selected_frames.items()):
            legend_name, color = self._get_legend_name_nd_color(folder_name)
            if re.match("#[0-9a-fA-F]{6}", color) is None:
                color = self.default_colors[ix]
            self._add_traces(dataframe, color, legend_name)

        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            },
            showlegend=True
        )

    def __call__(self):
        out_display = widgets.Tab([
            self.selector,
            VBox([
                self.select_xaxis,
                self.select_yaxis,
                self.quantile_slider,
                self.fig,
            ])
        ],
            layout=Layout(
            width="96%",
            height="600px",
            display="flex",
            flex_flow="column",
            justify_content="space-around",
            border="solid 2px gray"
        )
        )
        out_display.set_title(0, "Select Log")
        out_display.set_title(1, "Figure")

        return out_display
