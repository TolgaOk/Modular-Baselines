import optuna
import os
import pandas as pd
from collections import namedtuple, defaultdict
import ipywidgets
from itertools import chain
import numpy as np
import plotly.graph_objects as go


class TuneScalarRender():

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

    def __init__(self, logs_dir):
        if not os.path.exists(logs_dir):
            raise FileNotFoundError("Logs directory does not exist")

        self.dataframes_dict = defaultdict(list)
        self.log_dirs_dict = {name: os.path.join(logs_dir, name) for name in os.listdir(logs_dir)
                              if os.path.isdir(os.path.join(logs_dir, name))}

        storage_url = "".join(("sqlite:///", os.path.join(
            logs_dir,
            "store.db")))
        self.study = optuna.load_study(
            study_name=logs_dir.split("/")[-1],
            storage=storage_url)
        self.study_df = self.study.trials_dataframe()
        self.params = [param for param in self.study_df.columns
                       if param.startswith("params_") and not param.endswith("seed")]
        self.param_tuple = namedtuple("ParamTuple", " ".join(self.params))

        for index, row in self.study_df.iterrows():
            if not row.state == "COMPLETE":
                continue
            row = row[self.params]
            trial_name = "Trial_{:4d}".format(index + 1).replace(" ", "0")
            dir_path = self.log_dirs_dict[trial_name]
            for sub_dir in os.listdir(dir_path):
                dataframe = pd.read_csv(os.path.join(dir_path, sub_dir, "progress.csv"))
                self.dataframes_dict[self.param_tuple(**row.to_dict())].append(dataframe)

        self.param_activations = {name: False for name in self.params}
        self.make_figure()
        self.x_name = None
        self.y_name = None
        self.quantile_value = 0.25
        self.make_sliders_and_axis_cheeckbox()

    def make_figure(self):
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True)

    def make_sliders_and_axis_cheeckbox(self):
        all_dfs = list(chain(df_list for df_list in self.dataframes_dict.values()))[0]
        names, intersect_monotonic_names = self._get_column_names(all_dfs)

        self.select_yaxis = ipywidgets.Dropdown(
            options=names,
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)

        self.select_xaxis = ipywidgets.Dropdown(
            options=intersect_monotonic_names,
            value=None,
            description="X axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_xaxis.observe(self.set_x_axis)

        self.quantile_slider = ipywidgets.FloatSlider(
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

    def make_param_checkbox(self, n_columns=3):
        grid = ipywidgets.GridspecLayout(
            int(np.ceil(len(self.params) / n_columns)), n_columns, width="100%")
        for index, param in enumerate(self.params):
            widget = ipywidgets.Checkbox(
                value=False,
                description=param[7:],
                disabled=False,
                indent=False,
            )
            grid[index // n_columns, index % n_columns] = widget
            widget.observe(self.param_handler)
        return grid

    def param_handler(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.param_activations["_".join(("params", change["owner"].description))] = change["new"]
        self.render_figure()

    def set_quantile(self, change):
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.quantile_value = change["new"]
        self.render_figure()

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
        for index, (active_param, df_list) in enumerate(self.get_active_frame_dict().items()):
            color = self.default_colors[index % len(self.default_colors)]
            name = ", ".join(("{}: {}".format(key[7:], value)
                             for key, value in active_param._asdict().items()))
            self._add_traces(df_list, color, legend_name=name)

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
                name=legend_name,
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
                name=legend_name,
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
        grid = self.make_param_checkbox()
        out_display = ipywidgets.VBox([
            grid,
            self.select_xaxis,
            self.select_yaxis,
            self.quantile_slider,
            self.fig,
        ],
            layout=ipywidgets.Layout(
                width="95%",
                height="760px",
                display="flex",
                flex_flow="column",
                justify_content="space-around",
                border="solid 2px gray"
        ))
        return out_display

    def get_active_frame_dict(self):
        active_params = tuple(key for key, value in self.param_activations.items() if value)
        active_tuple = namedtuple("ActiveParams", " ".join(active_params))
        active_df_dict = defaultdict(list)
        for param_tuple, df_list in self.dataframes_dict.items():
            active_param_dict = {key: value for key,
                                 value in param_tuple._asdict().items() if key in active_params}
            active_param = active_tuple(**active_param_dict)
            for df in df_list:
                active_df_dict[active_param].append(df)

        return active_df_dict

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

    def plot_optimization(self):
        return optuna.visualization.plot_optimization_history(self.study)
