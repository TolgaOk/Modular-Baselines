from typing import Any, Dict, List
import json
from IPython.display import display


def prepare_progress_data(file_names: List[str]) -> List[Dict[str, Any]]:
    progress_dicts = []
    for index, file_name in enumerate(file_names):
        with open(file_name, "r") as fobj:
            progress_dicts += [{"seed": f"{index+1}", **
                                json.loads(line)} for line in fobj.readlines()]
    return progress_dicts


def vega_single_seed_experiment(progress_dicts, x_key="time/steps"):
    names = list(progress_dicts[0].keys())
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Single experiment figure",
        "width": 500,
        "height": 300,
        "data": {"values": progress_dicts},
        "params": [
            {
                "name": "selected_legend",
                "value": "collector/env_reward_mean",
                #  "select": {"type": "point", "fields": ["legend_name"]},
                "bind": {"input": "select", "options": names, "name": "Choose y axis:  "}
                #  "bind": "legend"
            },
        ],
        "title": {
            "text": "Training Statistics",
            "fontSize": 20,
            "fontWeight": 300,
        },
        "transform": [
            {"fold": names, "as": ["legend_name", "legend_value"]},
            {"filter": {"field": "legend_name", "equal": {"expr": "selected_legend"}}},
        ],
        "encoding": {
            "x": {"field": x_key,
                  "type": "quantitative",
                  "axis": {
                      "offset": 10,
                      "titleFontSize": 14,
                      "titleFontWeight": 500,
                      "title": x_key,
                  }
                  },
            "color": {"field": "seed", "type": "nominal"},
            "opacity": {
                "condition": {"param": "selected_legend", "value": 1.0},
                "value": 0.0
            }
        },
        "layer": [
            {
                "mark": {"type": "circle", "opacity": 0.5, "size": 30},
                "encoding": {
                    "y": {"field": "legend_value", "type": "quantitative", "axis": {"title": "values", "offset": 10}},
                    "tooltip": [
                        {"field": x_key, "type": "quantitative", "title": "Timestep"},
                        {"field": "legend_value", "type": "quantitative", "title": "Y value"}
                    ],
                }
            },
            {
                "mark": {"type": "line", "color": "red", "size": 2},
                "transform": [
                    {"loess": "legend_value", "on": x_key, "bandwidth": 0.5}
                ],
                "encoding": {
                    "y": {"field": "legend_value", "type": "quantitative", "title": "Local REGRESSION", "axis": {"title": None}},
                    "color": {"legend": None}
                }
            }
        ]
    }


def vega_multi_seed_experiment(progress_dicts, x_key="time/steps"):
    names = list(progress_dicts[0].keys())
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Multi seed experiments figure",
        "width": 500,
        "height": 300,
        "data": {"values": progress_dicts},
        "params": [
            {
                "name": "selected_legend",
                "value": "collector/env_reward_mean",
                "bind": {"input": "select", "options": names, "name": "Choose y axis:  "}
            },
        ],
        "title": {
            "text": "Training Statistics",
            "fontSize": 20,
            "fontWeight": 300,
        },
        "transform": [
            {"fold": names, "as": ["legend_name", "legend_value"]},
            {"filter": {"field": "legend_name", "equal": {"expr": "selected_legend"}}},
        ],
        "encoding": {
            "x": {"field": x_key,
                  "type": "quantitative",
                  "axis": {
                          "offset": 10,
                          "titleFontSize": 14,
                          "titleFontWeight": 500,
                          "title": x_key,
                  }
                  },
        },
        "layer": [
            {
                "mark": {"type": "errorband",
                         "opacity": 0.5,
                         "interpolate": "basis",
                         "extent": "iqr",
                         "borders": {
                             "opacity": 0.2,
                             "strokeDash": [1, 1],
                             "color": "gray"
                         }},
                "encoding": {
                    "y": {"field": "legend_value",
                          "type": "quantitative",
                          "axis": {
                              "title": "values",
                              "offset": 10,
                              "titleFontSize": 14,
                              "titleFontWeight": 500,
                          }
                          },
                }
            },
            {
                "mark": {"type": "line", "opacity": 1.0, "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "mean",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}}
                }
            },
            {
                "mark": {"type": "line", "opacity": 0.5, "color": "gray", "strokeDash": [2, 3], "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "max",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}}
                }
            },
            {
                "mark": {"type": "line", "opacity": 0.5, "color": "gray", "strokeDash": [2, 3], "interpolate": "basis"},
                "encoding": {
                    "y": {
                        "aggregate": "min",
                        "field": "legend_value",
                    },
                    "color": {"field": "legend_name", "type": "nominal", "axis": {"title": "Traces"}}
                }
            },
        ]
    }


def vega_notebook_render(schema: Dict[str, Any]) -> None:
    display(
        {"application/vnd.vegalite.v5+json": schema},
        raw=True
    )
