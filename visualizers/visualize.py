import json

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

from .histvis import HistRender
from .scalarvis import ScalarRender


def render_layout(log_dir, layout):

    row_boxes = []
    for row in layout:
        row_vis = []
        for letter in row[:2]:
            if letter == "H":
                row_vis.append(HistRender(log_dir)())
            elif letter == "S":
                row_vis.append(ScalarRender(log_dir)())
            else:
                raise ValueError("Unrecognized letter!")
        row_boxes.append(VBox(row_vis,
            layout=Layout(
                width="100%",
                height="500px",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
        )))
    return VBox(row_boxes,
            layout=Layout(
                width="100%",
                height="{}px".format(str(500 * len(row_boxes))),
                display="flex",
                flex_flow="column",
                justify_content="space-around",
        ))

__all__ = ["visualize"]