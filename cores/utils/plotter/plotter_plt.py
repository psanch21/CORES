from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from cores.utils.plotter.plotter import Plotter

layout_fn_dict = dict(
    circular=nx.circular_layout,
    kumada_layout=nx.kamada_kawai_layout,
    planar=nx.planar_layout,
    random=nx.random_layout,
    shell=nx.shell_layout,
    spectral=nx.spectral_layout,
    spiral=nx.spiral_layout,
    spring=nx.spring_layout,
)


class MatplotlibFigure:
    def __init__(self, fig: plt.Figure, axis: plt.Axes):
        self.fig = fig
        self.axis = axis


class MatplotlibPlotter(Plotter):
    @staticmethod
    def create_legend(
        handles_dict: Dict[str, Dict[str, Any]] = None,
        extra_dict: Dict[str, Dict[str, Any]] = None,
        num_columns: int = 2,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Any:
        if handles_dict is None:
            handles_dict = {
                "Model1": {"marker": "o", "linestyle": "solid", "color": "blue"},
                "Model2": {
                    "marker": "^",
                    "linestyle": "dashed",
                    "color": "green",
                    "markersize": 8,
                },  # etc.
            }

        if extra_dict is not None:
            for label, label_dict in handles_dict.items():
                label_dict.update(extra_dict)
        # Create custom handles
        custom_handles = []

        for label, label_dict in handles_dict.items():
            line = Line2D(
                [0],
                [0],
                label=label,
                **label_dict,
            )
            custom_handles.append(line)

        # Create a figure with reduced height
        fig2, ax2 = plt.subplots()  # Adjust the figsize as needed

        # Display legend on new axes
        labels = list(handles_dict.keys())  # Labels corresponding to your custom handles

        legend = ax2.legend(custom_handles, labels, loc="center", ncol=num_columns)

        if title is not None:
            legend.set_title(title)

        # Fit figure to legend size and remove empty space
        fig2.tight_layout()
        ax2.axis("off")

        # Save only the legend area without empty space

        if file_path is not None:
            fig2.savefig(file_path, bbox_inches="tight", pad_inches=0)
        else:
            # Show
            plt.show()

    def create_figure(self, *args, **kwargs) -> MatplotlibFigure:
        fig, ax = plt.subplots(*args, **kwargs)
        figure = MatplotlibFigure(fig, ax)
        return figure

    def get_axis(self, figure: MatplotlibFigure) -> plt.Axes:
        return figure.axis

    def close(self, figure: MatplotlibFigure) -> Any:
        plt.close(figure.fig)

    def close_all(self) -> None:
        plt.close("all")

    def save(self, figure: MatplotlibFigure, file_path: str) -> Any:
        figure.fig.savefig(file_path)

    def show(self, figure: MatplotlibFigure) -> None:
        figure.fig.show()

    def _get_nodes_position(
        self, graph_nx: nx.DiGraph | nx.Graph, layout_fn: Callable | str
    ) -> dict:
        if layout_fn == "pos":
            pos = nx.get_node_attributes(graph_nx, "pos")
        elif layout_fn is None:
            pos = nx.kamada_kawai_layout(graph_nx)
        else:
            pos = layout_fn(graph_nx)
        return pos

    def flatten_list(self, nested_list: List[List[Any]]) -> List[Any]:
        if isinstance(nested_list[0], list):
            return [item for sublist in nested_list for item in sublist]
        else:
            return nested_list

    def convert_color_id_to_str(self, color_list: List[int]) -> List[str]:
        color_list = self.flatten_list(color_list)

        if isinstance(color_list[0], str):
            return color_list

        colors_num = int(max(color_list) + 1)

        colours = self.colorblind_colors(colors_num)
        color_list_str = [colours[int(color)] for color in color_list]
        return color_list_str

    def _plot_graph_nx(
        self,
        graph: nx.DiGraph | nx.Graph,
        show: bool = True,
        file_path: str = None,
        layout_fn: Callable | str = None,
        node_color_attr: str | None = None,
        node_edgecolors_attr: str | None = None,
        edge_width_attr: str | None = None,
        edge_color_attr: str | None = None,
    ) -> None:
        if isinstance(layout_fn, str):
            layout_fn = layout_fn_dict[layout_fn]
        # Create a subplot with plt.subplots
        figure = self.create_figure()
        ax = self.get_axis(figure=figure)

        # Initialize default values
        pos = self._get_nodes_position(graph, layout_fn)
        node_color = self.colorblind_colors(1)[0]
        node_edgecolor = None
        edge_width = None
        edge_color = "k"

        if node_color_attr is not None:
            node_color = list(nx.get_node_attributes(graph, node_color_attr).values())
            if len(node_color) == 0:
                node_color = self.colorblind_colors(1)[0]
            node_color = self.convert_color_id_to_str(node_color)

        if node_edgecolors_attr is not None:
            node_edgecolor = list(nx.get_node_attributes(graph, node_edgecolors_attr).values())

        nx.draw_networkx_nodes(graph, pos, node_color=node_color, edgecolors=node_edgecolor, ax=ax)

        if edge_width_attr is not None:
            edge_width = list(nx.get_edge_attributes(graph, edge_width_attr).values())
            if len(edge_width) == 0:
                edge_width = None
            else:
                edge_width = self.flatten_list(edge_width)

        if edge_color_attr is not None:
            edge_color = list(nx.get_edge_attributes(graph, edge_color_attr).values())
            if len(edge_color) == 0:
                edge_color = "k"

            edge_color = self.convert_color_id_to_str(edge_color)

        # Draw edge thickness based on weight
        nx.draw_networkx_edges(graph, pos, width=edge_width, edge_color=edge_color)

        if show:
            self.show(figure)

        if file_path is not None:
            self.save(figure, file_path=file_path)
