from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import networkx as nx
import seaborn as sns
import torch_geometric.data as pygd
import torch_geometric.utils as pygu

from cores.core.contracts.graph import Graph


class Figure:
    pass


class Axis:
    pass


color_mapping = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 0, 0],  # Maroon
    [0, 128, 0],  # Green (dark)
    [0, 0, 128],  # Navy
    [128, 128, 128],  # Gray
]


class Plotter(ABC):
    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int] | tuple[float, float, float]):
        if isinstance(rgb[0], float):
            rgb = tuple([int(x * 255) for x in rgb])
        return "#%02x%02x%02x" % rgb

    @staticmethod
    def hex_to_rgb(hex: str, format: str = "float") -> int | float:
        hex = hex.lstrip("#")
        hlen = len(hex)
        rgb = tuple(int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))
        if format == "float":
            rgb = tuple([x / 255 for x in rgb])
        return rgb

    @staticmethod
    def colorblind_colors(n: int, idx: int = 0, format: str = "hex"):
        color_list_rgb = sns.color_palette("colorblind")

        color_list = [Plotter.rgb_to_hex(x) for x in color_list_rgb]

        color_list.extend(
            [
                "#803E75",  # Strong Purple
                "#A6BDD7",  # Very Light Blue
                "#CEA262",  # Grayish Yellow
                "#007D34",  # Vivid Green
                "#00538A",  # Strong Blue
                "#53377A",  # Strong Blueish Purple
                "#FFFFFF",  # White
                "#000000",  # Black
            ]
        )

        # Shift idx positions
        color_list = color_list[idx:] + color_list[:idx]
        color_list_n = color_list[:n]

        if format == "hex":
            return color_list_n
        else:
            return [Plotter.hex_to_rgb(x, format=format) for x in color_list_n]
        return

    @abstractmethod
    def create_figure(self, *args, **kwargs) -> Figure:
        pass

    @abstractmethod
    def get_axis(self, figure: Figure) -> Axis:
        pass

    @abstractmethod
    def close(self, figure: Any) -> None:
        pass

    @abstractmethod
    def close_all(self) -> None:
        pass

    @abstractmethod
    def save(self, figure: Any, file_path: str) -> None:
        pass

    @abstractmethod
    def show(self, figure: Any) -> None:
        pass

    def plot_graph(
        self,
        graph: nx.DiGraph | nx.Graph | pygd.Data,
        show: bool = True,
        file_path: str = None,
        layout_fn: Callable | str = None,
        node_color_attr: str | None = None,
        node_edgecolors_attr: str | None = None,
        edge_width_attr: str | None = None,
        edge_color_attr: str | None = None,
    ) -> None:
        if isinstance(graph, pygd.Data):
            is_undi = not graph.is_directed()
            node_attrs = []
            edge_attrs = []

            if node_color_attr is not None:
                node_attrs.append(node_color_attr)
            if node_edgecolors_attr is not None:
                node_attrs.append(node_edgecolors_attr)

            if edge_width_attr is not None:
                edge_attrs.append(edge_width_attr)
            if edge_color_attr is not None:
                edge_attrs.append(edge_color_attr)

            if len(node_attrs) == 0:
                node_attrs = None
            if len(edge_attrs) == 0:
                edge_attrs = None

            graph_nx = pygu.to_networkx(
                graph, to_undirected=is_undi, node_attrs=node_attrs, edge_attrs=edge_attrs
            )

        elif isinstance(graph, (nx.DiGraph, nx.Graph)):
            graph_nx = graph
        else:
            raise NotImplementedError(f"Graph type {type(graph)} not supported")

        self._plot_graph_nx(
            graph=graph_nx,
            show=show,
            file_path=file_path,
            layout_fn=layout_fn,
            node_color_attr=node_color_attr,
            node_edgecolors_attr=node_edgecolors_attr,
            edge_width_attr=edge_width_attr,
            edge_color_attr=edge_color_attr,
        )

    @abstractmethod
    def _plot_graph_nx(
        self,
        graph: Graph,
        show: bool = True,
        file_path: str = None,
        layout_fn: Callable | str = None,
        node_color_attr: str | None = None,
        node_edgecolors_attr: str | None = None,
        edge_width_attr: str | None = None,
    ):
        pass
