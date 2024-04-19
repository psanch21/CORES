from __future__ import annotations

import os
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import torch_geometric.data as pygd
import torch_geometric.utils as pygu
from ydata_profiling import ProfileReport

from cores.utils.eda.eda import EDA


class EDAGraph(EDA):
    def report_nx(
        self,
        graph_nx: nx.DiGraph | nx.Graph,
        title: str = "Data Report",
        output_file: str = "report.htaml",
        plot: bool = True,
        **kwargs,
    ):
        output_file_no_ext = os.path.splitext(output_file)[0]

        output_file_x = f"{output_file_no_ext}_x.html"

        x_np = np.array(list(nx.get_node_attributes(graph_nx, "x").values()))

        df_x = pd.DataFrame(x_np, columns=[f"x{i}" for i in range(x_np.shape[1])])
        is_undi = not nx.is_directed(graph_nx)
        df_x["is_undirected"] = is_undi

        if is_undi:
            in_degree = np.array(list(dict(graph_nx.degree()).values()))
            out_degree = in_degree
        else:
            in_degree = np.array(list(dict(graph_nx.in_degree()).values()))
            out_degree = np.array(list(dict(graph_nx.out_degree()).values()))

        df_x["in_degree"] = in_degree
        df_x["out_degree"] = out_degree

        degree_centrality = list(nx.degree_centrality(graph_nx).values())
        df_x["degree_centrality"] = degree_centrality

        closeness_centrality = list(nx.closeness_centrality(graph_nx).values())
        df_x["closeness_centrality"] = closeness_centrality

        df_x["num_edges"] = graph_nx.number_of_edges()

        edge_attr = np.array(list(nx.get_edge_attributes(graph_nx, "edge_attr").values()))

        df_x["has_edge_attr"] = len(edge_attr) > 0

        profile = ProfileReport(df_x, title=title, **kwargs)

        profile.to_file(output_file=os.path.join(self.root, output_file_x))

        if len(edge_attr) > 0:
            output_file_e = f"{output_file_no_ext}_e.html"
            df_e = pd.DataFrame(edge_attr, columns=[f"e{i}" for i in range(edge_attr.shape[1])])

            df_e["num_edges"] = graph_nx.number_of_edges()

            profile = ProfileReport(df_e, title=title, **kwargs)

            df_e["is_undirected"] = is_undi

            profile.to_file(output_file=os.path.join(self.root, output_file_e))

        if plot and self.plotter is not None:
            output_file_plot = f"{output_file_no_ext}_graph.png"
            self.plotter.plot_graph(
                graph=graph_nx,
                file_path=os.path.join(self.root, output_file_plot),
                node_color_attr="node_color",
                edge_color_attr="edge_color",
                edge_width_attr="edge_width",
            )

    def report_batch(
        self,
        batch: pygd.Batch,
        title: str = "Data Report",
        output_file: str = "report.html",
        plot: bool = False,
        report_samples: List[int] = None,
        **kwargs,
    ):
        output_file_no_ext = os.path.splitext(output_file)[0]

        if report_samples is None:
            report_samples = []

        info_list = []
        for i, data in enumerate(batch.to_data_list()):
            graph_nx = self.to_networkx(graph=data)
            graph_dict = self.graph_dict(graph=graph_nx)
            info_list.append(graph_dict)
            output_file_i = f"{output_file_no_ext}_{i}.html"

            if i in report_samples:
                self.report_nx(
                    graph_nx=graph_nx, title=title, output_file=output_file_i, plot=plot, **kwargs
                )

        df = pd.DataFrame(info_list)

        profile = ProfileReport(df, title=title, **kwargs)

        output_file_graph = f"{output_file_no_ext}_graph.html"

        profile.to_file(output_file=os.path.join(self.root, output_file_graph))

    def graph_dict(self, graph: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
        graph_dict = dict()

        graph_dict["num_nodes"] = graph.number_of_nodes()
        graph_dict["num_edges"] = graph.number_of_edges()

        graph_dict["is_directed"] = nx.is_directed(graph)

        graph_dict["density"] = nx.density(graph)

        try:
            graph_dict["diameter"] = nx.diameter(graph)
        except nx.NetworkXError:
            graph_dict["diameter"] = 0

        # graph_dict["radius"] = nx.radius(graph)

        graph_dict["number_of_connected_components"] = nx.number_connected_components(
            graph.to_undirected()
        )

        # graph_dict["edge_connectivity"] = nx.edge_connectivity(graph)

        # graph_dict["average_clustering"] = nx.average_clustering(graph)

        # graph_dict["is_bipartite"] = nx.is_bipartite(graph)

        for name, attr in graph.graph.items():
            graph_dict[name] = attr

        return graph_dict

    def to_networkx(self, graph: pygd.Data) -> nx.DiGraph | nx.Graph:
        is_undirected = pygu.is_undirected(graph.edge_index)
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        node_attrs, edge_attrs = [], []
        graph_attr = dict()

        for name_el, el in graph:
            if el.shape[0] == num_edges:
                edge_attrs.append(name_el)
            elif el.shape[0] == num_nodes:
                node_attrs.append(name_el)
            elif el.numel() == 1:
                graph_attr[name_el] = el.item()

        graph_nx = pygu.to_networkx(
            graph, node_attrs=node_attrs, edge_attrs=edge_attrs, to_undirected=is_undirected
        )

        # Add graph attributes
        graph_nx.graph.update(graph_attr)

        return graph_nx

    def report(
        self,
        graph: pygd.Data | nx.DiGraph | nx.Graph,
        title: str = "Data Report",
        output_file: str = "report.html",
        plot: bool = True,
        **kwargs,
    ):
        if isinstance(graph, pygd.Data):
            graph_nx = self.to_networkx(graph)

        else:
            graph_nx = graph

        self.report_nx(graph_nx=graph_nx, title=title, output_file=output_file, plot=plot, **kwargs)
