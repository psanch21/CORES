from __future__ import annotations

import networkx as nx
import torch_geometric.data as pygd
import torch_geometric.utils as pygu


class GraphConverter:
    def pyg_to_networkx(
        data: pygd.Data, node_attrs=["x"], edge_attrs=None, to_undirected=False
    ) -> nx.DiGraph | nx.Graph:
        """
        Convert a list of triples into a NetworkX directed graph.
        """

        graph = pygu.to_networkx(
            data,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            to_undirected=to_undirected,
            remove_self_loops=False,
        )
        return graph
