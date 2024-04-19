from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

from cores.core.entities.edge_id import EdgeID
from cores.core.entities.node_id import NodeID
from cores.core.values.constants import GraphFramework


class Graph(ABC):
    def __init__(self, graph: Any):
        self.graph = graph

    @abstractmethod
    def mode(self) -> GraphFramework:
        pass

    @property
    def num_nodes(self) -> int:
        raise NotImplementedError

    @property
    def num_edges(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, num_edges={self.num_edges})"

    def describe(self) -> str:
        my_str = f"DiGraph(num_nodes={self.num_nodes}, num_edges={self.num_edges})\n\n"
        for node_id, attr in self.nodes(data=True):
            my_str += f"{node_id}:\n\t{attr}\n"

        for edge_id, attr in self.edges(data=True):
            my_str += f"{edge_id}:\n\t{attr}\n"

        return my_str

    @abstractmethod
    def get_graph(self, framework: GraphFramework = None) -> Any:
        pass

    @abstractmethod
    def edges(
        self, data: bool = False
    ) -> Generator[EdgeID] | Generator[tuple[EdgeID, dict[str, Any]]]:
        pass

    @abstractmethod
    def get_edge(self, edge_id: EdgeID) -> Any:
        pass

    @abstractmethod
    def add_edge(self, edge_id: EdgeID, **attr) -> None:
        pass

    @abstractmethod
    def nodes(self, data=False) -> Generator[NodeID] | Generator[tuple[NodeID, dict[str, Any]]]:
        pass

    @abstractmethod
    def relabel_nodes(self, mapping: list[tuple[NodeID, NodeID]]) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: NodeID) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_nodes_attribute(self, attr: str, default: Any = None) -> Generator[Any]:
        pass

    @abstractmethod
    def neighbors(self, node_id: NodeID, mode: int = "default") -> Generator[NodeID]:
        pass

    @abstractmethod
    def neighbor_edges(self, node_id: NodeID, mode: int = "all") -> list[EdgeID]:
        pass

    @abstractmethod
    def get_edge_id_from_nodes(self, head: NodeID, tail: NodeID) -> EdgeID:
        pass

    @abstractmethod
    def set_edge_attr(self, edge_id: EdgeID, attr: str, value: Any) -> None:
        pass

    @abstractmethod
    def get_edge_attr(
        self, edge_id: EdgeID, attr: str | None = None, default: Any = None
    ) -> dict[str, Any] | Any:
        pass

    @abstractmethod
    def get_edges_attribute(self, attr: str, default: Any = None) -> Generator[Any]:
        pass

    @abstractmethod
    def get_edge_head(self, edge_id: EdgeID) -> NodeID:
        pass

    @abstractmethod
    def get_edge_tail(self, edge_id: EdgeID) -> NodeID:
        pass

    @abstractmethod
    def get_k_hop_subgraph_from_edge(
        self, edge_id: EdgeID, k: int, undirected: bool = False
    ) -> Graph:
        pass

    def subgraph_by_attribute(
        self,
        attr: str,
        default: Any = None,
        k: int = None,
        min_score: float = None,
        normalize: bool = False,
    ):
        if k is not None:
            index_list = self.get_top_k_edges_by_attribute(k, attr, default)

        elif min_score is not None:
            index_list = self.get_edges_by_attribute_min_score(min_score, attr, default, normalize)

        subgraph = self.subgraph_from_edges(index_list=index_list)

        return subgraph

    @abstractmethod
    def get_top_k_edges_by_attribute(self, k: int, attr: str, default: Any = None):
        pass

    @abstractmethod
    def get_edges_by_attribute_min_score(
        self, min_score: float, attr: str, default: Any = None, normalize: bool = False
    ):
        pass

    @abstractmethod
    def subgraph_from_edges(
        self, id_list: list[EdgeID] = None, index_list: list[int] = None
    ) -> Graph:
        pass

    def get_edge_ids_by(
        self, attr: str, value: Any, only_one: bool = False
    ) -> list[EdgeID] | EdgeID:
        edge_ids = [
            edge_id for edge_id in self.edges() if self.get_edge_attr(edge_id, attr) == value
        ]

        if only_one:
            assert len(edge_ids) == 1
            return edge_ids[0]
        else:
            return edge_ids
