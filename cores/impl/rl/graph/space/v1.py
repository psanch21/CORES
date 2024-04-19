import torch
from gymnasium.spaces import Space
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph


class Graph(Space):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.

    WARNING - Custom observation & action spaces can inherit from the `Space`
    class. However, most use-cases should be covered by the existing space
    classes (e.g. `Box`, `Discrete`, etc...), and container classes (`Tuple` &
    `Dict`). Note that parametrized probability distributions (through the
    `sample()` method), and batching functions (in `gym.vector.VectorEnv`), are
    only well-defined for instances of spaces provided in gym by default.
    Moreover, some implementations of Reinforcement Learning algorithms might
    not handle custom spaces properly. Use custom spaces with care.
    """

    def __init__(self, x_dim=None, edge_dim=None, deg=None, dtype=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.x_dim = None if x_dim is None else x_dim
        self.edge_dim = None if edge_dim is None else edge_dim
        self.deg = deg
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if
        sampling from this space.
        """
        if self._np_random is None:
            self.seed()

        return self._np_random

    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        num_nodes = 5
        edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=0.9)
        num_edges = edge_index.shape[1]
        x = torch.rand([num_nodes, self.x_dim])
        edge_attr = torch.rand([num_edges, self.edge_dim])
        # batch = torch.zeros(num_nodes)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        value = True
        if hasattr(x, "batch"):
            return False
        for attr_name in ["x", "edge_index", "edge_attr"]:
            if not hasattr(x, attr_name):
                value = False
                break

        return value

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

    def __repr__(self):
        return f"Graph(x_dim={self.x_dim}, edge_dim={self.edge_dim}, {self.dtype})"

    def __eq__(self, other):
        return (
            isinstance(other, Graph)
            and (self.x_dim == other.x_dim)
            and (self.edge_dim == other.edge_dim)
        )
