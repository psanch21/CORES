from .v1 import Graph


class GraphV2(Graph):
    def __init__(self, x_dim=None, edge_dim=None, deg=None, dtype=None):
        super(GraphV2, self).__init__(x_dim=x_dim + 1, edge_dim=edge_dim, deg=deg, dtype=dtype)
