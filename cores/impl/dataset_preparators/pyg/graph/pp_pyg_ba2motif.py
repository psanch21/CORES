from __future__ import annotations

import random

import torch
import torch_geometric.data as pygd
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif

import cores.core.values.constants as cte
from cores.impl.dataset_preparators.pyg.graph.pp_pyg_graph import PyGGraphPreparator


class MyBA2MotifsDataset(pygd.InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])
        self.data, self.slices = pygd.InMemoryDataset.collate(data_list=data_list)

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        self.save(self.data_list, self.processed_paths[0])


class BA2MotifPreparator(PyGGraphPreparator):
    @staticmethod
    def random_kwargs(seed: int):
        kwargs = PyGGraphPreparator.random_kwargs(seed)

        kwargs["num_graphs"] = random.choice([100, 200, 500])
        kwargs["num_motifs"] = random.choice([1, 2, 3])
        kwargs["ba_num_nodes"] = random.choice([7, 10, 15])
        kwargs["ba_num_edges"] = random.choice([2, 3, 5])

        return kwargs

    def __init__(
        self,
        *args,
        num_graphs: int,
        num_motifs: int = 1,
        ba_num_nodes: int = 10,
        ba_num_edges: int = 3,
        **kwargs,
    ):
        self.num_graphs = num_graphs
        self.num_motifs = num_motifs
        self.ba_num_nodes = ba_num_nodes
        self.ba_num_edges = ba_num_edges
        super().__init__(*args, name=cte.Datasets.BA2MOTIFS, **kwargs)

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y.float()
            return data

        return transform

    def _get_dataset_raw(self) -> MyBA2MotifsDataset:
        dataset_h = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=self.ba_num_nodes, num_edges=self.ba_num_edges),
            motif_generator=HouseMotif(),
            num_motifs=self.num_motifs,
            num_graphs=self.num_graphs,
        )

        dataset_c = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=self.ba_num_nodes, num_edges=self.ba_num_edges),
            motif_generator=CycleMotif(5),
            num_motifs=self.num_motifs,
            num_graphs=self.num_graphs,
        )

        data_list = []
        features_dim = self.features_dim()

        for i, dataset_i in enumerate([dataset_h, dataset_c]):
            for idx in range(len(dataset_i)):
                data = dataset_i[idx]
                node_mask = data.node_mask

                size = (data.num_nodes, features_dim)

                x = torch.zeros(size=size)

                size = (node_mask.sum().long(), features_dim)
                x[node_mask == 1.0, :] = torch.randn(size=size) - (i - 0.5)

                y = torch.LongTensor([[i]])

                data = pygd.Data(x=x, y=y, edge_index=data.edge_index)

                data_list.append(data)
        dataset = MyBA2MotifsDataset(
            root=self.root, data_list=data_list, transform=self.get_transform_fn()
        )

        # dataset = BA2MotifDataset(root=self.root, transform=self.get_transform_fn())

        # in_degree = PyGUtils.in_degree(dataset.data.edge_index, dataset.data.num_nodes)

        # out_degree = PyGUtils.out_degree(dataset.data.edge_index, dataset.data.num_nodes)

        # dataset.data.x[:, -2] = in_degree
        # dataset.data.x[:, -1] = out_degree
        # dataset.data.x = dataset.data.x[:, -2:]

        return dataset

    def classes_num(self) -> int:
        return 2

    def edge_attr_dim(self) -> int:
        return 0

    def features_dim(self) -> int:
        return 5

    def target_dim(self) -> int:
        return 1

    def target_num(self) -> int:
        return 1
