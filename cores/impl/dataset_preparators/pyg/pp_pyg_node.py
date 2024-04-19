from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch_geometric.data as pygd
import torch_geometric.loader as pygl
from torch_geometric.utils import degree

import cores.core.values.constants as cte
from cores.core.contracts.dataset_preparator import BaseDatasetPreparator
from cores.utils import PyGUtils, TorchUtils


class PyGPreparator(BaseDatasetPreparator):
    def __init__(
        self,
        *args,
        name: str,
        is_dense: bool,
        transductive: bool,
        **kwargs,
    ):
        self.transductive = transductive
        self.is_dense = is_dense

        super().__init__(*args, name=name, **kwargs)

        self.root = os.path.join(self.root, "graph")

    @property
    def type_of_data(self) -> cte.DataTypes:
        return cte.DataTypes.GRAPH

    def _get_dataset_raw(self):
        raise NotImplementedError

    def classes_num(self) -> int:
        raise NotImplementedError

    def edge_attr_dim(self) -> int:
        raise NotImplementedError

    def features_dim(self) -> int:
        raise NotImplementedError

    def get_y_from_dataset(self, dataset: Any) -> torch.Tensor:
        raise NotImplementedError

    def samples_num(self, split_name: cte.SplitNames = cte.SplitNames.TRAIN) -> int:
        raise NotImplementedError

    def _data_loader(self, dataset, batch_size, shuffle, num_workers=0):
        if self.is_dense:
            return pygl.DenseDataLoader(dataset, batch_size=batch_size)

        else:
            return pygl.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
            )

    def _get_target(self, batch: pygd.Batch) -> torch.Tensor:
        return batch.y

    def _split_dataset(self, dataset_raw: pygd.Data) -> Dict[str, pygd.Data]:
        if self.transductive:
            assert isinstance(dataset_raw, pygd.Data)
            datasets_list = PyGUtils.transductive_split_data(
                data=dataset_raw,
                split_sizes=self.split,
                task_domain=self.task_domain,
                k_fold=self.k_fold,
            )
        else:
            splits = TorchUtils.split_into_segments(
                original_len=len(dataset_raw),
                split_sizes=self.split,
                k_fold=self.k_fold,
            )
            datasets_list = []
            for sp in splits:
                datasets_list.append(dataset_raw[sp])

        datasets = {}
        for i, split_name in enumerate(self.split_names):
            datasets[split_name] = datasets_list[i]

        return datasets

    def get_deg(self) -> torch.Tensor:
        loader = self.get_dataloader_train(batch_size=1)

        max_degree = 0
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
            # Compute the in-degree histogram tensor
        deg_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())

        return deg_histogram

    def get_dataset_train(self) -> pygl.DataLoader | pygl.DenseDataLoader:
        return self.datasets[cte.SplitNames.TRAIN]

    def get_features_train(self) -> torch.Tensor:
        loader = self.get_dataloader_train(batch_size=self.num_samples())
        batch = next(iter(loader))
        return batch.x
