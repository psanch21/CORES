from __future__ import annotations

import os
import random
from typing import Any, Dict

import torch
import torch_geometric.data as pygd
import torch_geometric.loader as pygl
from torch_geometric.utils import degree

import cores.core.values.constants as cte
from cores.core.contracts.dataset_preparator import BaseDatasetPreparator
from cores.utils.preprocessing import InvertiblePreprocessingDF


class PyGPreparator(BaseDatasetPreparator):
    @staticmethod
    def random_kwargs(seed: int):
        kwargs = BaseDatasetPreparator.random_kwargs(seed)
        kwargs["is_dense"] = random.choice([False])

        return kwargs

    def __init__(
        self,
        *args,
        root,
        is_dense: bool = False,
        transductive: bool,
        **kwargs,
    ):
        self.transductive = transductive
        self.is_dense = is_dense

        self.pp_edge = None

        root = os.path.join(root, "pyg")

        super().__init__(*args, root=root, **kwargs)

    @property
    def type_of_data(self) -> cte.DataTypes:
        return cte.DataTypes.GRAPH

    def _get_dataset_raw(self):
        raise NotImplementedError

    def _split_dataset(self, dataset_raw: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def classes_num(self) -> int:
        raise NotImplementedError

    def edge_attr_dim(self) -> int:
        raise NotImplementedError

    def features_dim(self) -> int:
        raise NotImplementedError

    def samples_num(self, split_name: cte.SplitNames = cte.SplitNames.TRAIN) -> int:
        raise NotImplementedError

    def target_dim(self) -> int:
        raise NotImplementedError

    def fit_preprocessor(self):
        if self.preprocessing_dict_y is None and self.preprocessing_dict_x is None:
            return

        data = self.get_train_data()
        if self.preprocessing_dict_x is not None and len(self.preprocessing_dict_x) > 0:
            self.pp_x = InvertiblePreprocessingDF(preprocessing_dict=self.preprocessing_dict_x)

            self.pp_x.fit(data.x.numpy())

            for split_name in self.split_names:
                x = self.datasets[split_name].data.x
                dtype = x.dtype
                device = x.device
                x_norm_np = self.pp_x.transform(x.numpy())

                x_norm = torch.from_numpy(x_norm_np).to(dtype=dtype, device=device)

                self.datasets[split_name].data.x = x_norm
        if self.preprocessing_dict_y is not None:
            self.pp_y = InvertiblePreprocessingDF(preprocessing_dict=self.preprocessing_dict_y)

            self.pp_y.fit(data.y.numpy())

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

    def get_y_from_dataset(self, dataset: Any) -> torch.Tensor:
        loader = self._data_loader(
            dataset=dataset, batch_size=self.samples_num(), shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        return batch.y

    def get_train_data(self) -> pygd.Batch:
        loader = self.get_dataloader_train(batch_size=self.samples_num())
        batch = next(iter(loader))
        return batch

    def set_dtype(self, x: torch.Tensor, dtype: str) -> torch.Tensor:
        return x.type(dtype)
