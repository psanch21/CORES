from __future__ import annotations

from torch_geometric.datasets import TUDataset

import cores.core.values.constants as cte
from cores.impl.dataset_preparators.pyg.graph.pp_pyg_tu import TUPreparator


class COX2Preparator(TUPreparator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=cte.Datasets.COX2, **kwargs)

    def cleaned(self):
        return False

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y.unsqueeze(-1).float()
            return data

        return transform

    def _get_dataset_raw(self) -> TUDataset:
        dataset = super()._get_dataset_raw()

        # # The remaining columns are constants
        x_cols = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19]

        dataset.data.x = dataset.data.x[:, x_cols]

        return dataset

    def classes_num(self) -> int:
        return 2

    def edge_attr_dim(self) -> int:
        return 0

    def features_dim(self) -> int:
        return 10

    def target_dim(self) -> int:
        return 1

    def target_num(self) -> int:
        return 1

    def exploratory_data_analysis(self, root: str):
        raise NotImplementedError
