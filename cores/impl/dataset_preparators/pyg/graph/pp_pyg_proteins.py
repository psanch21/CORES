from __future__ import annotations

from torch_geometric.datasets import TUDataset

import cores.core.values.constants as cte
from cores.impl.dataset_preparators.pyg.graph.pp_pyg_tu import TUPreparator


class ProteinsPreparator(TUPreparator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=cte.Datasets.PROTEINS, **kwargs)

    def cleaned(self):
        return False

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y.unsqueeze(-1).float()
            return data

        return transform

    def _get_dataset_raw(self) -> TUDataset:
        dataset = super()._get_dataset_raw()

        x_0 = dataset.data.x[:, 0]

        # Get 1% and 99% quantiles
        q_01 = x_0.quantile(0.001)
        q_99 = x_0.quantile(0.999)

        # Set outliers to 1% and 99% quantiles
        dataset.data.x[x_0 < q_01, 0] = q_01
        dataset.data.x[x_0 > q_99, 0] = q_99

        return dataset

    def classes_num(self) -> int:
        return 2

    def edge_attr_dim(self) -> int:
        return 0

    def features_dim(self) -> int:
        return 4

    def target_dim(self) -> int:
        return 1

    def target_num(self) -> int:
        return 1

    def exploratory_data_analysis(self, root: str):
        raise NotImplementedError
