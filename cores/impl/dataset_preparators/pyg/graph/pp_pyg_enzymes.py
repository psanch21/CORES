from __future__ import annotations

import os

from torch_geometric.datasets import TUDataset

import cores.core.values.constants as cte
import cores.utils.eda as geda
import cores.utils.plotter as gplotter
from cores.impl.dataset_preparators.pyg.graph.pp_pyg_tu import TUPreparator


class ENZYMESPreparator(TUPreparator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=cte.Datasets.ENZYMES, **kwargs)

    def cleaned(self):
        return False

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y

            data.node_color = data.x[:, -3:].argmax(dim=-1).unsqueeze(dim=-1)
            data.x_cat = data.x[:, -3:].argmax(dim=-1).unsqueeze(dim=-1)

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
        return 6

    def edge_attr_dim(self) -> int:
        return 0

    def features_dim(self) -> int:
        return 21

    def target_dim(self) -> int:
        return 6

    def target_num(self) -> int:
        return 1

    def exploratory_data_analysis(self, root: str):
        plotter = gplotter.MatplotlibPlotter()

        for split_name in self.split_names:
            dataset = self.datasets[split_name]
            loader = self._data_loader(
                dataset, batch_size=len(dataset), shuffle=False, num_workers=0
            )
            batch = next(iter(loader))

            root_split = os.path.join(root, split_name)
            eda_graph = geda.EDAGraph(root=root_split, plotter=plotter)

            eda_graph.report_batch(
                batch=batch,
                title=f"ENZYMES {split_name} Report",
                output_file=f"enzymes_report_{split_name}.html",
                plot=True,
                report_samples=[0],
            )
