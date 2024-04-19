from __future__ import annotations

import os

import cores.core.values.constants as cte
import cores.utils.eda as geda
import cores.utils.plotter as gplotter
from cores.impl.dataset_preparators.pyg.graph.pp_pyg_tu import TUPreparator


class MUTAGPreparator(TUPreparator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=cte.Datasets.MUTAG, **kwargs)

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y.unsqueeze(-1).float()
            data.node_color = data.x.argmax(dim=-1).unsqueeze(dim=-1)
            data.edge_width = data.edge_attr.argmax(dim=-1).unsqueeze(dim=-1) + 1
            return data

        return transform

    def classes_num(self) -> int:
        return 2

    def edge_attr_dim(self) -> int:
        return 4

    def features_dim(self) -> int:
        return 7

    def target_dim(self) -> int:
        return 1

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
                title=f"MUTAG {split_name} Report",
                output_file=f"mutag_report_{split_name}.html",
                plot=True,
                report_samples=[0],
            )
