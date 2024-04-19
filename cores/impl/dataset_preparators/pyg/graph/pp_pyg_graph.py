from __future__ import annotations

import os
from typing import Any, Dict

import cores.core.values.constants as cte
from cores.impl.dataset_preparators.pyg.pp_pyg import PyGPreparator
from cores.utils import TorchUtils


class PyGGraphPreparator(PyGPreparator):
    @staticmethod
    def random_kwargs(seed: int):
        kwargs = PyGPreparator.random_kwargs(seed)

        return kwargs

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, transductive=False, **kwargs)

        self.root = os.path.join(self.root, "graph")

    def _get_dataset_raw(self):
        raise NotImplementedError

    def classes_num(self) -> int:
        raise NotImplementedError

    def edge_attr_dim(self) -> int:
        raise NotImplementedError

    def features_dim(self) -> int:
        raise NotImplementedError

    def target_dim(self) -> int:
        raise NotImplementedError

    def target_num(self) -> int:
        raise NotImplementedError

    def _split_dataset(self, dataset_raw: Any) -> Dict[str, Any]:
        # splits = TorchUtils.split_into_segments(
        #     original_len=len(dataset_raw),
        #     segment_sizes=self.split,
        #     k_fold=self.k_fold,
        # )

        splits = TorchUtils.stratify(
            y=dataset_raw.data.y.flatten(),
            original_len=len(dataset_raw),
            segment_sizes=self.split,
            k_fold=self.k_fold,
        )

        datasets_list = []
        for sp in splits:
            datasets_list.append(dataset_raw[sp])

        datasets = {}
        for i, split_name in enumerate(self.split_names):
            datasets[split_name] = datasets_list[i]

        return datasets

    def samples_num(self, split_name: cte.SplitNames = cte.SplitNames.TRAIN) -> int:
        return len(self.datasets[split_name])
