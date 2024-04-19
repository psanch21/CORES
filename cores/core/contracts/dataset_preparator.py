from __future__ import annotations

import os.path
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

import cores.core.values.constants as cte
import cores.core.values.typing as corest
from cores.core.values.constants import DataBalanceTypes, DataTypes
from cores.core.values.typing import SplitsList
from cores.utils import FileIO, NPUtils


class BaseDatasetPreparator(ABC):
    @staticmethod
    def random_kwargs(seed):
        np.random.seed(seed)
        random.seed(seed)
        kwargs = {}

        splits_num = np.random.randint(1, 5)

        # Sample from  a dirichlet_distribution
        kwargs["splits"] = np.random.dirichlet(
            [
                5,
            ]
            * splits_num
        )

        kwargs["shuffle_train"] = random.choice([True, False])
        kwargs["single_split"] = random.choice([True, False])
        kwargs["k_fold"] = random.choice([-1, 1, 2, 3, 4, 5])

        kwargs["root"] = os.path.join("..", "data")
        return kwargs

    def __init__(
        self,
        name: str,
        splits: SplitsList,
        shuffle_train: bool,
        single_split: bool,
        k_fold: int,
        root: str,
        preprocessing_dict_x: Optional[Dict[str, Dict[str, Any]]] = None,
        preprocessing_dict_y: Optional[Dict[str, Dict[str, Any]]] = None,
        balance: Optional[DataBalanceTypes] = None,
        device: str = "cpu",
    ):
        self.name = name

        self.split = splits

        if len(splits) == 1:
            self.split_names = [cte.SplitNames.TRAIN]
        elif len(splits) == 2:
            self.split_names = [cte.SplitNames.TRAIN, cte.SplitNames.TEST]
        elif len(splits) == 3:
            self.split_names = [
                cte.SplitNames.TRAIN,
                cte.SplitNames.VALID,
                cte.SplitNames.TEST,
            ]
        elif len(splits) == 4:
            self.split_names = [
                cte.SplitNames.TRAIN,
                cte.SplitNames.VALID,
                cte.SplitNames.CALIB,
                cte.SplitNames.TEST,
            ]
        else:
            raise ValueError(f"Invalid number of splits: {len(splits)}")
        self.current_split = None
        assert np.isclose(sum(splits), 1.0), f"Splits: {splits} {sum(splits)}"
        self.shuffle_train = shuffle_train
        self.single_split = single_split
        self.k_fold = k_fold

        self.preprocessing_dict_x = preprocessing_dict_x
        self.preprocessing_dict_y = preprocessing_dict_y

        self.pp_x = None
        self.pp_y = None

        self.datasets = None

        self.device = device

        root = os.path.join(root, name)

        if not os.path.exists(root):
            FileIO.makedirs(root)

        self.root = root

        self.weight = None
        self.balance = balance

    @property
    def type_of_data(self) -> DataTypes:
        raise NotImplementedError

    @abstractmethod
    def _data_loader(self, dataset, batch_size: int, shuffle: bool, **kwargs):
        pass

    @abstractmethod
    def _get_dataset_raw(self):
        pass

    @abstractmethod
    def _get_target(self, batch: Any) -> corest.Tensor2D:
        pass

    @abstractmethod
    def _split_dataset(self, dataset_raw: corest.Dataset) -> Dict[str, corest.Dataset]:
        pass

    @abstractmethod
    def classes_num(self) -> int:
        pass

    @abstractmethod
    def features_dim(self) -> int:
        pass

    @abstractmethod
    def fit_preprocessor(self):
        pass

    def get_dataset_train(self) -> corest.Dataset:
        return self.datasets[cte.SplitNames.TRAIN]

    @abstractmethod
    def get_train_data(self) -> Any:
        pass

    @abstractmethod
    def get_y_from_dataset(self, dataset: corest.Dataset) -> corest.Tensor2D:
        pass

    @abstractmethod
    def samples_num(self, split_name: cte.SplitNames = cte.SplitNames.TRAIN) -> int:
        pass

    @abstractmethod
    def set_dtype(self, x: corest.Tensor, dtype: str) -> corest.Tensor:
        pass

    @abstractmethod
    def target_dim(self) -> int:
        pass

    def get_target(self, batch: Any, dtype: Optional[str] = None) -> corest.Tensor2D:
        target = self._get_target(batch)

        if dtype is not None:
            target = self.set_dtype(target, dtype)
        return target

    # Implemented methods

    def exploratory_data_analysis(self, root: str):
        raise NotImplementedError

    @torch.no_grad()
    def on_start(self, device):
        self.device = device

    def _transform_dataset_pre_split(self, dataset_raw):
        return dataset_raw

    def prepare(self):
        dataset_raw = self._get_dataset_raw()

        dataset_raw = self._transform_dataset_pre_split(dataset_raw=dataset_raw)

        datasets = self._split_dataset(dataset_raw)
        self.datasets = datasets

        y = self.get_y_from_dataset(datasets[cte.SplitNames.TRAIN])

        self.fit_preprocessor()

        if y.ndim == 2:
            assert y.shape[-1] == 1
        y_np = y.flatten().numpy()

        if self.balance in [cte.DataBalanceTypes.DOWNSAMPLE, cte.DataBalanceTypes.UPSAMPLE]:
            idx_balanced = NPUtils.resample(
                target=y_np,
                upsample=self.balance == cte.DataBalanceTypes.UPSAMPLE,
                replace=True,
                random_state=self.k_fold,
            )

            datasets[0] = datasets[0][idx_balanced]
        elif self.balance == cte.DataBalanceTypes.WEIGHT:
            self.weight = compute_class_weight("balanced", classes=np.unique(y_np), y=y_np)

        if self.single_split in self.split_names:
            idx = self.split_names.index(self.single_split)
            for i in range(len(datasets)):
                if i != idx:
                    datasets[i] = datasets[idx]
        datasets = self._transform_after_split(datasets)
        self.datasets = datasets
        return

    def set_current_split(self, i):
        if isinstance(self.single_split, str):
            self.current_split = self.single_split
        else:
            self.current_split = self.split_names[i]

    def _transform_after_split(self, datasets):
        return datasets

    def get_dataloader_train(self, batch_size, num_workers=0, shuffle=None):
        assert isinstance(self.datasets, dict)

        dataset = self.datasets[cte.SplitNames.TRAIN]
        shuffle = self.shuffle_train if shuffle is None else shuffle
        loader_train = self._data_loader(
            dataset, batch_size, shuffle=shuffle, num_workers=num_workers
        )

        return loader_train

    def get_dataloader(self, split: str, batch_size: int, shuffle: bool = True, **kwargs):
        dataset = self.datasets[split]
        return self._data_loader(dataset, batch_size, shuffle=shuffle, **kwargs)

    def get_dataloaders(self, batch_size, num_workers=0) -> Dict[str, Any]:
        assert isinstance(self.datasets, dict)
        loader_train = self.get_dataloader_train(batch_size, num_workers)

        loaders = {cte.SplitNames.TRAIN: loader_train}
        for name_i, dataset_i in self.datasets.items():
            if name_i == cte.SplitNames.TRAIN:
                continue
            loader = self._data_loader(
                dataset_i, batch_size, shuffle=False, num_workers=num_workers
            )
            loaders[name_i] = loader

        return loaders
