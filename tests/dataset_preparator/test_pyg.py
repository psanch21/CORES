from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest
import torch
import torch_geometric.data as pygd
import torch_geometric.loader as pygl

import cores.impl.dataset_preparators as corespp
from cores.core.contracts.dataset_preparator import BaseDatasetPreparator

PP_LIST = [
    corespp.MUTAGPreparator,
    corespp.ENZYMESPreparator,
    corespp.BA2MotifPreparator,
    corespp.BZRPreparator,
    corespp.COX2Preparator,
]


@pytest.mark.parametrize("Preparator", PP_LIST)
@pytest.mark.parametrize("seed", list(range(40)))
def test_preparator(Preparator: BaseDatasetPreparator, seed: int):
    kwargs = Preparator.random_kwargs(seed=seed)

    preparator = Preparator(**kwargs)

    preparator.prepare()

    classes_num = preparator.classes_num()
    assert isinstance(classes_num, int)
    assert classes_num >= 0

    features_dim = preparator.features_dim()
    assert isinstance(features_dim, int)
    assert features_dim >= 0
    target_tim = preparator.target_dim()
    assert isinstance(target_tim, int)
    assert target_tim >= 0

    dataset = preparator.get_dataset_train()
    assert isinstance(dataset, pygd.Dataset)

    data_train = preparator.get_train_data()

    assert isinstance(data_train, pygd.Batch)
    for dtype in [None, torch.float32, torch.float64, torch.int32, torch.int64]:
        y = preparator.get_target(data_train, dtype=dtype)
        assert isinstance(y, torch.Tensor)
    preparator.fit_preprocessor()

    loader_train = preparator.get_dataloader_train(batch_size=8)
    assert isinstance(loader_train, pygl.DataLoader), f"{loader_train}"

    loaders = preparator.get_dataloaders(batch_size=8)

    assert isinstance(loaders, dict)

    for key, loader in loaders.items():
        data = next(iter(loader))
        assert isinstance(data, pygd.Batch)
        assert data.x.shape[1] == preparator.features_dim()
        # assert data.y.ndim == 2
        # assert data.y.shape[1] == preparator.target_dim()


@pytest.mark.parametrize("Preparator", PP_LIST)
def test_eda(Preparator: BaseDatasetPreparator):
    kwargs = Preparator.random_kwargs(seed=0)

    preparator = Preparator(**kwargs)

    preparator.prepare()

    preparator.exploratory_data_analysis(root=os.path.join("results", "eda", preparator.name))

    plt.close("all")
