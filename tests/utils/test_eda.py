from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch_geometric.loader as pygl
from sklearn.datasets import fetch_openml
from torch_geometric.datasets import TUDataset

import cores.utils.eda as geda
import cores.utils.plotter as gplotter


# Session fixture
@pytest.fixture(scope="session")
def data():
    df_x, df_y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser="pandas")

    return df_x, df_y


# Session fixture
@pytest.fixture(scope="session")
def dataset_mutag():
    root = os.path.join("..", "data", "graph")
    name = "MUTAG"

    def transform(data):
        data.node_color = data.x.argmax(dim=-1).unsqueeze(dim=-1)
        data.y = data.y.unsqueeze(dim=-1)
        return data

    dataset = TUDataset(
        root=root,
        name=name,
        transform=transform,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=True,
        use_edge_attr=True,
        cleaned=True,
    )
    return dataset


def test_eda_dataframe(data):
    df_x, df_y = data

    root = os.path.join("tests", "utils", "eda")

    if not os.path.exists(root):
        os.makedirs(root)
    eda = geda.EDADataFrame(root=os.path.join("tests", "utils", "eda"))

    # Merge df_y into df_x

    df = pd.concat([df_x, df_y], axis=1)

    eda.report(df, title="Titanic Report", output_file="titanic_report.html")


@pytest.mark.parametrize("data_idx", list(range(10)))
def test_eda_graph(data_idx: int, dataset_mutag):
    root = os.path.join("tests", "utils", "eda")

    if not os.path.exists(root):
        os.makedirs(root)

    plotter = gplotter.MatplotlibPlotter()

    dataset = dataset_mutag
    eda = geda.EDAGraph(root=os.path.join("tests", "utils", "eda"), plotter=plotter)

    data = dataset[data_idx]
    data.node_color = data.x.argmax(dim=-1).unsqueeze(dim=-1)

    eda.report(graph=data, title="MUTAG Report", output_file=f"mutag_report_{data_idx}.html")


@pytest.mark.parametrize("seed", list(range(1)))
def test_eda_graph_batch(seed: int, dataset_mutag):
    root = os.path.join("tests", "utils", "eda")

    if not os.path.exists(root):
        os.makedirs(root)

    plotter = gplotter.MatplotlibPlotter()

    dataset = dataset_mutag
    eda = geda.EDAGraph(root=os.path.join("tests", "utils", "eda"), plotter=plotter)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = pygl.DataLoader(dataset, batch_size=10, shuffle=True)

    batch = next(iter(loader))

    eda.report_batch(
        batch=batch, title="MUTAG Batch Report", output_file=f"mutag_report_batch_{seed}.html"
    )
