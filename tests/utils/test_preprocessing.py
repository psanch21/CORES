from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import fetch_openml

from cores.utils.preprocessing import InvertiblePreprocessingDF, PreprocessingDF


# Session fixture
@pytest.fixture(scope="session")
def data():
    df_x, df_y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser="pandas")

    return df_x, df_y


@pytest.mark.parametrize("mode", ["pandas", "numpy"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", list(range(2)))
def test_preprocessing(mode: str, dtype, seed: int, data):
    df_x, df_y = data

    if mode == "pandas":
        columns = ["age", "fare"]
    else:
        # Get index of columns
        columns = [df_x.columns.get_loc(c) for c in ["age", "fare"]]
    pp_numeric = dict(
        transformer_list=[
            {"type": "imputer", "name": "simple", "kwargs": {"strategy": "median"}},
            {"type": "scaler", "name": "standard", "kwargs": {}},
        ],
        columns=columns,
    )

    if mode == "pandas":
        columns = ["embarked", "sex", "pclass"]
    else:
        # Get index of columns
        columns = [df_x.columns.get_loc(c) for c in ["embarked", "sex", "pclass"]]
    pp_cat = dict(
        transformer_list=[
            {"type": "encoder", "name": "onehot", "kwargs": {}},
        ],
        columns=columns,
    )

    preprocessing_dict = dict(numeric=pp_numeric, categorical=pp_cat)
    preprocessor = PreprocessingDF(preprocessing_dict=preprocessing_dict)

    if mode == "pandas":
        x = df_x
    else:
        x = df_x.to_numpy()

    preprocessor.fit(x)

    x_norm = preprocessor.transform(x, dtype=dtype)

    assert x_norm.dtype == dtype
    assert x_norm.shape[0] == df_x.shape[0]
    assert x_norm.shape[1] == 11


@pytest.mark.parametrize("mode", ["pandas", "numpy"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", list(range(2)))
def test_inv_preprocessing(mode: str, dtype, seed: int, data):
    df_x, df_y = data

    columns_all = []
    if mode == "pandas":
        columns = ["age", "fare"]
    else:
        # Get index of columns
        columns = [df_x.columns.get_loc(c) for c in ["age", "fare"]]
    pp_numeric = dict(
        transformer_list=[
            {"type": "scaler", "name": "standard", "kwargs": {}},
        ],
        columns=columns,
    )

    columns_all.extend(columns)

    if mode == "pandas":
        columns = ["embarked", "sex", "pclass"]
    else:
        # Get index of columns
        columns = [df_x.columns.get_loc(c) for c in ["embarked", "sex", "pclass"]]

    columns_all.extend(columns)
    pp_cat = dict(
        transformer_list=[
            # {"type": "encoder", "name": "label", "kwargs": {}},
            {"type": "encoder", "name": "onehot", "kwargs": {"sparse_output": False}},
        ],
        columns=columns,
    )

    preprocessing_dict = dict(numeric=pp_numeric, categorical=pp_cat)
    preprocessor = InvertiblePreprocessingDF(preprocessing_dict=preprocessing_dict)

    if mode == "pandas":
        x = df_x
    else:
        x = df_x.to_numpy()

    preprocessor.fit(x)

    x_norm = preprocessor.transform(x, dtype=dtype)
    assert x_norm.dtype == dtype
    assert x_norm.shape[0] == df_x.shape[0]
    assert x_norm.shape[1] == 11

    x_inv = preprocessor.inverse_transform(x_norm)

    if mode == "pandas":
        for column_i in columns_all:
            dtype_col = df_x[column_i].dtype

            for i in range(len(x_inv)):
                value_1 = x_inv[column_i].iloc[i]
                value_2 = df_x[column_i].iloc[i]
                # Check if dtype_col is numeric
                if dtype_col in [np.float32, np.float64]:
                    # Check if any of the values is nan
                    if np.isnan(value_1) or np.isnan(value_2):
                        assert np.isnan(value_1) and np.isnan(value_2)
                    else:
                        assert np.isclose(
                            value_1, value_2, atol=1e-4
                        ), f"[{i},{column_i}] {value_1} != {value_2}"
                else:
                    if value_1 != value_2:
                        assert np.isnan(value_1) and np.isnan(value_2)
                    else:
                        assert value_1 == value_2, f"[{i},{column_i}] {value_1} != {value_2}"
