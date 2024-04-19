from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.preprocessing as sklpp
from sklearn.pipeline import Pipeline


class Scalers:
    STANDARD = "standard"
    MINMAX = "minmax"
    IDENTITY = "identity"


class Imputers:
    SIMPLE = "simple"


class Encoders:
    ONEHOT = "onehot"
    LABEL = "label"


class InvertiblePreprocessingDF:
    def __init__(
        self,
        preprocessing_dict: Dict[str, Dict[str, Any]],
    ):
        self.preprocessing_dict = preprocessing_dict

        self.preprocessor = None

        self.dtype = None

        self.columns = None
        self.columns_norm = None

        self.features_dim = None
        self.features_norm_dim = None

    def get_encoder(self, name: str, **kwargs):
        if name == Encoders.ONEHOT:
            encoder = sklpp.OneHotEncoder(**kwargs)
        elif name == Encoders.LABEL:
            encoder = sklpp.LabelEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder: {name}")

        return encoder

    def get_scaler(self, name: str, **kwargs):
        if name == Scalers.STANDARD:
            scaler = sklpp.StandardScaler()
        elif name == Scalers.MINMAX:
            scaler = sklpp.MinMaxScaler(**kwargs)
        elif name == Scalers.IDENTITY:
            scaler = sklpp.FunctionTransformer(
                func=lambda x: x, inverse_func=lambda x: x, validate=True
            )
        else:
            raise ValueError(f"Unknown scaler: {name}")

        return scaler

    def _build_pipeline(self, transforer_list: List[Dict[str, Any]]) -> Pipeline:
        steps = []
        for tx_dict in transforer_list:
            tx_type = tx_dict["type"]
            tx_name = tx_dict["name"]
            kwargs = tx_dict["kwargs"]
            if tx_type == "imputer":
                imputer = self.get_imputer(tx_name, **kwargs)
                steps.append((tx_name, imputer))
            elif tx_type == "scaler":
                scaler = self.get_scaler(tx_name, **kwargs)
                steps.append((tx_name, scaler))
            elif tx_type == "encoder":
                encoder = self.get_encoder(tx_name, **kwargs)
                steps.append((tx_name, encoder))
            else:
                raise ValueError(f"Unknown transformer type: {tx_type}")
        pipeline = Pipeline(steps=steps)

        return pipeline

    def fit(self, x: pd.DataFrame | npt.NDArray) -> None:
        self.dtype = type(x)

        self.preprocessor = {}
        columns = []

        for name_i, data in self.preprocessing_dict.items():
            transformer_list = data["transformer_list"]
            columns_i = data["columns"]
            if isinstance(columns_i, str):
                assert columns_i == "all"
                if isinstance(x, pd.DataFrame):
                    columns_i = list(x.columns)
                elif isinstance(x, np.ndarray):
                    columns_i = list(range(x.shape[1]))
                else:
                    raise ValueError(f"Unknown type: {type(x)}")

                assert len(self.preprocessing_dict) == 1

            # Assert none of columns_i are in columns
            for column in columns_i:
                assert column not in columns

            columns.extend(columns_i)

            pipeline_i = self._build_pipeline(transformer_list)
            preprocessor_i = dict(
                pipeline=pipeline_i,
                columns=columns_i,
            )

            self.preprocessor[name_i] = preprocessor_i

        for name, pp_data in self.preprocessor.items():
            pipeline_i = pp_data["pipeline"]
            columns_i = pp_data["columns"]
            if isinstance(x, pd.DataFrame):
                pipeline_i.fit(x[columns_i])
            elif isinstance(x, np.ndarray):
                pipeline_i.fit(x[:, columns_i])
            else:
                raise ValueError(f"Unknown type: {type(x)}")

        self.columns = columns

        self.features_dim = len(columns)

    def fit_transform(
        self, x: pd.DataFrame | npt.NDArray, dtype=np.float32
    ) -> npt.NDArray[np.float32]:
        self.fit(x)
        return self.transform(x, dtype=dtype)

    def transform(self, x: pd.DataFrame | npt.NDArray, dtype=np.float32) -> npt.NDArray[np.float32]:
        x_norm = []
        columns_norm = {}
        idx = 0

        for name, pp_data in self.preprocessor.items():
            pipeline_i = pp_data["pipeline"]
            columns_i = pp_data["columns"]
            if isinstance(x, pd.DataFrame):
                x_norm_i = pipeline_i.transform(x[columns_i])
            else:
                x_norm_i = pipeline_i.transform(x[:, columns_i])

            if x_norm_i.ndim == 1:
                x_norm_i = x_norm_i.reshape(-1, 1)

            columns_norm[name] = [i + idx for i in range(x_norm_i.shape[1])]

            idx += x_norm_i.shape[1]

            x_norm.append(x_norm_i)

        x_norm = np.concatenate(x_norm, axis=1)

        self.columns_norm = columns_norm
        self.features_norm_dim = x_norm.shape[1]
        return x_norm.astype(dtype)

    def inverse_transform(self, x_norm: npt.NDArray) -> pd.DataFrame | npt.NDArray:
        assert self.preprocessor is not None
        x = []
        for name, pp_data in self.preprocessor.items():
            pipeline_i = pp_data["pipeline"]
            columns_i = self.columns_norm[name]
            x_i = pipeline_i.inverse_transform(x_norm[:, columns_i])
            x.append(x_i)
        x = np.concatenate(x, axis=1)

        if self.dtype == np.ndarray:
            col_idx = np.argsort(self.columns)
            return x[:, col_idx]
        else:
            df = pd.DataFrame(x, columns=self.columns)
            return df
