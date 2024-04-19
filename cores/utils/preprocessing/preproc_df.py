from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.preprocessing as sklpp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Scalers:
    STANDARD = "standard"
    MINMAX = "minmax"


class Imputers:
    SIMPLE = "simple"


class Encoders:
    ONEHOT = "onehot"


class PreprocessingDF:
    def __init__(
        self,
        preprocessing_dict: Dict[str, Dict[str, Any]],
        preprocessor: Optional[ColumnTransformer] = None,
    ):
        self.preprocessing_dict = preprocessing_dict

        self.preprocessor = preprocessor

        self.features_dim = None
        self.features_norm_dim = None

    def get_encoder(self, name: str, **kwargs):
        if name == Encoders.ONEHOT:
            encoder = sklpp.OneHotEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder: {name}")

        return encoder

    def get_scaler(self, name: str, **kwargs):
        if name == Scalers.STANDARD:
            scaler = sklpp.StandardScaler()
        elif name == Scalers.MINMAX:
            scaler = sklpp.MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unknown scaler: {name}")

        return scaler

    def get_imputer(self, name: str, **kwargs):
        if name == "simple":
            imputer = SimpleImputer(**kwargs)
        else:
            raise ValueError(f"Unknown imputer: {name}")
        return imputer

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
        columns = []
        transformers = []

        for name_i, data in self.preprocessing_dict.items():
            transformer_list = data["transformer_list"]
            columns_i = data["columns"]
            if isinstance(columns_i, str):
                assert columns_i == "all"
                if isinstance(x, pd.DataFrame):
                    columns_i = list(x.columns)
                else:
                    columns_i = list(range(x.shape[1]))

                assert len(self.preprocessing_dict) == 1

            # Assert none of columns_i are in columns
            for column in columns_i:
                assert column not in columns

            columns.extend(columns_i)

            pipeline_i = self._build_pipeline(transformer_list)

            transformers.append((name_i, pipeline_i, columns_i))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        self.preprocessor.fit(x)

        self.features_dim = len(columns)

    def fit_transform(
        self, x: pd.DataFrame | npt.NDArray, dtype=np.float32
    ) -> npt.NDArray[np.float32]:
        self.fit(x)
        return self.transform(x, dtype=dtype)

    def transform(self, x: pd.DataFrame | npt.NDArray, dtype=np.float32) -> npt.NDArray[np.float32]:
        assert self.preprocessor is not None
        x_norm = self.preprocessor.transform(x).astype(dtype)

        self.features_norm_dim = x_norm.shape[1]
        return x_norm
