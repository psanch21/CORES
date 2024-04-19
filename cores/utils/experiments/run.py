from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

from cores.utils.file_io import FileIO


class Run:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        config_file = os.path.join(root_folder, "config.yaml")
        config = FileIO.load_yaml(config_file)
        self.config = FileIO.flatten_dict(config)

        self.ckpt_files = [
            os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith(".ckpt")
        ]
        self.metric_folders = [
            os.path.join(root_folder, f)
            for f in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, f))
        ]

    def metrics_list(self, regex: Optional[str] = None) -> list:
        for metric_folder in self.metric_folders:
            base_name = os.path.basename(metric_folder)
            for metric_file in os.listdir(metric_folder):
                metric_name = os.path.join(base_name, metric_file)
                if regex is not None:
                    if re.search(regex, metric_name) is not None:
                        yield metric_name

    def get_config(self) -> pd.DataFrame:
        return pd.DataFrame([self.config])

    def get_last(self, keys: list) -> pd.DataFrame:
        data = []
        for key in keys:
            df_key = self.get(key).iloc[[-1]].reset_index(drop=True)
            data.append(df_key)

        # Concatenate all dataframes along axis 1 considering the index
        df = pd.concat(data, axis=1)

        df = df.reset_index(drop=True)
        return df

    def get_many(self, keys: list, index: str = "step", reset_index=True) -> pd.DataFrame:
        data = []
        for key in keys:
            df_key = self.get(key, index=index)
            data.append(df_key)

        # Concatenate all dataframes along axis 1 considering the index
        df = pd.concat(data, axis=1)

        if reset_index:
            df = df.reset_index()
            return df
        else:
            return df

    def get(self, key: str, index: str = "step") -> pd.DataFrame:
        metric_file_no_ext = os.path.join(self.root_folder, key)

        ext_list = [".txt", ".csv", ".pkl", ".json"]

        for ext in ext_list:
            metric_file_path = metric_file_no_ext + ext
            if os.path.isfile(metric_file_path):
                if ext == ".txt":
                    data = FileIO.txt_to_list(metric_file_path)
                    df = pd.DataFrame(data)
                    if index is not None:
                        df = df.set_index(index)

                    return df

                elif ext == ".csv":
                    raise NotImplementedError
                elif ext == ".pkl":
                    raise NotImplementedError
                elif ext == ".json":
                    raise NotImplementedError

    def __str__(self):
        my_str = f"Run: {self.root_folder}\n"

        my_str += "\n\tCONFIG\n"

        for key, value in self.config.items():
            if "." not in key:
                my_str += f"\t{key}: {value}\n"

        my_str += "\n\tMETRICS\n"
        for folder in self.metric_folders:
            my_str += f"\t{folder}\n"

        my_str += "\n\tCHECKPOINTS\n"
        for ckpt in self.ckpt_files:
            my_str += f"\t{ckpt}\n"

        return my_str
