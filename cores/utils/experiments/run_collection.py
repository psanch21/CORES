from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cores.utils.experiments.run import Run
from cores.utils.file_io import FileIO


class RunCollection:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder

    def load_df_runs(self) -> pd.DataFrame:
        file_path = os.path.join(self.root_folder, "df_runs.pkl")
        df_runs = pd.read_pickle(file_path)
        logging.info(f"Loaded df_runs from {file_path}")
        return df_runs

    def get_df_runs(
        self, metrics_list: Optional[List[str]] = None, cache: bool = True
    ) -> pd.DataFrame:
        file_path = os.path.join(self.root_folder, "df_runs.pkl")

        if cache and os.path.isfile(file_path):
            df_runs = pd.read_pickle(file_path)
            logging.info(f"Loaded df_runs from {file_path}")
            return df_runs

        config_files = glob.glob(
            os.path.join(self.root_folder, "**", "config.yaml"), recursive=True
        )

        num_folders = len(config_files)

        logging.info(f"Number of folder runs: {num_folders}")

        df_runs_data = []
        for i, config_file in enumerate(config_files):
            if i % 50 == 0:
                print(f"Processing run {i}/{len(config_files)}")

            run_folder = os.path.dirname(config_file)
            run = Run(run_folder)
            try:
                df_config = run.get_config()
                df_metrics = run.get_last(metrics_list)
                df_run = pd.concat([df_config, df_metrics], axis=1)
                df_runs_data.append(df_run)
            except:
                continue

        df_runs = pd.concat(df_runs_data, axis=0, ignore_index=True)

        # Remove columns whose values are all nan
        df_runs = df_runs.dropna(axis=1, how="all")

        for col in df_runs.columns:
            if df_runs[col].dtype == object:
                df_runs[col] = df_runs[col].astype(str)

        df_runs.to_pickle(file_path)

        num_runs = len(df_runs)
        ratio = num_runs / num_folders * 100

        logging.info(f"Saved df_runs to {file_path} ({ratio:.2f}%)")

        return df_runs

    def save_best_config(
        self,
        df_runs: pd.DataFrame,
        df_best: pd.DataFrame,
        folder: str,
        groupby_cols: List[str],
        exclude_columns: List[str] = None,
        columns_to_int: List[str] = None,
    ) -> None:
        config_best = []
        config_runs = []
        if columns_to_int is None:
            columns_to_int = ["graph_clf_kwargs.heads"]
        for c in df_best.columns:
            if c[1] == "":
                config_best.append(c)
                config_runs.append(c[0])

        if exclude_columns is None:
            exclude_columns = ["data_kwargs.k_fold", "root_folder"]

        if not os.path.exists(folder):
            os.makedirs(folder)

        for idx, top_config in df_best[config_best].iterrows():
            df_best_i = df_runs[config_runs]
            for k, v in top_config.items():
                if pd.isna(v):
                    continue
                else:
                    df_best_i = df_best_i[df_best_i[k[0]] == v]
            for c in df_best_i.columns:
                if c in exclude_columns:
                    continue

                if len(df_best_i[c].unique()) > 1:
                    assert False, f"{c}: {df_best_i[c].unique()}"

            # Drop nan columns
            series_best_i = df_best_i.dropna(axis=1, how="all").iloc[0]
            group_id_str = "_".join(series_best_i[groupby_cols].astype(str).values)

            my_str = ""
            for key, value in series_best_i.items():
                if key in exclude_columns:
                    continue
                if pd.isna(value) or value == "nan":
                    continue

                if key in columns_to_int:
                    value = int(value)

                my_str += f"{key}: {value}\n"
            file_path = os.path.join(folder, f"{group_id_str}.yaml")

            FileIO.string_to_txt(my_str, file_path)

    def get_df_best(
        self,
        df_runs: pd.DataFrame,
        groupby_cols: List[str] = None,
        exclude_hyperparams: List[str] = None,
        metrics: Dict[str, List[str]] = None,
        metric_objective: str | Tuple[str, str] = None,
        mode: str = "maximize",
        best_idx: int = 0,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        # Convert dtype of columns with list to string
        if groupby_cols is None:
            groupby_cols = ["data", "graph_clf"]

        if exclude_hyperparams is None:
            exclude_hyperparams = ["data_kwargs.k_fold"]

        if metrics is None:
            metrics = {
                "testing_test_end/1__accuracy": ["mean", "std", "count"],
                "testing_valid_end/1__accuracy": ["mean", "std", "count"],
                "testing_test_end/1__num_nodes_ratio": ["mean", "std"],
            }

        if metric_objective is None:
            metric_objective = ("testing_valid_end/1__accuracy", "mean")

        ascending = False if mode == "maximize" else True

        df_best_data = []

        for group_id, df_g in df_runs.groupby(groupby_cols):
            print(f"\n{group_id}: {len(df_g)}")
            key_list = self.get_hyperparameter_keys(df_g)

            # Remove element form list data_kwargs.k_fold
            for key in exclude_hyperparams:
                if key in key_list:
                    key_list.remove(key)

            df_hp = df_g.groupby(key_list).agg(metrics)

            if count is not None:
                col_count = None
                for col_tmp in df_hp.columns:
                    if col_tmp[1] == "count":
                        col_count = col_tmp
                        break

                if col_count is None:
                    raise ValueError("count is not found")

                cond = df_hp[col_count] >= count
                df_hp = df_hp.loc[cond]

            # Short by mean accuracy
            df_hp_all = df_hp.sort_values(metric_objective, ascending=ascending)
            try:
                df_hp = df_hp_all.iloc[[best_idx]].reset_index()
            except:
                logging.warning(f"best_idx={best_idx} is not found {df_hp_all.shape}")
                continue

            for name_id, name in enumerate(groupby_cols):
                df_hp[name] = group_id[name_id]

            # scatter_performance_vs_sparsity(df_hp_all, color_columns=reward_cols)
            # scatter_reward_params(df_hp_all, color_columns=perf_spar_cols)

            df_best_data.append(df_hp)

        df_best = pd.concat(df_best_data, axis=0, ignore_index=True)

        return df_best

    def get_df_corr(self, df_runs: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "reward_kwargs.desired_ratio",
            "reward_kwargs.lambda_1",
            "testing_valid_end/1__accuracy",
            "testing_valid_end/1__num_nodes_ratio",
            "testing_valid_end/1__num_edges_ratio",
        ]

        df_corr_data = []
        for group_id, df_g in df_runs.groupby(["data", "graph_clf"]):
            print(f"\n{group_id}: {len(df_g)}")

            df_corr = df_g[cols].corr()[2:][
                ["reward_kwargs.desired_ratio", "reward_kwargs.lambda_1"]
            ]

            # Convert a column with each cell of the df
            df_corr = df_corr.stack().reset_index().rename(columns={0: "correlation"})

            df_corr["data"] = group_id[0]
            df_corr["graph_clf"] = group_id[1]

            df_corr_data.append(df_corr)
        df_corr = pd.concat(df_corr_data, axis=0, ignore_index=True)
        df_corr["pair"] = df_corr["level_0"] + " vs " + df_corr["level_1"]
        df_corr = df_corr.drop(columns=["level_0", "level_1"])

        return df_corr

    def get_hyperparameter_keys(self, df_runs: pd.DataFrame) -> list:
        # Get unique values for each column
        key_list = []
        for col in df_runs.columns:
            if "." not in col:
                continue

            unique_num = len(df_runs[col].unique())

            if unique_num > 1 and unique_num < len(df_runs) * 0.9:
                # print(f"{col}: {df_runs[col].unique()}")
                key_list.append(col)
        return key_list
