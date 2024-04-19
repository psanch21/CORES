from __future__ import annotations

import numpy as np
import pandas as pd


class LaTeXUtils:
    @staticmethod
    def table_comparison(
        df: pd.DataFrame,
        metrics_list: list[str] = None,
        datasets_list: list[str] = None,
        models_list: list[str] = None,
        best_in_bold: bool = True,
        ranking: bool = True,
    ) -> str:
        """
        Generate LaTeX table code from the given dataframe.

        Args:
        df (pandas.DataFrame): The dataframe to convert into LaTeX table format.

        Returns:
        str: The LaTeX table code as a string.
        """
        # Replace _ by - in the values of the data column

        if models_list is None:
            models_list = df["model"].unique()
        num_models = len(models_list)

        if datasets_list is None:
            datasets_list = df["data"].unique()

        if metrics_list is None:
            metrics_list = ["Accuracy", "Node Ratio", "Edge Ratio"]

        num_metrics = len(metrics_list)

        # Start building the LaTeX table string

        if num_metrics == 1:
            latex_str = "\\begin{tabular}{l" + "c" * num_models + "}\n"
        else:
            latex_str = "\\begin{tabular}{lc" + "c" * num_models + "}\n"
        latex_str += "\\toprule\n"
        if num_metrics == 1:
            latex_str += (
                "\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{"
                + str(num_models)
                + "}{c}{\\textbf{Models}} \\\\ \n"
            )
            latex_str += "\\cmidrule(lr){2-" + str(2 + num_models - 1) + "}\n"
        else:
            latex_str += (
                "\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Metric}} & \\multicolumn{"
                + str(num_models)
                + "}{c}{\\textbf{Models}} \\\\ \n"
            )
            latex_str += "\\cmidrule(lr){3-" + str(3 + num_models - 1) + "}\n"

        latex_str += "&"
        for model in models_list:
            latex_str += " & \\textbf{" + model + "}"
        latex_str += "\\\\ \n"
        latex_str += "\\midrule\n"

        # The loop to fill in the data is needed here, following the format specified
        for data in datasets_list:
            df_d = df[df["data"] == data]
            for metric_id, metric in enumerate(metrics_list):
                row_str = ""
                if metric_id == 0:
                    row_str += "\\multirow{" + str(num_metrics) + "}{*}{" + data + "}"

                if num_metrics > 1:
                    row_str += " & \\textbf{" + metric + "}"

                for model in models_list:
                    df_m = df_d[df_d["model"] == model]
                    # print(f"{data} {metric} {model}")
                    if df_m.shape[0] == 0:
                        row_str += " & -"
                    else:
                        metric_mean = df_m[(metric, "mean")].values[0]
                        metric_std = df_m[(metric, "std")].values[0]

                        rank = df_m[(metric, "rank")].values[0]

                        # check if metric_mean is nan
                        if np.isnan(metric_mean):
                            row_str += " & -"
                        else:
                            # metric_str = f"{metric_mean:.2f}" + " $\\pm$ " + f"{metric_std:.2f}"
                            metric_str = f"{metric_mean:.2f}$_{{{metric_std:.2f}}}$"
                            if rank == 1 and best_in_bold:
                                row_str += " & \\bfseries " + f"{metric_str}"
                            else:
                                row_str += " & " + f"{metric_str}"
                row_str += "\\\\ \n"
                latex_str += row_str

            # if last data
            if data == datasets_list[-1]:
                latex_str += "\\bottomrule\n"
            else:
                if num_metrics == 1:
                    latex_str += "\\cmidrule(lr){1-" + str(1 + num_models) + "}\n"
                else:
                    latex_str += "\\cmidrule(lr){2-" + str(2 + num_models) + "}\n"

        # Add average rank

        if ranking:
            assert num_metrics > 1
            for metric_id, metric in enumerate(metrics_list):
                if metric_id == 0:
                    latex_str += "\\multirow{" + str(num_metrics) + "}{*}{\\textbf{Avg. Rank}}"
                latex_str += " & \\textbf{" + metric + "}"
                for model_id, model in enumerate(models_list):
                    df_m = df[df["model"] == model]
                    rank_avg = df_m[(metric, "rank")].mean()
                    if pd.isna(rank_avg):
                        latex_str += " & -"
                    else:
                        latex_str += " & " + f"{rank_avg:.2f}"
                latex_str += "\\\\ \n"
                if metric == metrics_list[-1]:
                    latex_str += "\\bottomrule\n"
                else:
                    latex_str += "\cmidrule(lr){2-" + str(2 + num_models) + "}\n"
        latex_str += "\\end{tabular}"

        return latex_str

    @staticmethod
    def table_config(
        df: pd.DataFrame,
        archi_list: list[str] = None,
        datasets_list: list[str] = None,
        params_list: list[str] = None,
        param_mapping: dict[str, str] = None,
    ) -> str:
        """
        Generate LaTeX table code from the given dataframe.

        Args:
        df (pandas.DataFrame): The dataframe to convert into LaTeX table format.

        Returns:
        str: The LaTeX table code as a string.
        """

        if datasets_list is None:
            datasets_list = df["data"].unique()
        num_datasets = len(datasets_list)

        if archi_list is None:
            archi_list = df["graph_clf"].unique()
        num_archi = len(archi_list)

        if params_list is None:
            params_list = [c for c in df.columns if "." in c]

        # Start building the LaTeX table string

        latex_str = "\\begin{tabular}{lc" + "c" * num_datasets + "}\n"
        latex_str += "\\toprule\n"
        latex_str += (
            "\\multirow{2}{*}{\\textbf{Param}} & \\multirow{2}{*}{\\textbf{Archi}} & \\multicolumn{"
            + str(num_datasets)
            + "}{c}{\\textbf{Datasets}} \\\\ \n"
        )
        latex_str += "\\cmidrule(lr){3-" + str(3 + num_datasets - 1) + "}\n"

        latex_str += "&"
        for data_i in datasets_list:
            latex_str += " & \\textbf{" + data_i + "}"
        latex_str += "\\\\ \n"
        latex_str += "\\midrule\n"

        # The loop to fill in the data is needed here, following the format specified

        caption = ""
        for i, param_i in enumerate(params_list):
            if param_mapping is not None:
                caption += f", {param_mapping[param_i]} (" + r"$\#$" + f"{i})"
            else:
                caption += f", {param_i} (" + r"$\#$" + f"{i})"
            for archi_id, archi in enumerate(archi_list):
                row_str = ""
                if archi_id == 0:
                    row_str += "\\multirow{" + str(num_archi) + "}{*}{" + r" $\#$" + f"{i}" + "}"
                row_str += " & \\textbf{" + archi + "}"

                for data in datasets_list:
                    df_d = df[df["data"] == data]
                    row = df_d[df_d["graph_clf"] == archi]

                    # print(f"{data} {metric} {model}")
                    if row.shape[0] == 0:
                        row_str += " & -"
                    else:
                        assert row.shape[0] == 1
                        row = row.iloc[0]
                        value = row[param_i]
                        if isinstance(value, float) and pd.isna(value):
                            value = "-"
                        elif isinstance(value, str):
                            print()
                            if pd.isna(value) or value == "nan":
                                value = "-"
                            else:
                                value = value[-5:]
                        row_str += " & " + f"{value}"
                row_str += "\\\\ \n"
                latex_str += row_str

            # if last data
            if param_i == params_list[-1]:
                latex_str += "\\bottomrule\n"
            else:
                latex_str += "\\cmidrule(lr){2-" + str(2 + num_datasets) + "}\n"

        latex_str += "\\end{tabular}\n"
        latex_str += f"\\caption{{{caption}}}\n"

        return latex_str
