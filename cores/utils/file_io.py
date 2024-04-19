from __future__ import annotations

import ast
import logging
import os
from typing import Any

import yaml


def save_eval(my_str):
    try:
        # Check if the string is a valid literal (e.g., integer, float)
        result = ast.literal_esval(my_str)
        return result
    except ValueError:
        return my_str


class FileIO:
    @staticmethod
    def makedirs(file_path: str) -> None:
        os.makedirs(file_path, exist_ok=True)
        return

    @staticmethod
    def load_yaml(file_path: str) -> dict[str, Any]:
        with open(file_path, "r") as f:
            my_dict = yaml.safe_load(f)
        return my_dict

    @staticmethod
    def flatten_dict(d, parent_key="", delimiter="."):
        flattened = {}
        for key, value in d.items():
            new_key = f"{parent_key}{delimiter}{key}" if parent_key else key

            if isinstance(value, dict):
                flattened.update(FileIO.flatten_dict(value, new_key, delimiter))
            else:
                flattened[new_key] = value

        return flattened

    @staticmethod
    def dict_to_txt(my_dict: dict[str, Any], file_path: str) -> str:
        with open(file_path, "w") as f:
            for key, value in my_dict.items():
                f.write(f"{key}\n{value}\n\n")
        return file_path

    # @staticmethod
    # def flatten_dict(my_dict: dict) -> dict:
    #     dict_out = {}
    #     for key1, value1 in my_dict.items():
    #         assert "__" not in key1
    #         if isinstance(value1, dict):
    #             for key2, value2 in value1.items():
    #                 assert "__" not in key2
    #                 dict_out[f"{key1}__{key2}"] = value2
    #         else:
    #             dict_out[f"{key1}"] = value1

    #     return dict_out

    @staticmethod
    def string_to_txt(my_str: str, file_path: str, mode: str = "w") -> str:
        if my_str[-1] != "\n":
            my_str += "\n"
        with open(file_path, mode) as f:
            f.write(my_str)

        logging.info(f"Saved to {file_path}")
        return file_path

    @staticmethod
    def txt_to_list(file_path: str, literal: bool = True) -> list[str]:
        with open(file_path, "r") as f:
            lines = f.readlines()

        output = []
        for x in lines:
            if literal:
                x = x.replace("nan", '"nan"')
                output.append(ast.literal_eval(x.strip()))
            else:
                output.append(x.strip())
        return output

    @staticmethod
    def dict_to_yaml(my_dict: dict[str, Any], file_path: str) -> str:
        with open(file_path, "w") as f:
            yaml.dump(my_dict, f)
        return file_path

    @staticmethod
    def txt_to_dict(file_path: str, tab="---") -> dict[str, Any]:
        my_dict = {}

        with open(file_path, "r") as f:
            lines = f.readlines()
            key = None
            for line in lines:
                if line == "\n":
                    key = None
                    continue

                line = line.strip()
                if key is None:
                    key = line
                    my_dict[key] = []
                    continue

                line_split = line.split(tab)
                if len(line_split) == 0:
                    continue

                if len(line_split) == 1:
                    line_split = save_eval(line_split[0].strip())
                else:
                    line_split = [save_eval(x.strip()) for x in line_split]
                my_dict[key].append(line_split)

        for key in list(my_dict.keys()):
            if len(my_dict[key]) == 1:
                my_dict[key] = my_dict[key][0]
        return my_dict
