from __future__ import annotations

import itertools
import os
import random
from typing import List, Optional

from omegaconf import OmegaConf


class RandomSweep:

    def __init__(
        self,
        sweep_file_path: Optional[str] = None,
        param_files: Optional[str] = None,
        sweep_config: Optional[dict] = None,
    ):
        if sweep_file_path is not None:
            # Check if the sweep file exists
            if not os.path.exists(sweep_file_path):
                raise FileNotFoundError(f"The sweep file does not exist: {sweep_file_path}")
            sweep_config = OmegaConf.load(sweep_file_path)

            sweep_config = OmegaConf.to_object(sweep_config)
        elif sweep_config is not None:
            sweep_config = sweep_config

        if param_files is not None:
            for file_name_i in param_files:
                if not os.path.exists(file_name_i):
                    raise FileNotFoundError(f"The parameter file does not exist: {file_name_i}")
                sweep_config_i = OmegaConf.load(file_name_i)

                sweep_config_i = OmegaConf.to_object(sweep_config_i)

                for key, key_dict in sweep_config_i.items():
                    sweep_config["parameters"][key] = key_dict
        self.parameters = sweep_config["parameters"]

    def generate_all(self) -> List[str]:
        """
        Generate all possible configurations based on defined parameter distributions.

        Returns:
            list: A list of all possible configurations given as string param1=value1 param2=value2.
        """
        params_dict = {}
        for key, key_dict in self.parameters.items():
            distribution = key_dict["distribution"]
            if distribution == "constant":
                value = [key_dict["value"]]
            elif distribution == "categorical":
                value = key_dict["values"]
            # elif distribution == "uniform":
            #     value = random.uniform(key_dict["min"], key_dict["max"])
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            params_dict[key] = value

        param_values = [params_dict[key] for key in params_dict]

        combinations = list(itertools.product(*param_values))

        configs = []

        for combination in combinations:
            my_config_list = []
            for i, key in enumerate(params_dict):
                my_config_list.append(f"{key}={combination[i]}")
            my_config = " ".join(my_config_list)
            configs.append(my_config)

        return configs

    def sample(self, seed: int = -1, count: int = 1, start: int = 0) -> List[str]:
        """
        Generate unique configurations based on defined parameter distributions.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to -1.
            count (int, optional): Number of configurations to generate. Defaults to 1.
            start (int, optional): Starting index for configuration generation. Defaults to 0.

        Returns:
            list: A list of unique configurations given as string param1=value1 param2=value2.
        """

        if seed != -1:
            random.seed(seed)
        i = 0

        if count == 0:
            return self.generate_all()
        configs = []
        while i < (count + start):
            my_config_list = []
            for key, key_dict in self.parameters.items():
                distribution = key_dict["distribution"]
                if distribution == "constant":
                    value = key_dict["value"]
                elif distribution == "categorical":
                    value = random.choice(key_dict["values"])
                # elif distribution == "uniform":
                #     value = random.uniform(key_dict["min"], key_dict["max"])
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")

                if isinstance(value, list):
                    value = str(value)
                    value = f"'{value}'"
                my_config_list.append(f"{key}={value}")

            my_config = " ".join(my_config_list)

            if i < start:
                i += 1
            elif my_config not in configs:
                i += 1
                configs.append(my_config)

        return configs
