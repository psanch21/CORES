import os
import random
import shutil

import numpy as np
import pytest

from cores.impl.logger import DummyLogger, FSLogger, ListLogger

TYPES_LIST = ["int", "float", "str", "bool", "list"]


def rm_if_exists(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def create_value(mode, seed):
    np.random.seed(seed)
    if mode == "int":
        value = np.random.randint(1, 10)
    elif mode == "float":
        value = np.random.rand() * 10
    elif mode == "str":
        value = random.choice(["my_str_1", "my_str_2"])
    elif mode == "bool":
        value = random.choice([True, False])
    elif mode == "list":
        value = [1, 2, 3]
    elif mode == "dict":
        value = {"a": 1, "b": 2}
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return value


def create_data(num_elements, seed):
    type_list = ["int", "float", "str", "bool"]
    data = {}
    for i in range(num_elements):
        type_ = random.choice(type_list)
        data[f"elem_{i}"] = create_value(type_, seed)

    return data


def create_logger(seed):
    np.random.seed(seed)
    logger_classes = [FSLogger, DummyLogger]
    LoggerClass = random.choice(logger_classes)

    if LoggerClass == FSLogger:
        folder = os.path.join("tests", "logs")
        logger = FSLogger(config=dict(a=1, b=2), folder=folder)
    elif LoggerClass == DummyLogger:
        logger = DummyLogger()
    else:
        raise ValueError(f"Wrong logger class {LoggerClass}")

    return logger


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("mode", TYPES_LIST)
@pytest.mark.parametrize("step", [5000, None])
@pytest.mark.parametrize("seed", list(range(2)))
def test_fs_logger_value(device: str, mode: str, step: int, seed: int):
    folder = os.path.join("tests", "logs")

    rm_if_exists(folder)
    logger = FSLogger(config=dict(a=1, b=2), folder=folder)

    value = create_value(mode, seed)

    key = f"{mode}_key"

    logger.track_value(key, value, step=step)

    file_path = os.path.join(folder, f"{key}.txt")

    assert os.path.exists(file_path), f"file_path: {file_path}"

    with open(file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = eval(lines[0])
        assert isinstance(data, dict)
        if step is None:
            assert data["step"] == 0
        else:
            assert data["step"] == step

        assert data[key] == value

    # Remove logger.file_path
    os.remove(file_path)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("num_elements", list(range(10)))
@pytest.mark.parametrize("step", [5000, None])
@pytest.mark.parametrize("seed", list(range(2)))
def test_fs_logger_data(device: str, num_elements: int, step: int, seed: int):
    folder = os.path.join("tests", "logs")
    rm_if_exists(folder)
    logger = FSLogger(config=dict(a=1, b=2), folder=folder)

    data = create_data(num_elements=num_elements, seed=seed)

    logger.track_data(data, step=step)

    # Assert number of txt files is num_elements
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    assert len(txt_files) == num_elements

    for key in data.keys():

        file_path = os.path.join(folder, f"{key}.txt")

        assert os.path.exists(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = eval(lines[0])
            assert isinstance(data, dict)
            if step is None:
                assert data["step"] == 0
            else:
                assert data["step"] == step

            del data["step"]

        assert len(data) == 1


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("num_loggers", list(range(10)))
@pytest.mark.parametrize("step", [5000, None])
@pytest.mark.parametrize("seed", list(range(2)))
def test_list_logger(device: str, num_loggers: int, step: int, seed: int):
    logger_list = []

    folder = os.path.join("tests", "logs")
    rm_if_exists(folder)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    # Remove if folder exists
    for i in range(num_loggers):
        logger_i = create_logger(seed)
        logger_list.append(logger_i)

    logger = ListLogger(logger_list=logger_list)

    mode = random.choice(TYPES_LIST)
    value = create_value(mode, seed)

    key = f"{mode}_key"

    logger.track_value(key, value, step=step)

    num_elements = random.choice(list(range(1, 10)))
    data = create_data(num_elements=num_elements, seed=seed)
    logger.track_data(data, step)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("mode", TYPES_LIST)
@pytest.mark.parametrize("seed", list(range(2)))
def test_dummy_logger(device: str, mode: str, seed: int):
    logger = DummyLogger()

    value = create_value(mode, seed)

    key = f"{mode}_key"

    logger.track_value(key, value)
