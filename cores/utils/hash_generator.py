from __future__ import annotations

import ast
import hashlib
from typing import Any


def save_eval(my_str):
    try:
        # Check if the string is a valid literal (e.g., integer, float)
        result = ast.literal_eval(my_str)
        return result
    except ValueError:
        return my_str


class HashGenerator:
    @staticmethod
    def from_dict(my_dict: dict[str, Any]) -> str:
        return hashlib.md5(str(my_dict).encode("utf-8")).hexdigest()
