from __future__ import annotations


class NodeID:
    def __init__(self, value: int | str):
        self.value = value

    def __eq__(self, other: NodeID | None) -> bool:
        if other is None:
            return False
        else:
            return self.value == other.value

    def __contains__(self, item: int | str | NodeID) -> bool:
        if isinstance(item, str):
            return item in self.value
        elif isinstance(item, NodeID):
            return item == self
        else:
            return item == self.value

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.value})"

    def to_int(self) -> int:
        return int(self.value)
