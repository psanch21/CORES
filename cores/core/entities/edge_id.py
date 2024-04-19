from __future__ import annotations

from cores.core.entities.node_id import NodeID


class EdgeID:
    def __init__(self, head: NodeID, tail: NodeID):
        self.head = head
        self.tail = tail

    def __eq__(self, other: EdgeID) -> bool:
        return self.head == other.head and self.tail == other.tail

    def __contains__(self, node_id: NodeID) -> bool:
        return self.head == node_id or self.tail == node_id

    def __str__(self) -> str:
        class_name = self.__class__.__name__

        return f"{class_name}({self.head}, {self.tail})"

    def to_tuple(self) -> tuple[int | str, int | str]:
        return (self.head.value, self.tail.value)

    def get_other_node(self, node_id: NodeID) -> NodeID:
        if node_id == self.head:
            return self.tail
        elif node_id == self.tail:
            return self.head
        else:
            raise ValueError(f"Node {node_id} is not in edge {self}")
