from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Node:
    """A node in the workflow graph.

    Notes:
        CE-Graph treats workflows as *editable* DAGs. To keep this repo
        lightweight, we store a node's semantics as a (type, config) pair.
        Downstream task runners decide how to execute each type.
    """

    node_id: str
    node_type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A simple DAG workflow representation.

    Attributes:
        nodes: List of nodes.
        edges: List of (src, dst) pairs. Must form a DAG (not enforced).
        metadata: Free-form metadata (e.g., task name, version).
    """

    nodes: List[Node] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def node_index(self) -> Dict[str, Node]:
        return {n.node_id: n for n in self.nodes}

    def successors(self, node_id: str) -> List[str]:
        return [d for s, d in self.edges if s == node_id]

    def predecessors(self, node_id: str) -> List[str]:
        return [s for s, d in self.edges if d == node_id]

    def add_node(self, node: Node) -> None:
        if node.node_id in self.node_index():
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes.append(node)

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.append((src, dst))

    def remove_edge(self, src: str, dst: str) -> None:
        self.edges = [(s, d) for s, d in self.edges if not (s == src and d == dst)]

    def to_dict(self) -> Dict[str, Any]:
        import copy as _copy
        return {
            "nodes": [
                {"id": n.node_id, "type": n.node_type, "config": _copy.deepcopy(n.config)}
                for n in self.nodes
            ],
            "edges": [{"src": s, "dst": d} for s, d in self.edges],
            "metadata": _copy.deepcopy(self.metadata),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Workflow":
        import copy as _copy
        nodes = [Node(node_id=x["id"], node_type=x["type"], config=_copy.deepcopy(x.get("config", {}))) for x in d.get("nodes", [])]
        edges = [(e["src"], e["dst"]) for e in d.get("edges", [])]
        return Workflow(nodes=nodes, edges=edges, metadata=_copy.deepcopy(d.get("metadata", {})))

    def copy(self) -> "Workflow":
        # cheap deep-ish copy
        return Workflow.from_dict(self.to_dict())
