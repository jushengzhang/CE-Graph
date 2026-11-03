from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .workflow import Workflow, Node
from .failures import FailureMode


@dataclass
class Operator:
    """An operator-constrained graph edit from the library O.

    The operator:
      - has a stable name (for ablations / logging)
      - includes a predicate deciding applicability under a failure mode
      - defines an `apply` function that returns an edited workflow
    """

    name: str
    is_applicable: Callable[[Workflow, FailureMode], bool]
    apply: Callable[[Workflow, FailureMode], Workflow]


class OperatorLibrary:
    """A small, opinionated set of safe graph edits.

    These are meant as defaults. Replace or extend with your domain-specific edits.
    """

    def __init__(self, operators: Optional[Sequence[Operator]] = None):
        self.operators: List[Operator] = list(operators) if operators is not None else []

    def add(self, op: Operator) -> None:
        self.operators.append(op)

    def propose(self, workflow: Workflow, mode: FailureMode, top_k: int = 5) -> List[Operator]:
        cands = [op for op in self.operators if op.is_applicable(workflow, mode)]
        return cands[:top_k]


def default_operator_library() -> OperatorLibrary:
    """Default operator library used in the example script.

    The edits are intentionally conservative and easy to ablate.
    """

    lib = OperatorLibrary()

    def _has_node_type(w: Workflow, t: str) -> bool:
        return any(n.node_type == t for n in w.nodes)

    # 1) Add a lightweight verifier node when failures suggest output format issues.
    def app_add_verifier(w: Workflow, m: FailureMode) -> bool:
        if _has_node_type(w, "verifier"):
            return False
        return any("invalid_json" in kw or "format" in kw for kw in m.keywords)

    def do_add_verifier(w: Workflow, m: FailureMode) -> Workflow:
        w2 = w.copy()
        v = Node(node_id=f"verifier_{len(w2.nodes)}", node_type="verifier", config={"strict": True})
        w2.add_node(v)
        # attach verifier after the last node (conservative)
        if w2.nodes:
            last = w2.nodes[-2].node_id if len(w2.nodes) >= 2 else w2.nodes[0].node_id
            w2.add_edge(last, v.node_id)
        return w2

    lib.add(Operator(name="add_verifier", is_applicable=app_add_verifier, apply=do_add_verifier))

    # 2) Add a reflection/self-check node for semantic errors (common in agent tasks).
    def app_add_reflection(w: Workflow, m: FailureMode) -> bool:
        if _has_node_type(w, "reflection"):
            return False
        return any(kw in {"wrong", "incorrect", "mismatch", "halluc", "contrad"} for kw in m.keywords)

    def do_add_reflection(w: Workflow, m: FailureMode) -> Workflow:
        w2 = w.copy()
        r = Node(
            node_id=f"reflection_{len(w2.nodes)}",
            node_type="reflection",
            config={
                "instruction": "Check the draft answer against the evidence and revise if inconsistent.",
            },
        )
        w2.add_node(r)
        if w2.nodes:
            # insert before final node if exists, else append
            if len(w2.nodes) >= 2:
                prev = w2.nodes[-2].node_id
            else:
                prev = w2.nodes[0].node_id
            w2.add_edge(prev, r.node_id)
        return w2

    lib.add(Operator(name="add_reflection", is_applicable=app_add_reflection, apply=do_add_reflection))

    # 3) Increase tool retry budget when tool errors dominate.
    def app_more_retries(w: Workflow, m: FailureMode) -> bool:
        return any("tool" in kw or "timeout" in kw for kw in m.keywords)

    def do_more_retries(w: Workflow, m: FailureMode) -> Workflow:
        w2 = w.copy()
        for n in w2.nodes:
            if n.node_type in {"tool", "search", "browser"}:
                n.config["max_retries"] = int(n.config.get("max_retries", 1)) + 1
        return w2

    lib.add(Operator(name="increase_tool_retries", is_applicable=app_more_retries, apply=do_more_retries))

    # 4) Add an explicit "constraints" prompt prefix to the first LLM node.
    def app_add_constraints(w: Workflow, m: FailureMode) -> bool:
        return True  # safe, almost always applicable

    def do_add_constraints(w: Workflow, m: FailureMode) -> Workflow:
        w2 = w.copy()
        for n in w2.nodes:
            if n.node_type in {"llm", "policy"}:
                prefix = n.config.get("prompt_prefix", "")
                if "Follow the tool schema" not in prefix:
                    n.config["prompt_prefix"] = (
                        "Follow the tool schema strictly. If uncertain, ask the tool for evidence.\n" + prefix
                    )
                break
        return w2

    lib.add(Operator(name="add_constraints_prefix", is_applicable=app_add_constraints, apply=do_add_constraints))

    return lib
