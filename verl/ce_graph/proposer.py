"""LLM-based Proposer for CE-Graph (paper §3.2.3 Propose-and-Verify).

Given the current workflow W_t and a dominant failure mode b*_t (with exemplar
counterexamples), an LLM proposes N candidate edits Δ_i ∈ A(W_t, O). Edits are
constrained to the paper's operator library O = {RevisePrompt, InsertNode, DeleteNode}.
The verifier (see refine.py) then estimates V(Δ_i) via Monte Carlo on b*_t.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .counterexamples import ExecutionTrace
from .failures import FailureMode
from .llm import AIDPClient
from .workflow import Node, Workflow


VALID_OPS = ("RevisePrompt", "InsertNode", "DeleteNode", "InsertPythonNode")


@dataclass
class Edit:
    """A single graph edit in the constrained library O."""

    op: str
    args: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op {self.op!r}; must be one of {VALID_OPS}")

    def short(self) -> str:
        return f"{self.op}({json.dumps(self.args, ensure_ascii=False)})"


def apply_edit(workflow: Workflow, edit: Edit) -> Workflow:
    """Return a new workflow with the edit applied. Raises on malformed edits."""

    w = workflow.copy()
    idx = w.node_index()

    if edit.op == "RevisePrompt":
        nid = edit.args.get("node_id")
        if nid not in idx:
            raise ValueError(f"RevisePrompt: unknown node_id {nid}")
        node = idx[nid]
        new_prefix = edit.args.get("prompt_prefix")
        new_system = edit.args.get("system")
        new_user_template = edit.args.get("user_template")
        if new_prefix is not None:
            node.config["prompt_prefix"] = str(new_prefix)
        if new_system is not None:
            node.config["system"] = str(new_system)
        if new_user_template is not None:
            node.config["user_template"] = str(new_user_template)
        if all(x is None for x in (new_prefix, new_system, new_user_template)):
            raise ValueError("RevisePrompt: at least one of prompt_prefix/system/user_template must be set")
        return w

    if edit.op == "InsertNode":
        node_type = edit.args.get("node_type")
        if not node_type:
            raise ValueError("InsertNode: node_type required")
        new_node = Node(
            node_id=edit.args.get("node_id") or f"{node_type}_{len(w.nodes)}",
            node_type=str(node_type),
            config=dict(edit.args.get("config") or {}),
        )
        if new_node.node_id in idx:
            new_node = Node(node_id=f"{new_node.node_id}_{len(w.nodes)}", node_type=new_node.node_type, config=new_node.config)
        position = edit.args.get("position", "append")
        after = edit.args.get("after")
        if position == "after" and after and after in idx:
            after_pos = next(i for i, n in enumerate(w.nodes) if n.node_id == after)
            w.nodes.insert(after_pos + 1, new_node)
            if after_pos + 2 < len(w.nodes):
                downstream = w.nodes[after_pos + 2].node_id
                w.add_edge(new_node.node_id, downstream)
            w.add_edge(after, new_node.node_id)
        else:
            if w.nodes:
                last = w.nodes[-1].node_id
                w.add_node(new_node)
                w.add_edge(last, new_node.node_id)
            else:
                w.add_node(new_node)
        return w

    if edit.op == "InsertPythonNode":
        # A Python repair node: agent-synthesized deterministic post-processor.
        # config.code: function source defining `def repair(state) -> str`.
        # config.predicate (optional): predicate source defining `def applies(state) -> bool`.
        # If `applies` returns False the node passes the input through unchanged.
        code = edit.args.get("code")
        if not code or not isinstance(code, str):
            raise ValueError("InsertPythonNode: args.code (str) is required")
        node_id = edit.args.get("node_id") or f"py_repair_{len(w.nodes)}"
        if node_id in idx:
            node_id = f"{node_id}_{len(w.nodes)}"
        config: Dict[str, Any] = {
            "code": code,
            "predicate": edit.args.get("predicate", ""),
            "function_name": edit.args.get("function_name", "repair"),
        }
        new_node = Node(node_id=node_id, node_type="python", config=config)
        position = edit.args.get("position", "append")
        after = edit.args.get("after")
        before = edit.args.get("before")
        if position == "before" and before and before in idx:
            # Self-Compilation `BypassLLM`-style insertion: gate a downstream
            # LLM node behind this Python predicate+repair. If the repair
            # returns terminal=True, executor short-circuits the LLM call.
            before_pos = next(i for i, n in enumerate(w.nodes) if n.node_id == before)
            w.nodes.insert(before_pos, new_node)
            for s, d in list(w.edges):
                if d == before:
                    w.edges.remove((s, d))
                    w.add_edge(s, new_node.node_id)
            w.add_edge(new_node.node_id, before)
        elif position == "after" and after and after in idx:
            after_pos = next(i for i, n in enumerate(w.nodes) if n.node_id == after)
            w.nodes.insert(after_pos + 1, new_node)
            if after_pos + 2 < len(w.nodes):
                w.add_edge(new_node.node_id, w.nodes[after_pos + 2].node_id)
            w.add_edge(after, new_node.node_id)
        else:
            if w.nodes:
                last = w.nodes[-1].node_id
                w.add_node(new_node)
                w.add_edge(last, new_node.node_id)
            else:
                w.add_node(new_node)
        return w

    if edit.op == "DeleteNode":
        nid = edit.args.get("node_id")
        if nid not in idx:
            raise ValueError(f"DeleteNode: unknown node_id {nid}")
        if len(w.nodes) <= 1:
            raise ValueError("DeleteNode: cannot delete the last remaining node")
        w.nodes = [n for n in w.nodes if n.node_id != nid]
        w.edges = [(s, d) for s, d in w.edges if s != nid and d != nid]
        return w

    raise AssertionError("unreachable")  # VALID_OPS guard


class LLMProposer:
    """Constrained Proposer LLM (paper §3.2.3).

    Generates N diverse candidate edits in JSON. Each edit must be in
    {RevisePrompt, InsertNode, DeleteNode}. Falls back to heuristic edits if
    the LLM output cannot be parsed.
    """

    SYSTEM = (
        "You are an expert workflow optimizer. Given a current workflow and a recurring failure mode, "
        "propose a small set of targeted edits that will most likely repair the failure mode without "
        "breaking other behavior. You must reply with strictly valid JSON only."
    )

    USER_TEMPLATE = """Current workflow (nodes are executed in listed order):
{workflow_json}

Available operator library O (you MUST pick from these):
- RevisePrompt(node_id, prompt_prefix?|system?|user_template?): rewrite an LLM node's instructions.
- InsertNode(node_type, node_id?, config?, position="after", after?): add a node. Useful node types:
    "reflection" (config.instruction="<short check-and-revise instruction>"): runs the LLM again to re-examine and possibly revise the draft answer. EXCELLENT for code tasks and complex reasoning.
    "self_consistency" (config.k=N, config.temperature=T): re-runs the LLM K times and votes on the answer. Helps with numerical math.
    "verifier" (config.require_integer=True|config.require_json=True): cheap deterministic format check between two LLM steps; useful when a downstream node assumes a specific output shape.
    "test_reflect_loop" (config.max_rounds=N): MaAS-style operator — runs visible asserts / docstring tests, on failure feeds the error back to LLM for a fix, loops up to N rounds. EXCELLENT for inserting between MaAS multi_gen/sc_pick and the original test_loop to add another verification pass.
- DeleteNode(node_id): remove a node from the chain.

**Hard guidance for code tasks (HumanEval / MBPP) — prefer InsertNode over RevisePrompt**:
- At least HALF of your {n} proposals MUST be InsertNode (reflection, test_reflect_loop, verifier, self_consistency).
- Prompt rewording alone almost never reduces sandbox assertion failures; structural edits do.
- Good InsertNode targets: add a 2nd reflection AFTER test_loop with code-review framing; add a verifier between sc_pick and test_loop; add a second test_reflect_loop with different max_rounds; insert reflection focused on a specific failure-keyword.

For numeric/text tasks (math): RevisePrompt and InsertNode("self_consistency") work well.

Dominant failure mode summary:
  size: {mode_size}
  keywords: {keywords}
  exemplar errors:
{exemplars}

Propose exactly {n} *diverse* edits, each plausibly reducing this failure mode.
Output a JSON object of the form:
{{"edits": [
  {{"op": "RevisePrompt"|"InsertNode"|"DeleteNode", "args": {{...}}, "rationale": "<one sentence>"}},
  ...
]}}
Constraints:
- args must reference actual node_ids when used.
- Do not output anything outside the JSON object.
- Keep prompt edits concise (under 600 chars).
"""

    def __init__(self, llm: AIDPClient, n_candidates: int = 4, max_tokens: int = 1200):
        self.llm = llm
        self.n_candidates = int(n_candidates)
        self.max_tokens = int(max_tokens)

    def propose(
        self,
        workflow: Workflow,
        mode: FailureMode,
        exemplar_traces: Sequence[ExecutionTrace],
    ) -> List[Edit]:
        prompt = self.USER_TEMPLATE.format(
            workflow_json=json.dumps(workflow.to_dict(), ensure_ascii=False, indent=2),
            mode_size=mode.size,
            keywords=", ".join(mode.keywords) or "(none)",
            exemplars=_format_exemplars(exemplar_traces),
            n=self.n_candidates,
        )
        resp = self.llm.chat(
            [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.6,  # moderate diversity
        )
        edits = _parse_edits(resp.text)
        if not edits:
            edits = _fallback_edits(workflow, mode, self.n_candidates)
        return edits[: self.n_candidates]


# ---- helpers ---------------------------------------------------------------


def _format_exemplars(traces: Sequence[ExecutionTrace], k: int = 3) -> str:
    lines: List[str] = []
    for t in list(traces)[:k]:
        out = (t.final_output or "")[:200].replace("\n", " ")
        err = (t.error or "(no error tag)")[:80]
        pred = t.metrics.get("pred") if t.metrics else None
        gold = t.metrics.get("gold") if t.metrics else None
        lines.append(f"  - id={t.instance_id} error={err} pred={pred} gold={gold} out=\"{out}\"")
    return "\n".join(lines) if lines else "  (no exemplars)"


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_edits(text: str) -> List[Edit]:
    if not text:
        return []
    # Strip markdown fences if present.
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    raw_edits = obj.get("edits") if isinstance(obj, dict) else None
    if not isinstance(raw_edits, list):
        return []
    out: List[Edit] = []
    for e in raw_edits:
        if not isinstance(e, dict):
            continue
        op = e.get("op")
        if op not in VALID_OPS:
            continue
        args = e.get("args") or {}
        if not isinstance(args, dict):
            continue
        try:
            out.append(Edit(op=op, args=args, rationale=str(e.get("rationale", ""))[:200]))
        except ValueError:
            continue
    return out


def _fallback_edits(workflow: Workflow, mode: FailureMode, n: int) -> List[Edit]:
    """Heuristic edits when the proposer fails to return valid JSON."""

    cand: List[Edit] = []
    llm_nodes = [n.node_id for n in workflow.nodes if n.node_type == "llm"]
    if llm_nodes:
        cand.append(
            Edit(
                op="RevisePrompt",
                args={
                    "node_id": llm_nodes[0],
                    "prompt_prefix": "Be careful with arithmetic. Re-read the question. Show the calculation explicitly before stating the final answer.",
                },
                rationale="fallback: emphasize careful arithmetic",
            )
        )
    if not any(n.node_type == "reflection" for n in workflow.nodes):
        cand.append(
            Edit(
                op="InsertNode",
                args={
                    "node_type": "reflection",
                    "config": {
                        "instruction": "Re-check the arithmetic step by step. If you find an error, output the corrected '#### <number>'. Otherwise repeat the original final line.",
                    },
                },
                rationale="fallback: append reflection",
            )
        )
    if not any(n.node_type == "verifier" for n in workflow.nodes):
        cand.append(
            Edit(
                op="InsertNode",
                args={"node_type": "verifier", "config": {"require_integer": True}},
                rationale="fallback: verifier for numeric output",
            )
        )
    return cand[:n]
