"""TTR — Test-Time Repair: Code-Synthesizing Proposer.

Generalizes CE-Graph's LLMProposer. Instead of emitting only prompt edits or
template-based InsertNode operators, this proposer can synthesize a
**deterministic Python repair function** as a new workflow node.

Mechanically:
  1. See the dominant failure cluster's exemplars with RICH information:
     question / model raw output / expected output / error / parse details.
  2. Decide whether the failure is best fixed by:
     - RevisePrompt           (the prompt instruction was wrong/under-specified)
     - InsertNode(reflection) (the model can self-correct given a second pass)
     - **InsertPythonNode(code, predicate)** (the failure has a deterministic
       fix: format normalization, type coercion, extraction, retry-with-rules)
  3. The Python repair node is the headline novelty — once committed, it runs
     in O(1) without any LLM call, so spillover is mathematically bounded by
     the predicate.

The proposer emits a small set of diverse candidate edits. The outer refine
loop verifies each via Monte Carlo on the cluster, then commits the best one
that passes a regression-set check on previously-passing examples.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .counterexamples import ExecutionTrace
from .failures import FailureMode
from .llm import AIDPClient
from .proposer import Edit, _fallback_edits, _JSON_BLOCK_RE, VALID_OPS
from .workflow import Workflow


def format_rich_exemplars(traces: Sequence[ExecutionTrace], k: int = 5) -> str:
    """Show the proposer the full input/output/error context of failures.

    The Failure-Driven Repair literature (and our empirical experience on
    HumanEval) shows that proposers given only an error string produce vague,
    high-spillover edits. Showing the model's *actual output* lets the proposer
    write precise deterministic fixes.
    """
    lines: List[str] = []
    for i, t in enumerate(list(traces)[:k]):
        q = (t.node_outputs[0].get("question") if t.node_outputs else "") or ""
        if not q:
            # the question lives on the executor's context; fall back to metric
            q = str(t.metrics.get("question", ""))
        out = t.final_output or ""
        err = t.error or ""
        pred = t.metrics.get("pred") if t.metrics else None
        gold = t.metrics.get("gold") if t.metrics else None
        lines.append(f"--- exemplar {i+1} ({t.instance_id}) ---")
        lines.append(f"  error: {err[:300]}")
        if gold is not None:
            lines.append(f"  expected: {str(gold)[:200]}")
        if pred is not None:
            lines.append(f"  parsed_prediction: {str(pred)[:200]}")
        if out:
            snippet = out[:600].replace("\n", "\n    ")
            lines.append(f"  model_output (first 600 chars):\n    {snippet}")
    return "\n".join(lines) if lines else "  (no exemplars)"


SYSTEM_PROMPT = (
    "You are an expert workflow repair engineer. You receive a recurring failure "
    "cluster from an LLM workflow and propose targeted edits to fix it. "
    "You have FIVE edit primitives:\n"
    "  - RevisePrompt: rewrite an existing LLM node's instructions.\n"
    "  - InsertNode: add a templated node (verifier / reflection).\n"
    "  - DeleteNode: remove a node.\n"
    "  - InsertPythonNode: insert a NEW node that runs a deterministic Python "
    "function on the workflow state. This is your strongest tool for **structural** "
    "failures (formatting, parsing, code shape): it executes in O(1) without any "
    "LLM call, so spillover is mathematically bounded by its `applies(state)` predicate.\n"
    "       **SELF-COMPILATION mode**: if the python repair can produce the FINAL "
    "answer directly from the input question (without needing the LLM), return "
    "`{\"output\": <answer>, \"terminal\": True}` from `repair(state)`. Place the "
    "node BEFORE the LLM (position=\"before\", before=\"<llm_node_id>\"). The "
    "executor will skip the LLM call entirely for inputs the predicate matches. "
    "This **reduces inference cost** while preserving accuracy — propose it when "
    "you spot a sub-cluster the LLM keeps getting wrong but is mechanically "
    "trivial (simple arithmetic patterns, regex extraction from input, lookup).\n"
    "  - InsertNode(type=\"self_consistency\"): inserts a K-sample self-consistency "
    "node that re-asks the LLM K times with temperature>0 and takes the majority answer. "
    "This is your strongest tool for **reasoning-quality** failures (wrong_answer, "
    "AssertionError, calculation mistakes) on math/MC tasks — Wang et al. 2022 shows "
    "+10-18% gains. Cost: K× LLM calls per question. Use K=3 by default, K=5 for hard cases.\n\n"
    "DECISION RULE:\n"
    "  - Structural failure (format / parse / indentation): InsertPythonNode\n"
    "  - Reasoning failure (wrong_answer / assertion logical bug) on math/multiple-choice: InsertNode(self_consistency)\n"
    "  - Format-instruction failure (model ignores boxed/####): RevisePrompt\n"
    "  - Simple semantic gap: InsertNode(reflection)\n\n"
    "Reply with strict JSON only."
)


USER_TEMPLATE = """Current workflow (nodes are executed in listed order):
{workflow_json}

Dominant failure mode:
  size: {mode_size}
  signature: {keywords}

Exemplar failures (input -> model output -> error):
{exemplars}

Now propose {n} DIVERSE edits, each plausibly reducing this failure mode.
Output a JSON object exactly of the form:
{{"edits": [
  {{
    "op": "InsertPythonNode" | "RevisePrompt" | "InsertNode" | "DeleteNode",
    "args": {{...}},
    "rationale": "<one sentence>"
  }},
  ...
]}}

ARGS CONTRACT:

For InsertPythonNode (preferred for structural failures):
  args = {{
    "node_id": "<descriptive_id>",
    "code": "def repair(state):\\n    # state['question'] is the original input.\\n    # state['last_output'] is the previous node's output (string).\\n    # state['error'] is the error string from earlier nodes (mostly empty).\\n    text = state['last_output']\\n    # ... return new text\\n    return text",
    "predicate": "def applies(state):\\n    # Decide cheaply whether to apply this repair.\\n    # IMPORTANT: you only see the workflow state, NOT grader feedback.\\n    # Gate on visible properties of state['last_output'] (e.g., missing\\n    # marker, mismatched braces, unparseable code), NOT on hypothetical\\n    # downstream grader errors.\\n    out = state['last_output']\\n    return '<some-visible-marker>' not in out",
    "function_name": "repair",
    "position": "after",
    "after": "<existing_node_id>"
  }}
  - Available libs in code: re, json, math, itertools, ast, functools, string, collections.
  - Code must be valid Python. Quote properly.
  - `applies(state)` MUST return False for inputs that look already-correct.
    This is how spillover is bounded mathematically: the repair only runs on
    inputs matching its visible predicate.
  - The repair can do its OWN diagnosis (e.g., wrap a compile(code) in try/except)
    rather than relying on external error information.

For RevisePrompt:
  args = {{"node_id": "<node>", "system"|"user_template"|"prompt_prefix": "<new text>"}}
For InsertNode:
  args = {{"node_type": "verifier"|"reflection"|"self_consistency", "config": {{...}}, "position": "after", "after": "<id>"}}
  For self_consistency, config = {{"k": 3, "temperature": 0.7, "system": "<sys>", "user_template": "<template using {{question}}>"}}.
  The node automatically extracts the answer with regexes matching #### NUM, \\boxed{{}}, or Answer: (X) and returns the majority.
For DeleteNode:
  args = {{"node_id": "<id>"}}

CONSTRAINTS:
- Pick at least 1 InsertPythonNode if the failure looks like a format issue.
- Keep code under 30 lines per repair.
- Output nothing outside the JSON object.
"""


class CodeSynthProposer:
    """LLM proposer that can synthesize Python repair nodes.

    Drop-in replacement for `LLMProposer`. Honors the same `.propose(...)`
    signature so the outer refine loop is unchanged.
    """

    def __init__(
        self,
        llm: AIDPClient,
        n_candidates: int = 4,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        self.llm = llm
        self.n_candidates = int(n_candidates)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def propose(
        self,
        workflow: Workflow,
        mode: FailureMode,
        exemplar_traces: Sequence[ExecutionTrace],
    ) -> List[Edit]:
        prompt = USER_TEMPLATE.format(
            workflow_json=json.dumps(workflow.to_dict(), ensure_ascii=False, indent=2),
            mode_size=mode.size,
            keywords=", ".join(mode.keywords) or "(none)",
            exemplars=format_rich_exemplars(exemplar_traces, k=min(5, len(exemplar_traces) or 1)),
            n=self.n_candidates,
        )
        resp = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        edits = _parse_edits_lenient(resp.text)
        if not edits:
            edits = _fallback_edits(workflow, mode, self.n_candidates)
        return edits[: self.n_candidates]


def _parse_edits_lenient(text: str) -> List[Edit]:
    """Parse a JSON edit list. More forgiving than the base parser — handles
    fenced code blocks and minor format drift.
    """
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Try to fix common JSON issues: trailing commas, unescaped newlines in code strings.
        repaired = m.group(0)
        repaired = re.sub(r",(\s*[\]}])", r"\1", repaired)  # trailing comma
        try:
            obj = json.loads(repaired)
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
