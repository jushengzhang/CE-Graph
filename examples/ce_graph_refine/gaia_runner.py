"""GAIA stand-in runner for CE-Graph.

The original GAIA benchmark is gated on HuggingFace and requires the agent to
use tools (web search, file I/O). For a pure-reasoning analogue we use BIG-Bench
Hard (lukaemon/bbh) tasks, which similarly require multi-step reasoning over
long-context inputs and have crisp multiple-choice ground truth like `(A)`.

Default subtasks span 3 reasoning flavors:
  - logical_deduction_three_objects (constraint reasoning)
  - date_understanding (temporal/arithmetic reasoning)
  - tracking_shuffled_objects_three_objects (state-tracking reasoning)
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

from verl.ce_graph.counterexamples import ExecutionTrace
from verl.ce_graph.executor import Executor
from verl.ce_graph.llm import AIDPClient
from verl.ce_graph.workflow import Workflow


_CHOICE_RE = re.compile(r"\(([A-Z])\)")


@dataclass
class GAIAExample:
    qid: str
    question: str
    gold: str  # e.g. "(A)"
    subtask: str = ""


def _normalize_choice(s: str) -> str:
    if not s:
        return ""
    m = _CHOICE_RE.search(s.strip())
    if m:
        return f"({m.group(1).upper()})"
    return s.strip().upper()


def parse_pred(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer explicit "answer: (X)" or "the answer is (X)" markers near the end
    lower = text.lower()
    for marker in ("final answer is", "the answer is", "answer:", "i choose", "therefore,"):
        idx = lower.rfind(marker)
        if idx >= 0:
            tail = text[idx:idx + 80]
            m = _CHOICE_RE.search(tail)
            if m:
                return f"({m.group(1).upper()})"
    # Otherwise: last (X) in the whole response
    matches = _CHOICE_RE.findall(text)
    if matches:
        return f"({matches[-1].upper()})"
    return None


def load_gaia_proxy(
    n: int,
    subtasks: Tuple[str, ...] = (
        "logical_deduction_three_objects",
        "date_understanding",
        "tracking_shuffled_objects_three_objects",
    ),
    split: str = "test",
) -> List[GAIAExample]:
    from datasets import load_dataset

    per_subtask = max(1, (n + len(subtasks) - 1) // len(subtasks))
    out: List[GAIAExample] = []
    for sub in subtasks:
        try:
            ds = load_dataset("lukaemon/bbh", sub, split=split)
        except Exception as e:
            print(f"  [gaia_runner] subtask {sub} load failed: {e}")
            continue
        n_in_sub = sum(1 for x in out if x.subtask == sub)
        for i, row in enumerate(ds):
            if n_in_sub >= per_subtask:
                break
            out.append(
                GAIAExample(
                    qid=f"{sub}-{i}",
                    question=row["input"],
                    gold=_normalize_choice(row["target"]),
                    subtask=sub,
                )
            )
            n_in_sub += 1
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


class GAIATaskRunner:
    def __init__(
        self,
        examples: List[GAIAExample],
        llm: Optional[AIDPClient] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = False,
        n_workers: int = 8,
    ):
        self.examples = list(examples)
        self.id_to_example = {ex.qid: ex for ex in self.examples}
        self.llm = llm or AIDPClient()
        self.executor = Executor(llm=self.llm, max_tokens=max_tokens, temperature=temperature)
        self.verbose = bool(verbose)
        self.n_workers = max(1, int(n_workers))

    def _run_one(self, workflow: Workflow, ex: GAIAExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=ex.question)
        pred = parse_pred(trace.final_output or "")
        is_correct = pred is not None and pred == ex.gold
        trace.is_success = is_correct
        trace.metrics["pred"] = pred
        trace.metrics["gold"] = ex.gold
        trace.metrics["subtask"] = ex.subtask
        if not is_correct:
            if pred is None:
                trace.error = trace.error or "parse_failure:no_choice"
            else:
                trace.error = trace.error or "wrong_answer"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[GAIAExample]) -> List[ExecutionTrace]:
        if self.n_workers <= 1 or len(examples) <= 1:
            return [self._run_one(workflow, ex) for ex in examples]
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len(examples))) as pool:
            return list(pool.map(lambda ex: self._run_one(workflow, ex), examples))

    def __call__(self, workflow: Workflow) -> Tuple[float, List[ExecutionTrace]]:
        traces = self._run_batch(workflow, self.examples)
        correct = sum(1 for t in traces if t.is_success)
        return correct / max(1, len(self.examples)), traces

    def verify(self, workflow: Workflow, instance_ids: List[str]) -> float:
        seen = [self.id_to_example[i] for i in instance_ids if i in self.id_to_example]
        if not seen:
            return 0.0
        traces = self._run_batch(workflow, seen)
        return sum(1 for t in traces if t.is_success) / len(seen)
