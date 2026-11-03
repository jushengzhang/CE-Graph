"""MultiArith runner for CE-Graph.

Dataset: `ChilleD/MultiArith` (180 elementary-school arithmetic word problems).
Answer is a single integer; grading is exact-match on the parsed integer.
Mirrors the structure of `gsm8k_runner.py`.
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


FINAL_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


@dataclass
class MultiArithExample:
    qid: str
    question: str
    gold: float


def _parse_gold(answer: str) -> float:
    nums = FINAL_NUM_RE.findall(str(answer))
    if not nums:
        raise ValueError(f"no number in answer: {answer!r}")
    return float(nums[-1].replace(",", ""))


def _parse_pred(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    for marker in ("final answer is", "the answer is", "answer:", "答案是"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx:]
            nums = FINAL_NUM_RE.findall(tail)
            if nums:
                try:
                    return float(nums[0].replace(",", ""))
                except ValueError:
                    pass
    nums = FINAL_NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def load_multiarith(n: int, split: str = "test") -> List[MultiArithExample]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("ChilleD/MultiArith", split=split)
    out: List[MultiArithExample] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        try:
            gold = _parse_gold(row["final_ans"])
        except ValueError:
            continue
        out.append(MultiArithExample(qid=f"ma-{i}", question=row["question"].strip(), gold=gold))
    return out


class MultiArithTaskRunner:
    def __init__(
        self,
        examples: List[MultiArithExample],
        llm: Optional[AIDPClient] = None,
        max_tokens: int = 384,
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

    def _run_one(self, workflow: Workflow, ex: MultiArithExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=ex.question)
        pred = _parse_pred(trace.final_output or "")
        is_correct = pred is not None and abs(pred - ex.gold) < 1e-6
        trace.is_success = is_correct
        trace.metrics["pred"] = pred
        trace.metrics["gold"] = ex.gold
        if not is_correct:
            if pred is None:
                trace.error = trace.error or "parse_failure:no_number"
            else:
                trace.error = trace.error or "wrong_answer"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[MultiArithExample]) -> List[ExecutionTrace]:
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
