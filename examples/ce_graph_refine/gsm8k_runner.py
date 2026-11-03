"""GSM8K runner for CE-Graph.

Build a `runner(workflow) -> (score, traces)` that:
  1. Executes the workflow on a fixed evaluation set.
  2. Grades each output against the gold numeric answer.
  3. Returns (accuracy, list[ExecutionTrace]) where each trace has is_success set
     and (on failure) an `error` tag that the failure-clustering uses.

Dataset loading:
  - If `datasets` is installed and network is available, pull GSM8K test split.
  - Else, load from a local JSONL at `examples/ce_graph_refine/gsm8k_sample.jsonl`.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from verl.ce_graph.counterexamples import ExecutionTrace
from verl.ce_graph.executor import Executor
from verl.ce_graph.llm import AIDPClient
from verl.ce_graph.workflow import Workflow


GSM8K_ANS_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")
FINAL_INT_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


@dataclass
class GSM8KExample:
    qid: str
    question: str
    gold: float  # numeric gold answer


def _parse_gold(answer_field: str) -> float:
    m = GSM8K_ANS_RE.search(answer_field)
    if m:
        return float(m.group(1).replace(",", ""))
    # fall back: last number in the field
    nums = FINAL_INT_RE.findall(answer_field)
    if nums:
        return float(nums[-1].replace(",", ""))
    raise ValueError(f"cannot parse gold from: {answer_field[:80]}")


def _parse_pred(text: str) -> Optional[float]:
    """Pull the final numeric answer out of a model response."""
    if not text:
        return None
    # Prefer explicit '####' or '答案是' or 'Answer:' patterns.
    m = GSM8K_ANS_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    for marker in ("final answer", "the answer is", "answer:", "答案是", "结果是"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx:]
            nums = FINAL_INT_RE.findall(tail)
            if nums:
                try:
                    return float(nums[0].replace(",", ""))
                except ValueError:
                    pass
    # Last resort: last number in the entire response.
    nums = FINAL_INT_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def load_gsm8k(n: int, split: str = "test", cache_path: Optional[str] = None) -> List[GSM8KExample]:
    """Load `n` examples from GSM8K. Falls back to a local JSONL cache if HF
    `datasets` isn't installed or the network is unavailable.
    """

    cache_path = cache_path or os.path.join(os.path.dirname(__file__), "gsm8k_sample.jsonl")
    examples: List[GSM8KExample] = []

    # Try HuggingFace datasets first.
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i, row in enumerate(ds):
            if i >= n:
                break
            examples.append(
                GSM8KExample(qid=f"{split}-{i}", question=row["question"], gold=_parse_gold(row["answer"]))
            )
        if examples:
            return examples
    except Exception:
        pass

    # Local cache fallback.
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(
                    GSM8KExample(qid=row.get("qid", f"local-{len(examples)}"), question=row["question"], gold=float(row["gold"]))
                )
                if len(examples) >= n:
                    break
        if examples:
            return examples

    raise RuntimeError(
        f"Could not load GSM8K. Install `datasets` for online load, or place a JSONL at {cache_path}."
    )


class GSM8KTaskRunner:
    """Task runner for CE-Graph on GSM8K.

    Exposes:
      - `__call__(workflow)` -> (score, traces): evaluate on the full set.
      - `verify(workflow, instance_ids)` -> success_rate: run only on the listed
        instances (used for Monte Carlo verification of candidate edits).
    """

    def __init__(
        self,
        examples: List[GSM8KExample],
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

    def _run_one(self, workflow: Workflow, ex: GSM8KExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=ex.question)
        pred = _parse_pred(trace.final_output or "")
        is_correct = pred is not None and abs(pred - ex.gold) < 1e-6
        trace.is_success = is_correct
        trace.metrics["pred"] = pred
        trace.metrics["gold"] = ex.gold
        if not is_correct:
            if pred is None:
                trace.error = trace.error or "parse_failure:no_number"
            elif trace.metrics.get("invalid_json"):
                trace.error = trace.error or "invalid_json"
            else:
                trace.error = trace.error or "wrong_answer"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[GSM8KExample]) -> List[ExecutionTrace]:
        if self.n_workers <= 1 or len(examples) <= 1:
            return [self._run_one(workflow, ex) for ex in examples]
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len(examples))) as pool:
            return list(pool.map(lambda ex: self._run_one(workflow, ex), examples))

    def __call__(self, workflow: Workflow) -> Tuple[float, List[ExecutionTrace]]:
        traces = self._run_batch(workflow, self.examples)
        correct = sum(1 for t in traces if t.is_success)
        if self.verbose:
            for ex, t in zip(self.examples, traces):
                tag = "OK" if t.is_success else "X"
                print(f"  [{tag}] {ex.qid}  pred={t.metrics.get('pred')}  gold={ex.gold}")
        return correct / max(1, len(self.examples)), traces

    def verify(self, workflow: Workflow, instance_ids: List[str]) -> float:
        """Monte Carlo verification: run `workflow` on the listed instances."""
        seen = [self.id_to_example[i] for i in instance_ids if i in self.id_to_example]
        if not seen:
            return 0.0
        traces = self._run_batch(workflow, seen)
        return sum(1 for t in traces if t.is_success) / len(seen)


# Back-compat: keep build_gsm8k_runner returning a callable.
def build_gsm8k_runner(
    examples: List[GSM8KExample],
    llm: Optional[AIDPClient] = None,
    max_tokens: int = 384,
    temperature: float = 0.0,
    verbose: bool = False,
) -> "GSM8KTaskRunner":
    return GSM8KTaskRunner(examples=examples, llm=llm, max_tokens=max_tokens, temperature=temperature, verbose=verbose)
