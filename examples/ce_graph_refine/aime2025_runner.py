"""AIME 2025 runner — integer answers 0-999, 30 problems (AIME I + II).

Dataset: `MathArena/aime_2025_I` + `MathArena/aime_2025_II`, released Feb-Mar 2025.
gpt-4o-mini knowledge cutoff predates this → genuinely out-of-distribution.
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

_INT_RE = re.compile(r"-?\d+")


@dataclass
class AIME2025Example:
    qid: str
    problem: str
    gold: int


def _parse_pred(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        try: return int(m.group(1))
        except: pass
    m = re.search(r"\\boxed\s*\{\s*(-?\d+)\s*\}", text)
    if m:
        try: return int(m.group(1))
        except: pass
    for marker in ("final answer is", "the answer is", "answer:"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx:idx + 200]
            nums = _INT_RE.findall(tail)
            if nums:
                try: return int(nums[0])
                except: pass
    return None


def load_aime_2025(n: int = 30) -> List[AIME2025Example]:
    from datasets import load_dataset

    out: List[AIME2025Example] = []
    for paper, dsname in [("I", "MathArena/aime_2025_I"), ("II", "MathArena/aime_2025_II")]:
        ds = load_dataset(dsname, split="train")
        for row in ds:
            try:
                gold = int(str(row.get("answer", "")).strip())
            except (ValueError, TypeError):
                continue
            qid = f"aime2025_{paper}-{row.get('problem_idx', len(out))}"
            out.append(AIME2025Example(qid=qid, problem=row["problem"], gold=gold))
            if len(out) >= n:
                return out
    return out


def aime2025_format(ex: AIME2025Example) -> Tuple[str, str]:
    """Returns (system, user) prompt pair for LBG/TT-TS agents."""
    sys_p = (
        "You are an expert competition mathematician. The answer to AIME problems "
        "is an integer in 0-999. Solve the problem step by step, then end your "
        "response with the final answer in the format: #### N"
    )
    user_p = (
        f"Problem (AIME 2025): {ex.problem}\n\n"
        f"Reason carefully, then conclude with: #### <integer 0-999>"
    )
    return sys_p, user_p


def aime2025_grade(ex: AIME2025Example, output: str) -> Tuple[bool, str]:
    pred = _parse_pred(output or "")
    if pred is None:
        return False, "parse_failure"
    if pred == ex.gold:
        return True, ""
    return False, f"wrong_answer:{pred}!={ex.gold}"


class AIME2025TaskRunner:
    def __init__(self, examples, llm=None, max_tokens=2000, temperature=0.0, verbose=False, n_workers=8):
        self.examples = list(examples)
        self.id_to_example = {ex.qid: ex for ex in self.examples}
        self.llm = llm or AIDPClient()
        self.executor = Executor(llm=self.llm, max_tokens=max_tokens, temperature=temperature)
        self.verbose = verbose
        self.n_workers = max(1, int(n_workers))

    def _run_one(self, workflow, ex):
        trace = self.executor.run(workflow, instance_id=ex.qid, question=ex.problem)
        pred = _parse_pred(trace.final_output or "")
        ok = pred is not None and pred == ex.gold
        trace.is_success = ok
        trace.metrics["pred"] = pred
        trace.metrics["gold"] = ex.gold
        if not ok:
            trace.error = trace.error or ("parse_failure:no_number" if pred is None else "wrong_answer")
        return trace

    def __call__(self, workflow):
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len(self.examples) or 1)) as pool:
            traces = list(pool.map(lambda ex: self._run_one(workflow, ex), self.examples))
        correct = sum(1 for t in traces if t.is_success)
        return correct / max(1, len(self.examples)), traces

    def verify(self, workflow, instance_ids):
        seen = [self.id_to_example[i] for i in instance_ids if i in self.id_to_example]
        if not seen:
            return 0.0
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len(seen) or 1)) as pool:
            traces = list(pool.map(lambda ex: self._run_one(workflow, ex), seen))
        return sum(1 for t in traces if t.is_success) / len(seen)
