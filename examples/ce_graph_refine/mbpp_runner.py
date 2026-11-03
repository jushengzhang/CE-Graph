"""MBPP runner for CE-Graph.

Dataset: `mbpp` config `sanitized` (257 short Python tasks with assertion tests).
Grading: append `test_list` assertions to the candidate code, run in a
subprocess, success if all assertions pass.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

from verl.ce_graph.counterexamples import ExecutionTrace
from verl.ce_graph.executor import Executor
from verl.ce_graph.llm import AIDPClient
from verl.ce_graph.workflow import Workflow

from examples.ce_graph_refine.code_sandbox import extract_python_block, run_python


@dataclass
class MBPPExample:
    qid: str
    prompt: str  # natural-language problem description
    test_list: List[str]
    test_imports: List[str]
    # one assertion (the first) serves as a canonical example for the LLM prompt
    sample_assert: str


def _safe_list(field) -> List[str]:
    """`test_list` / `test_imports` come back as Python-list-style strings sometimes."""
    if isinstance(field, list):
        return [str(x) for x in field]
    if isinstance(field, str):
        try:
            v = ast.literal_eval(field)
            if isinstance(v, list):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
        return [field]
    return []


def load_mbpp(n: int, split: str = "test") -> List[MBPPExample]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    out: List[MBPPExample] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        tests = _safe_list(row.get("test_list", []))
        imports = _safe_list(row.get("test_imports", []))
        if not tests:
            continue
        out.append(
            MBPPExample(
                qid=str(row.get("task_id", f"mbpp-{i}")),
                prompt=row["prompt"],
                test_list=tests,
                test_imports=imports,
                sample_assert=tests[0],
            )
        )
    return out


def _grade(completion: str, ex: MBPPExample, timeout_s: float = 8.0) -> Tuple[bool, str]:
    code = extract_python_block(completion) or completion
    if not code.strip():
        return False, "empty_completion"

    imports = "\n".join(ex.test_imports)
    asserts = "\n".join(ex.test_list)
    full = f"{imports}\n{code}\n{asserts}\n"
    res = run_python(full, timeout_s=timeout_s)
    if res.ok:
        return True, ""
    return False, res.error or "test_failed"


class MBPPTaskRunner:
    def __init__(
        self,
        examples: List[MBPPExample],
        llm: Optional[AIDPClient] = None,
        max_tokens: int = 768,
        temperature: float = 0.0,
        verbose: bool = False,
        n_workers: int = 8,
        sandbox_timeout: float = 8.0,
        expose_public_tests: bool = False,
    ):
        self.examples = list(examples)
        self.id_to_example = {ex.qid: ex for ex in self.examples}
        self.llm = llm or AIDPClient()
        self.executor = Executor(llm=self.llm, max_tokens=max_tokens, temperature=temperature)
        self.verbose = bool(verbose)
        self.n_workers = max(1, int(n_workers))
        self.sandbox_timeout = float(sandbox_timeout)
        # When True, splice the full MBPP `test_list` (the same asserts used for
        # final grading) into the question via [PUBLIC_TESTS] markers, matching
        # MaAS's mbpp_public_test.jsonl pathway. Single sample_assert leaves the
        # model free to overfit it (e.g. tuple lex sort passes one assert but
        # not the per-column sort the rest expect).
        self.expose_public_tests = bool(expose_public_tests)

    def _question_for(self, ex: MBPPExample) -> str:
        if not self.expose_public_tests:
            return (
                f"{ex.prompt.strip()}\n\nYour code must satisfy this assertion (and others like it):\n"
                f"{ex.sample_assert}"
            )
        asserts = "\n".join(ex.test_list)
        return (
            f"{ex.prompt.strip()}\n\n[PUBLIC_TESTS]\n"
            f"# Your function must pass ALL of these assertions:\n"
            f"{asserts}\n"
            f"[/PUBLIC_TESTS]"
        )

    def _run_one(self, workflow: Workflow, ex: MBPPExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=self._question_for(ex))
        completion = trace.final_output or ""
        ok, err = _grade(completion, ex, timeout_s=self.sandbox_timeout)
        trace.is_success = ok
        if not ok:
            trace.error = trace.error or f"sandbox:{err}"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[MBPPExample]) -> List[ExecutionTrace]:
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
