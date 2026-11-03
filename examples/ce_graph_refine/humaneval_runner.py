"""HumanEval runner for CE-Graph.

Dataset: `openai/openai_humaneval` (164 Python function-completion tasks).
Grading: append the canonical test suite to the candidate completion, run in a
subprocess, success if `check(<entry_point>)` exits zero.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

from verl.ce_graph.counterexamples import ExecutionTrace
from verl.ce_graph.executor import Executor
from verl.ce_graph.llm import AIDPClient
from verl.ce_graph.workflow import Workflow

from examples.ce_graph_refine.code_sandbox import extract_python_block, run_python


@dataclass
class HumanEvalExample:
    qid: str
    prompt: str  # function signature + docstring
    test: str  # check(candidate) ...
    entry_point: str


def load_humaneval(n: int, split: str = "test") -> List[HumanEvalExample]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split=split)
    out: List[HumanEvalExample] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append(
            HumanEvalExample(
                qid=row["task_id"],
                prompt=row["prompt"],
                test=row["test"],
                entry_point=row["entry_point"],
            )
        )
    return out


def _grade(completion: str, ex: HumanEvalExample, timeout_s: float = 8.0) -> Tuple[bool, str]:
    """Run the candidate solution through the test harness.

    Chat-mode LLMs may return either (a) the full function def or (b) just the
    indented body. Either way the test harness needs the prompt's preamble
    (imports like `from typing import List`) on the front.
    """
    import textwrap

    code = extract_python_block(completion) or completion
    if not code.strip():
        return False, "empty_completion"

    def_marker = f"def {ex.entry_point}"
    if def_marker in code:
        # Model emitted the full def — use it directly, but prepend the prompt's
        # preamble (imports etc.) so type hints resolve. Dedent first so a model
        # that nests the def inside an indented code block still parses.
        normalized = textwrap.dedent(code) if code.startswith((" ", "\t")) else code
        idx = ex.prompt.find(def_marker)
        preamble = ex.prompt[:idx] if idx > 0 else ""
        full_program = f"{preamble}{normalized}\n\n{ex.test}\n\ncheck({ex.entry_point})\n"
    else:
        # Model returned only the body. extract_python_block().strip() can lose
        # the first line's indent while keeping later lines indented, producing
        # mixed indentation. Dedent to the common minimum first, then re-add a
        # single 4-space indent so it sits inside the prompt's open def.
        dedented = textwrap.dedent(code)
        body = "\n".join("    " + line if line.strip() else line for line in dedented.splitlines())
        full_program = f"{ex.prompt}{body}\n\n{ex.test}\n\ncheck({ex.entry_point})\n"
    res = run_python(full_program, timeout_s=timeout_s)
    if res.ok:
        return True, ""
    return False, res.error or "test_failed"


class HumanEvalTaskRunner:
    def __init__(
        self,
        examples: List[HumanEvalExample],
        llm: Optional[AIDPClient] = None,
        max_tokens: int = 1024,
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
        # When True, splice the HumanEval `test` field (the same check() suite
        # used for final grading) into the question with [PUBLIC_TESTS] markers,
        # matching what MaAS does via humaneval_public_test.jsonl in its Test
        # operator. Required to reproduce paper's 92.85% MaAS W0.
        self.expose_public_tests = bool(expose_public_tests)

    def _question_for(self, ex: HumanEvalExample) -> str:
        if not self.expose_public_tests:
            return ex.prompt
        return (
            f"{ex.prompt}\n\n[PUBLIC_TESTS]\n"
            f"# These tests will be used to verify your implementation.\n"
            f"{ex.test}\ncheck({ex.entry_point})\n"
            f"[/PUBLIC_TESTS]"
        )

    def _run_one(self, workflow: Workflow, ex: HumanEvalExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=self._question_for(ex))
        completion = trace.final_output or ""
        ok, err = _grade(completion, ex, timeout_s=self.sandbox_timeout)
        trace.is_success = ok
        trace.metrics["entry_point"] = ex.entry_point
        if not ok:
            trace.error = trace.error or f"sandbox:{err}"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[HumanEvalExample]) -> List[ExecutionTrace]:
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
