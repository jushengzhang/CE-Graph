"""MATH runner for CE-Graph.

The MATH benchmark (Hendrycks et al. 2021) is harder than GSM8K — answers are
LaTeX expressions wrapped in \\boxed{...}. Baseline gpt-4o-mini accuracy is
~30-40%, leaving plenty of headroom for the refinement loop to show effect.

Dataset: HuggingFace `EleutherAI/hendrycks_math` (7 subjects, ~1.2k test rows each).
Grading: extract last \\boxed{...} from both pred and gold, then string-compare
after light LaTeX normalization. This is conservative — some equivalent answers
(e.g. 1/2 vs 0.5) won't match. That's fine: it underestimates accuracy uniformly,
so refinement delta is still meaningful.
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


_BOXED_RE = re.compile(r"\\boxed\s*\{")


@dataclass
class MATHExample:
    qid: str
    problem: str
    gold: str  # normalized boxed content
    subject: str = ""
    level: str = ""


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of the LAST \\boxed{...} in `text`, handling nested braces."""
    if not text:
        return None
    # Find last \boxed{
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None  # unbalanced


def _normalize_answer(s: str) -> str:
    """Light LaTeX normalization for string-equality grading."""
    if s is None:
        return ""
    s = s.strip()
    # Remove surrounding $...$
    s = re.sub(r"^\$+|\$+$", "", s)
    # Strip redundant outer braces, e.g. an agent wrapping \boxed{{10}} yields "{10}".
    # Repeat to collapse multiple layers ("{{10}}" -> "10").
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        # Only strip when the braces are a single balanced pair around the whole string.
        depth = 0
        balanced_whole = True
        for i, c in enumerate(s):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced_whole = False
                    break
        if balanced_whole:
            s = s[1:-1].strip()
        else:
            break
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    # Common LaTeX equivalences
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\quad", "").replace("\\qquad", "")
    # --- reproduction fix: reduce format-only false-negatives (applied to gold AND pred) ---
    s = re.sub(r"\\(?:text|mbox|mathrm)\s*\{[^{}]*\}", "", s)   # drop \text{degrees} etc
    s = re.sub(r"^[A-Za-z]\\in", "", s)                            # "x\in[-2,7]" -> "[-2,7]"
    s = re.sub(r"^[A-Za-z]=", "", s)                                # "x=5" -> "5"
    s = re.sub(r"\\frac\{([^{}]+)\}([0-9A-Za-z])", r"\\frac{\1}{\2}", s)  # \frac{270}7 -> \frac{270}{7}
    s = re.sub(r"\\frac([0-9A-Za-z])([0-9A-Za-z])", r"\\frac{\1}{\2}", s)   # \frac12 -> \frac{1}{2}
    # Trailing punctuation
    s = s.rstrip(".,;")
    # Decimal "0.5" vs "1/2" not unified here — too risky to normalize without parser.
    return s


def parse_math_gold(solution_field: str) -> str:
    boxed = _extract_boxed(solution_field)
    return _normalize_answer(boxed) if boxed is not None else _normalize_answer(solution_field.split()[-1] if solution_field else "")


def parse_math_pred(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None:
        return _normalize_answer(boxed)
    # Fallback: look for "answer: X" patterns
    for marker in ("final answer is", "answer is", "answer:", "the answer is"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx + len(marker):].strip()
            # take up to a newline or ~50 chars
            tail = tail.split("\n", 1)[0][:60]
            return _normalize_answer(tail)
    return None


def load_math(
    n: int,
    subjects: Tuple[str, ...] = ("algebra", "counting_and_probability", "number_theory", "prealgebra"),
    split: str = "test",
) -> List[MATHExample]:
    """Load `n` MATH examples balanced across subjects."""
    from datasets import load_dataset  # type: ignore

    out: List[MATHExample] = []
    per_subject = max(1, (n + len(subjects) - 1) // len(subjects))
    for subj in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
        except Exception as e:
            print(f"  [math_runner] subject {subj} load failed: {e}")
            continue
        for i, row in enumerate(ds):
            if len([x for x in out if x.subject == subj]) >= per_subject:
                break
            gold = parse_math_gold(row["solution"])
            if not gold:
                continue
            out.append(
                MATHExample(
                    qid=f"{subj}-{i}",
                    problem=row["problem"],
                    gold=gold,
                    subject=subj,
                    level=str(row.get("level", "")),
                )
            )
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


class MATHTaskRunner:
    """Mirror of GSM8KTaskRunner for the MATH benchmark."""

    def __init__(
        self,
        examples: List[MATHExample],
        llm: Optional[AIDPClient] = None,
        max_tokens: int = 768,
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

    def _run_one(self, workflow: Workflow, ex: MATHExample) -> ExecutionTrace:
        trace = self.executor.run(workflow, instance_id=ex.qid, question=ex.problem)
        pred = parse_math_pred(trace.final_output or "")
        is_correct = pred is not None and pred == ex.gold
        trace.is_success = is_correct
        trace.metrics["pred"] = pred
        trace.metrics["gold"] = ex.gold
        trace.metrics["subject"] = ex.subject
        trace.metrics["level"] = ex.level
        if not is_correct:
            if pred is None:
                trace.error = trace.error or "parse_failure:no_boxed"
            else:
                trace.error = trace.error or "wrong_answer"
        return trace

    def _run_batch(self, workflow: Workflow, examples: List[MATHExample]) -> List[ExecutionTrace]:
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
                print(f"  [{tag}] {ex.qid}  pred={t.metrics.get('pred')}  gold={ex.gold[:30]}")
        return correct / max(1, len(self.examples)), traces

    def verify(self, workflow: Workflow, instance_ids: List[str]) -> float:
        seen = [self.id_to_example[i] for i in instance_ids if i in self.id_to_example]
        if not seen:
            return 0.0
        traces = self._run_batch(workflow, seen)
        return sum(1 for t in traces if t.is_success) / len(seen)
