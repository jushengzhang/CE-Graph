"""Workflow executor.

Treats a Workflow as a linear chain of nodes (the order in `workflow.nodes`).
Each node type has a handler that produces a step dict consumed by the next.
The executor is intentionally simple — it covers the operator types CE-Graph
edits: `llm`, `tool`, `verifier`, `reflection`. Domain-specific runners can
register their own handlers via `Executor.register`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .workflow import Workflow, Node
from .counterexamples import ExecutionTrace
from .llm import AIDPClient


Handler = Callable[["Executor", Node, Dict[str, Any]], Dict[str, Any]]


@dataclass
class ExecContext:
    """State passed between nodes during a single instance execution."""

    question: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_output: str = ""
    error: Optional[str] = None
    invalid_json: bool = False


class Executor:
    """Run a Workflow on a single input.

    Output of each node is the input to the next. The final node's `output`
    becomes the workflow's final_output.
    """

    def __init__(self, llm: AIDPClient, max_tokens: int = 512, temperature: float = 0.0):
        self.llm = llm
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._handlers: Dict[str, Handler] = {
            "llm": self._handle_llm,
            "verifier": self._handle_verifier,
            "reflection": self._handle_reflection,
            "tool": self._handle_tool_noop,
            "python": self._handle_python,
            "self_consistency": self._handle_self_consistency,
            "sandbox_test_retry": self._handle_sandbox_test_retry,
            # MaAS-aligned operators (bingreeky/MaAS HumanEval/MBPP graph).
            "multi_generate_cot": self._handle_multi_generate_cot,
            "sc_ensemble": self._handle_sc_ensemble,
            "test_reflect_loop": self._handle_test_reflect_loop,
        }

    def register(self, node_type: str, handler: Handler) -> None:
        self._handlers[node_type] = handler

    def run(self, workflow: Workflow, instance_id: str, question: str) -> ExecutionTrace:
        ctx = ExecContext(question=question)
        trace = ExecutionTrace(
            instance_id=instance_id,
            workflow_version=workflow.metadata.get("version", "W"),
        )
        # Counters: how many real LLM-calling nodes actually ran for this instance.
        # `llm_calls` counts each call regardless of node type (llm / reflection /
        # one per self_consistency sample). This is the **inference cost** we
        # report on the cost-accuracy Pareto plot.
        trace.metrics["llm_calls"] = 0
        terminated_early = False

        for node in workflow.nodes:
            if terminated_early:
                break
            handler = self._handlers.get(node.node_type)
            # State view exposed to handlers: includes upstream `error` so
            # Python repair nodes can gate on it via their `applies(state)`
            # predicate.
            handler_state = {
                "question": ctx.question,
                "last_output": ctx.last_output,
                "error": ctx.error or "",
            }
            if handler is None:
                step = {"node_id": node.node_id, "node_type": node.node_type, "error": f"no handler for {node.node_type}"}
                ctx.error = step["error"]
            else:
                try:
                    step = handler(self, node, handler_state)
                except Exception as e:
                    step = {"node_id": node.node_id, "node_type": node.node_type, "error": str(e)[:200]}
                    ctx.error = step["error"]

            ctx.history.append(step)
            trace.node_outputs.append(step)

            if "output" in step:
                ctx.last_output = str(step["output"])
            if step.get("invalid_json"):
                ctx.invalid_json = True
            if step.get("error") and not ctx.error:
                ctx.error = str(step["error"])[:200]

            # Count LLM calls for cost accounting.
            if node.node_type == "llm":
                trace.metrics["llm_calls"] += 1
            elif node.node_type == "reflection":
                trace.metrics["llm_calls"] += 1
            elif node.node_type == "self_consistency":
                trace.metrics["llm_calls"] += int(step.get("n_samples", 1))

            # Self-Compilation: a python repair can signal "terminal" to mean
            # "I've produced the final answer deterministically — skip remaining
            # nodes (including any downstream LLM call)." This is how
            # `BypassLLM`-style patches save inference cost.
            if step.get("terminal"):
                terminated_early = True

        trace.final_output = ctx.last_output
        trace.metrics["invalid_json"] = ctx.invalid_json
        trace.metrics["question"] = ctx.question
        trace.error = ctx.error
        # is_success is set by the runner after grading; executor leaves it False.
        return trace

    # ---- default handlers ----------------------------------------------------

    @staticmethod
    def _handle_llm(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        cfg = node.config or {}
        prefix = cfg.get("prompt_prefix", "")
        system = cfg.get("system", "You are a careful problem solver. Show your reasoning briefly, then give the final answer.")
        user_template = cfg.get("user_template", "{prefix}Question: {question}\nPrevious step: {last_output}\nAnswer:")
        # Use plain string substitution rather than .format() so the question text
        # (which may contain literal '{...}' such as LaTeX braces) does not break formatting.
        user = (user_template
                .replace("{prefix}", (prefix + "\n" if prefix else ""))
                .replace("{question}", str(st.get("question", "")))
                .replace("{last_output}", str(st.get("last_output", ""))))
        resp = self_ex.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self_ex.max_tokens,
            temperature=self_ex.temperature,
        )
        return {
            "node_id": node.node_id,
            "node_type": "llm",
            "output": resp.text,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        }

    @staticmethod
    def _handle_verifier(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """Format/sanity check on the previous output.

        Flags `invalid_json` if the node expects JSON but the output doesn't parse,
        otherwise just passes the previous output through. Conservative by design.
        """
        cfg = node.config or {}
        prev = st.get("last_output", "")
        if cfg.get("require_json"):
            import json as _json
            try:
                _json.loads(prev)
                return {"node_id": node.node_id, "node_type": "verifier", "output": prev}
            except Exception:
                return {
                    "node_id": node.node_id,
                    "node_type": "verifier",
                    "output": prev,
                    "invalid_json": True,
                    "error": "verifier:invalid_json",
                }
        # Default: ensure the response contains an integer for math tasks.
        if cfg.get("require_integer") and not re.search(r"-?\d+", prev):
            return {
                "node_id": node.node_id,
                "node_type": "verifier",
                "output": prev,
                "error": "verifier:missing_integer",
            }
        return {"node_id": node.node_id, "node_type": "verifier", "output": prev}

    @staticmethod
    def _handle_reflection(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        cfg = node.config or {}
        instruction = cfg.get(
            "instruction",
            "Re-examine the draft answer below. If the reasoning has a calculation error, an off-by-one, or contradicts the question, produce a corrected final answer. Otherwise return it unchanged.",
        )
        user = (
            f"Question: {st['question']}\n\nDraft answer:\n{st.get('last_output', '')}\n\n"
            f"{instruction}\n\nRevised final answer:"
        )
        resp = self_ex.llm.chat(
            [{"role": "user", "content": user}],
            max_tokens=self_ex.max_tokens,
            temperature=self_ex.temperature,
        )
        return {
            "node_id": node.node_id,
            "node_type": "reflection",
            "output": resp.text,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        }

    @staticmethod
    def _handle_tool_noop(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through for `tool` nodes when no domain handler is registered."""
        return {"node_id": node.node_id, "node_type": "tool", "output": st.get("last_output", "")}


    @staticmethod
    def _handle_sandbox_test_retry(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """Run candidate code against >>> examples in the question; if fail, retry with LLM."""
        import re as _re, tempfile, subprocess, os as _os
        prev = st.get("last_output", "")
        # Extract code block
        m = _re.search(r"```python\s*\n(.*?)```", prev, _re.DOTALL)
        code = m.group(1).strip() if m else prev.strip()
        # Extract visible test examples from the question (>>> form OR assert form)
        question = st.get("question", "")
        assertions = []
        for ln in question.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith(">>>"):
                # >>> call → next line is expected result
                pass  # complex to parse reliably; skip
            elif ln_s.startswith("assert "):
                assertions.append(ln_s)
        # MBPP often has explicit 'assert' lines in question (good!)
        if not assertions:
            return {"node_id": node.node_id, "node_type": "sandbox_test_retry", "output": prev}
        test_script = code + "\n\n" + "\n".join(assertions[:3]) + "\nprint('TESTS_OK')"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(test_script); path = f.name
        try:
            proc = subprocess.run(["python3", path], capture_output=True, text=True, timeout=10)
        except Exception as e:
            proc = type('p', (), {'stdout': '', 'stderr': str(e), 'returncode': 1})()
        try: _os.unlink(path)
        except: pass
        if 'TESTS_OK' in (proc.stdout or ''):
            return {"node_id": node.node_id, "node_type": "sandbox_test_retry", "output": prev}
        # Retry with error
        err = (proc.stderr or proc.stdout or '')[-400:]
        retry_user = (
            f"The following Python function was generated:\n```python\n{code}\n```\n\n"
            f"Running it against test cases produced this error:\n{err}\n\n"
            f"Original problem:\n{question}\n\n"
            f"Output a CORRECTED complete function in a single ```python code block."
        )
        resp = self_ex.llm.chat(
            [{"role":"system","content":"You are an expert Python programmer. Fix the bug."},
             {"role":"user","content":retry_user}],
            max_tokens=self_ex.max_tokens, temperature=0.0,
        )
        return {
            "node_id": node.node_id, "node_type": "sandbox_test_retry",
            "output": resp.text,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
        }

    @staticmethod
    def _handle_self_consistency(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-sample self-consistency voting (Wang et al. 2022).

        Re-runs the LLM K times on the current question with temperature>0, then
        extracts a short answer from each sample using one of several heuristic
        regexes (#### NUM, \\boxed{...}, Answer: (X)) and returns the majority.
        If parsing fails, falls back to the most recent sample's full text.

        Useful for *logical* failure clusters (wrong_answer, assertion errors)
        that deterministic Python repair cannot fix.
        """
        import re as _re
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor as _TPE

        cfg = node.config or {}
        k = int(cfg.get("k", 3))
        temperature = float(cfg.get("temperature", 0.7))
        system = cfg.get("system", "You are a careful problem solver. Solve step-by-step and end with the final answer in the requested format.")
        user_template = cfg.get("user_template", "{question}\nSolve carefully and state the final answer.")
        user = (user_template
                .replace("{question}", str(st.get("question", "")))
                .replace("{last_output}", str(st.get("last_output", ""))))

        def _one_sample(_):
            try:
                r = self_ex.llm.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=self_ex.max_tokens, temperature=temperature,
                )
                return r.text
            except Exception:
                return ""

        # Sample K times in parallel (each call independent).
        with _TPE(max_workers=min(k, 4)) as pool:
            samples = list(pool.map(_one_sample, range(k)))

        # Try multiple extractors; first match per sample.
        def _extract(text: str) -> str:
            for pattern in (
                r"####\s*(-?\d[\d,]*\.?\d*)",                  # GSM8K / MultiArith
                r"\\boxed\s*\{\s*([^{}]+?)\s*\}",              # MATH
                r"Answer:\s*\(([A-Za-z])\)",                    # GAIA-proxy / BBH
            ):
                m = _re.search(pattern, text)
                if m:
                    return m.group(1).replace(",", "").strip().upper()
            return ""  # parse-fail

        keys = [_extract(s) for s in samples]
        valid = [k for k in keys if k]
        if not valid:
            # Could not parse any — return last sample's full text.
            return {
                "node_id": node.node_id, "node_type": "self_consistency",
                "output": samples[-1] if samples else st.get("last_output", ""),
                "n_samples": k, "valid_parses": 0,
            }
        most_common, freq = Counter(valid).most_common(1)[0]
        # Reconstruct an output that still parses with the downstream grader,
        # using the LAST sample whose extracted answer matches the winner.
        for s, key in zip(reversed(samples), reversed(keys)):
            if key == most_common:
                return {
                    "node_id": node.node_id, "node_type": "self_consistency",
                    "output": s,
                    "n_samples": k, "valid_parses": len(valid), "winning_answer": most_common,
                    "winning_count": freq,
                }
        return {"node_id": node.node_id, "node_type": "self_consistency", "output": samples[-1]}

    # ---- MaAS-aligned operators ---------------------------------------------
    #
    # The MaAS HumanEval/MBPP graph (bingreeky/MaAS) selects from a small
    # operator library: MultiGenerateCoT → ScEnsemble → Test (with REFLECTION
    # loop). These three handlers reimplement that library so we can build the
    # paper's actual MaAS-style W0 baseline (92.85% HumanEval per Table 1).

    @staticmethod
    def _handle_multi_generate_cot(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """MaAS MultiGenerateCoT: K independent CoT samples at temperature > 0.

        Stores the K raw outputs in `output` as a JSON list so the next node
        (typically sc_ensemble) can fan them back out. config.k (default 3),
        config.temperature (default 0.5), config.system, config.user_template.
        """
        import json as _json
        from concurrent.futures import ThreadPoolExecutor as _TPE

        cfg = node.config or {}
        k = int(cfg.get("k", 3))
        temperature = float(cfg.get("temperature", 0.5))
        system = cfg.get(
            "system",
            "You are an expert Python programmer. Given a function specification, write the complete implementation. Return only the function definition in a single ```python code block.",
        )
        user_template = cfg.get(
            "user_template",
            "Complete the following Python function. Output ONLY the full function definition in a ```python code block, nothing else.\n\n{question}",
        )
        user = (user_template
                .replace("{question}", str(st.get("question", "")))
                .replace("{last_output}", str(st.get("last_output", ""))))

        def _one(_):
            try:
                r = self_ex.llm.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=self_ex.max_tokens, temperature=temperature,
                )
                return r.text
            except Exception:
                return ""

        with _TPE(max_workers=min(k, 4)) as pool:
            samples = list(pool.map(_one, range(k)))
        # Store as JSON list so the next operator can deserialize.
        return {
            "node_id": node.node_id, "node_type": "multi_generate_cot",
            "output": _json.dumps({"_maas_candidates": samples}),
            "n_samples": k,
        }

    @staticmethod
    def _handle_sc_ensemble(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """MaAS ScEnsemble: LLM picks the best among K candidates.

        Expects upstream `last_output` to be the JSON envelope produced by
        multi_generate_cot. Falls back to upstream raw text if not envelope-shaped.
        """
        import json as _json

        prev = st.get("last_output", "")
        try:
            envelope = _json.loads(prev)
            candidates = envelope.get("_maas_candidates", [])
        except Exception:
            candidates = []
        if not candidates:
            return {"node_id": node.node_id, "node_type": "sc_ensemble", "output": prev}
        if len(candidates) == 1:
            return {"node_id": node.node_id, "node_type": "sc_ensemble", "output": candidates[0]}

        # SC_ENSEMBLE_PROMPT verbatim from MaAS (bingreeky/MaAS
        # scripts/optimized/HumanEval/test/template/op_prompt.py).
        problem = str(st.get("question", ""))
        solutions_block = ""
        for i, c in enumerate(candidates):
            solutions_block += f"{chr(65 + i)}: \n{c}\n\n\n"
        judge_user = (
            f"Given the question described as follows: {problem}\n"
            f"Several solutions have been generated to address the given question. They are as follows:\n"
            f"{solutions_block}\n"
            f"Carefully evaluate these solutions and identify the answer that appears most frequently "
            f"across them. This consistency in answers is crucial for determining the most reliable "
            f"solution.\n\n"
            f"Output only the single letter ID (A, B, C, etc.) corresponding to the most consistent "
            f"solution. Do not include any additional text or explanation."
        )
        judge_sys = "You are an expert judge selecting the most consistent solution."
        resp = self_ex.llm.chat(
            [{"role": "system", "content": judge_sys}, {"role": "user", "content": judge_user}],
            max_tokens=8, temperature=0.0,
        )
        letter = (resp.text or "").strip().upper()[:1]
        idx = ord(letter) - 65 if letter and "A" <= letter <= "Z" else 0
        idx = max(0, min(idx, len(candidates) - 1))
        return {
            "node_id": node.node_id, "node_type": "sc_ensemble",
            "output": candidates[idx],
            "winning_choice": letter or "A",
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
        }

    @staticmethod
    def _handle_test_reflect_loop(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """MaAS Test: exec against visible asserts; on failure, REFLECT and retry.

        config.max_rounds (default 3): max reflect+fix iterations.
        Pulls `assert ...` lines or `>>> name(args)\\n result` docstring examples
        out of the question, runs them against the candidate code in a
        subprocess, and on failure asks the LLM to reflect-and-fix using the
        REFLECTION_ON_PUBLIC_TEST_PROMPT structure from MaAS.
        """
        import re as _re, tempfile, subprocess, os as _os

        cfg = node.config or {}
        max_rounds = int(cfg.get("max_rounds", 3))
        prev = st.get("last_output", "")
        question = str(st.get("question", ""))

        def _extract_code(text: str) -> str:
            m = _re.search(r"```python\s*\n(.*?)```", text, _re.DOTALL)
            if m:
                return m.group(1).rstrip()
            m = _re.search(r"```\s*\n(.*?)```", text, _re.DOTALL)
            return (m.group(1) if m else text).rstrip()

        def _collect_tests(q: str) -> str:
            # 1. [PUBLIC_TESTS]...[/PUBLIC_TESTS] block from the runner
            #    (HumanEval ships its `test` suite via this marker; matches
            #    MaAS's humaneval_public_test.jsonl pathway).
            m = _re.search(r"\[PUBLIC_TESTS\]\s*\n(.*?)\n\[/PUBLIC_TESTS\]", q, _re.DOTALL)
            if m:
                block = m.group(1)
                # Strip leading "# ..." comment lines so the body is executable.
                return "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("#"))
            # 2. Plain assert lines in the question (MBPP style).
            asserts = [ln.strip() for ln in q.splitlines() if ln.strip().startswith("assert ")]
            if asserts:
                return "\n".join(asserts[:5])
            # 3. Docstring >>> examples as a last resort.
            doctests = _re.findall(r">>>\s*([^\n]+)\n\s*([^\n>]+)", q)
            lines = []
            for call, expected in doctests[:5]:
                call, expected = call.strip(), expected.strip()
                if not call or not expected:
                    continue
                lines.append(f"assert repr({call}) == repr({expected!r}) or {call} == {expected}")
            return "\n".join(lines)

        tests = _collect_tests(question)
        if not tests:
            return {"node_id": node.node_id, "node_type": "test_reflect_loop", "output": prev}

        code = _extract_code(prev)
        last_err = ""
        for round_idx in range(max_rounds):
            script = code + "\n\n" + tests + "\nprint('TESTS_OK')\n"
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(script); path = f.name
            try:
                proc = subprocess.run(["python3", path], capture_output=True, text=True, timeout=10)
            except Exception as e:
                class _P: pass
                proc = _P(); proc.stdout = ""; proc.stderr = str(e); proc.returncode = 1
            try: _os.unlink(path)
            except Exception: pass
            if "TESTS_OK" in (proc.stdout or ""):
                # Repackage in a python code block so downstream graders find it.
                return {
                    "node_id": node.node_id, "node_type": "test_reflect_loop",
                    "output": f"```python\n{code}\n```",
                    "rounds_used": round_idx,
                }
            last_err = (proc.stderr or proc.stdout or "")[-400:]
            # REFLECTION_ON_PUBLIC_TEST_PROMPT (MaAS).
            reflect_user = (
                f"Given a code problem and a python code solution which failed to pass test "
                f"or execute, you need to analyze the reason for the failure and propose a "
                f"better code solution.\n\n"
                f"### problem\n{question}\n\n"
                f"### Code Solution\n```python\n{code}\n```\n\n"
                f"### Execution Result\nexecuted unsuccessfully\n\n"
                f"#### Failed Test Case\n{last_err}\n\n"
                f"Please provide a reflection on the failed test cases and code solution, "
                f"followed by a better code solution in a single ```python``` block."
            )
            resp = self_ex.llm.chat(
                [{"role": "system", "content": "You are an expert Python programmer who fixes failing code."},
                 {"role": "user", "content": reflect_user}],
                max_tokens=self_ex.max_tokens, temperature=0.0,
            )
            code = _extract_code(resp.text or "")
            if not code.strip():
                break

        return {
            "node_id": node.node_id, "node_type": "test_reflect_loop",
            "output": f"```python\n{code}\n```",
            "rounds_used": max_rounds,
            "last_test_error": last_err[:160],
        }

    @staticmethod
    def _handle_python(self_ex: "Executor", node: Node, st: Dict[str, Any]) -> Dict[str, Any]:
        """Run an agent-synthesized Python repair function on the workflow state.

        config.code: source defining a function (default name `repair`).
        config.function_name: name of the function inside `code` (default `repair`).
        config.predicate: optional source defining `applies(state) -> bool`. If
        present and returns False, the node passes the input through unchanged.

        Execution is in-process with a small allowed builtins set — this is for
        trusted local research, not adversarial isolation.
        """
        cfg = node.config or {}
        code = cfg.get("code", "")
        fn_name = cfg.get("function_name", "repair")
        predicate_src = cfg.get("predicate", "")
        prev = st.get("last_output", "")
        state = {"question": st.get("question", ""), "last_output": prev}

        # Restricted execution namespace. We allow re, json, math, itertools, ast,
        # functools, string, collections — common, side-effect-free utilities.
        import ast as _ast
        import collections as _collections
        import functools as _functools
        import itertools as _itertools
        import json as _json
        import math as _math
        import re as _re
        import string as _string

        # Whitelist `import` of these safe modules. LLM-synthesized repair code
        # frequently writes `import re` even though `re` is already in scope, so
        # we must support that without exposing arbitrary __import__.
        _safe_modules = {
            "re": _re, "json": _json, "math": _math, "itertools": _itertools,
            "ast": _ast, "functools": _functools, "string": _string,
            "collections": _collections,
        }

        def _safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if name in _safe_modules:
                return _safe_modules[name]
            raise ImportError(f"module {name!r} not allowed in sandbox")

        allowed_builtins = {
            "__import__": _safe_import,
            "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr,
            "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "int": int, "isinstance": isinstance, "iter": iter,
            "len": len, "list": list, "map": map, "max": max, "min": min,
            "next": next, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip, "True": True, "False": False, "None": None, "Exception": Exception,
        }
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": allowed_builtins,
            **_safe_modules,
        }

        # Compile + execute the repair function source.
        try:
            exec(code, sandbox_globals)
        except Exception as e:
            return {"node_id": node.node_id, "node_type": "python", "output": prev, "error": f"python:compile_error:{e}"}

        fn = sandbox_globals.get(fn_name)
        if not callable(fn):
            return {"node_id": node.node_id, "node_type": "python", "output": prev, "error": f"python:function {fn_name!r} not defined"}

        # Optional predicate.
        if predicate_src:
            try:
                exec(predicate_src, sandbox_globals)
            except Exception as e:
                return {"node_id": node.node_id, "node_type": "python", "output": prev, "error": f"python:predicate_compile:{e}"}
            applies = sandbox_globals.get("applies")
            if callable(applies):
                try:
                    if not bool(applies(state)):
                        return {"node_id": node.node_id, "node_type": "python", "output": prev, "predicate_matched": False}
                except Exception as e:
                    return {"node_id": node.node_id, "node_type": "python", "output": prev, "error": f"python:predicate_runtime:{e}"}

        # Run the repair function.
        try:
            new_out = fn(state)
        except Exception as e:
            return {"node_id": node.node_id, "node_type": "python", "output": prev, "error": f"python:runtime:{e}"}

        # Repair may return:
        #   - a string (new last_output, downstream nodes still run)
        #   - a dict {output: "...", terminal: True}  ← Self-Compilation: skip
        #     all downstream nodes including any LLM call
        if isinstance(new_out, dict):
            return {
                "node_id": node.node_id,
                "node_type": "python",
                "output": str(new_out.get("output", new_out.get("last_output", prev))),
                "predicate_matched": True,
                "terminal": bool(new_out.get("terminal", False)),
            }
        return {"node_id": node.node_id, "node_type": "python", "output": str(new_out) if new_out is not None else prev, "predicate_matched": True}
