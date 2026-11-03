"""CE-Graph refinement loop (paper §3, Algorithm 1).

This implements:
  1. Sample failure distribution p(s | W_t): run the runner, collect failures.
  2. Gradient step: cluster failures, pick the densest mode b*_t.
  3. Optimal edit: ask the Proposer for N candidate edits, then estimate
     V(Δ_i) by sampling K instances from b*_t (Monte Carlo verification, Eq. 5).
  4. Commit if it improves validation score; otherwise count toward patience.
  5. Stop on convergence (no failures, no improvement over `patience` rounds,
     or `max_iters`).

Two refinement modes are supported:
  - **Proposer mode** (default when `llm` or `proposer` is provided): mirrors the
    paper's LLM-based Propose-and-Verify.
  - **Operator library mode** (legacy, for the offline toy demo): tries each
    applicable operator from a static library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .code_synth_proposer import CodeSynthProposer
from .counterexamples import CounterexamplePool, ExecutionTrace
from .failures import FailureClustering, FailureMode
from .llm import AIDPClient
from .operators import OperatorLibrary
from .proposer import Edit, LLMProposer, apply_edit
from .workflow import Workflow


Runner = Callable[[Workflow], Tuple[float, List[ExecutionTrace]]]


@dataclass
class RefinementReport:
    iteration: int
    base_score: float
    candidate_score: float
    accepted: bool
    chosen_mode_size: int = 0
    chosen_mode_keywords: List[str] = field(default_factory=list)
    chosen_edit: str = ""
    edit_rationale: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Candidate:
    display: str
    workflow: Workflow
    rationale: str
    verify_score: float = 0.0
    regression_score: float = 1.0
    guard_regression_score: float = 1.0


def refine_workflow(
    workflow: Workflow,
    runner: Runner,
    *,
    llm: Optional[AIDPClient] = None,
    proposer: Optional[LLMProposer] = None,
    operator_lib: Optional[OperatorLibrary] = None,
    pool: Optional[CounterexamplePool] = None,
    clustering: Optional[FailureClustering] = None,
    max_iters: int = 20,
    n_proposals: int = 4,
    verify_k: int = 4,
    patience: int = 3,
    accept_if_improves: bool = True,
    verbose: bool = True,
    # --- TTR additions -----------------------------------------------------
    use_code_synth: bool = False,
    regression_check_k: int = 0,
    regression_threshold: float = 0.95,
    guard_runner: Optional[Runner] = None,
) -> Tuple[Workflow, List[RefinementReport]]:
    """Run the failure-driven refinement loop.

    TTR additions:
      use_code_synth: if True, use the code-synthesis proposer (can emit
        `InsertPythonNode` edits in addition to prompt/template edits).
      regression_check_k: number of previously-passing examples to verify each
        candidate against. If > 0 and the candidate's pass rate on these is
        below `regression_threshold`, the candidate is rejected even if its
        local V is high. Prevents train→val overfitting (HumanEval -6% issue).
    """

    pool = pool or CounterexamplePool(capacity=5000)
    clustering = clustering or FailureClustering()

    use_proposer = proposer is not None or llm is not None
    if use_proposer and proposer is None:
        if use_code_synth:
            proposer = CodeSynthProposer(llm=llm, n_candidates=n_proposals)  # type: ignore[arg-type]
        else:
            proposer = LLMProposer(llm=llm, n_candidates=n_proposals)  # type: ignore[arg-type]
    if not use_proposer and operator_lib is None:
        raise ValueError("Provide either `llm`/`proposer` (Proposer mode) or `operator_lib` (legacy mode).")

    reports: List[RefinementReport] = []
    base_score, traces = runner(workflow)
    pool.extend(traces)
    if verbose:
        print(f"[iter 0] baseline score={base_score:.4f}  failures_in_pool={len(pool)}")

    verify_fn = getattr(runner, "verify", None)
    # IDs of train examples currently passing under `workflow`. Used by the
    # in-distribution regression check.
    current_pass_ids: List[str] = [t.instance_id for t in traces if t.is_success]

    # Optional held-out **guard** set: a separate slice (disjoint from both
    # train and val) used to detect train→val distribution drift before commit.
    # When provided, the guard check is much stronger evidence that an edit
    # generalizes than the in-distribution train regression check.
    guard_verify_fn = getattr(guard_runner, "verify", None) if guard_runner is not None else None
    guard_pass_ids: List[str] = []
    if guard_runner is not None:
        g_score, g_traces = guard_runner(workflow)
        guard_pass_ids = [t.instance_id for t in g_traces if t.is_success]
        if verbose:
            print(f"[iter 0] guard score={g_score:.4f}  guard_pass_set_size={len(guard_pass_ids)}")

    no_improve = 0
    import random as _r

    for it in range(1, max_iters + 1):
        failed = pool.all()
        if not failed:
            if verbose:
                print(f"[iter {it}] no failures — stopping")
            break

        modes, _labels = clustering.cluster(failed)
        if not modes:
            if verbose:
                print(f"[iter {it}] no failure modes — stopping")
            break

        modes.sort(key=lambda m: m.size, reverse=True)
        chosen_mode = modes[0]
        exemplar_traces = [failed[i] for i in chosen_mode.exemplar_indices[: max(verify_k, 4)]]
        exemplar_ids = [t.instance_id for t in exemplar_traces]
        if verbose:
            print(
                f"[iter {it}] dominant mode size={chosen_mode.size} "
                f"keywords={chosen_mode.keywords[:4]}"
            )

        candidates = _build_candidates(
            workflow=workflow,
            mode=chosen_mode,
            exemplar_traces=exemplar_traces,
            n=n_proposals,
            proposer=proposer if use_proposer else None,
            operator_lib=operator_lib,
            verbose=verbose,
        )

        if not candidates:
            no_improve += 1
            reports.append(
                RefinementReport(
                    iteration=it,
                    base_score=base_score,
                    candidate_score=base_score,
                    accepted=False,
                    chosen_mode_size=chosen_mode.size,
                    chosen_mode_keywords=chosen_mode.keywords,
                    notes={"reason": "no_applicable_edits"},
                )
            )
            if no_improve >= patience:
                if verbose:
                    print(f"[iter {it}] patience exhausted")
                break
            continue

        # --- Verify (Monte Carlo on the failure cluster) -----------------
        # Sample for the in-distribution regression check (train-side).
        regression_sample: List[str] = []
        if regression_check_k > 0 and callable(verify_fn) and current_pass_ids:
            k = min(regression_check_k, len(current_pass_ids))
            regression_sample = _r.sample(current_pass_ids, k)
        # Sample for the held-out guard regression check (val-distribution).
        guard_sample: List[str] = []
        if regression_check_k > 0 and callable(guard_verify_fn) and guard_pass_ids:
            k = min(regression_check_k, len(guard_pass_ids))
            guard_sample = _r.sample(guard_pass_ids, k)

        for cand in candidates:
            if callable(verify_fn) and exemplar_ids:
                cand.verify_score = verify_fn(cand.workflow, exemplar_ids[:verify_k])
            else:
                cand.verify_score, _ = runner(cand.workflow)
            # Train regression check.
            if regression_sample and callable(verify_fn):
                reg = verify_fn(cand.workflow, regression_sample)
                cand.regression_score = reg
            # Guard (val-distribution) regression check.
            guard_reg = 1.0
            if guard_sample and callable(guard_verify_fn):
                guard_reg = guard_verify_fn(cand.workflow, guard_sample)
                cand.guard_regression_score = guard_reg
            if verbose:
                parts = [f"V={cand.verify_score:.3f}"]
                if regression_sample:
                    parts.append(f"reg={cand.regression_score:.3f}")
                if guard_sample:
                    parts.append(f"guard={guard_reg:.3f}")
                print(f"  verify {cand.display}  " + "  ".join(parts))

        # Filter out candidates that fail either regression bar (if enabled).
        if regression_sample or guard_sample:
            def _passes(c: "_Candidate") -> bool:
                if regression_sample and c.regression_score < regression_threshold:
                    return False
                if guard_sample and c.guard_regression_score < regression_threshold:
                    return False
                return True
            survivors = [c for c in candidates if _passes(c)]
            if not survivors:
                if verbose:
                    print(f"  all candidates fail regression>= {regression_threshold} (train or guard) — skipping")
                no_improve += 1
                reports.append(
                    RefinementReport(
                        iteration=it,
                        base_score=base_score,
                        candidate_score=base_score,
                        accepted=False,
                        chosen_mode_size=chosen_mode.size,
                        chosen_mode_keywords=chosen_mode.keywords,
                        notes={"reason": "regression_fail",
                               "best_reg": round(max(c.regression_score for c in candidates), 4)},
                    )
                )
                if no_improve >= patience:
                    if verbose:
                        print(f"[iter {it}] patience exhausted")
                    break
                continue
            survivors.sort(key=lambda c: c.verify_score, reverse=True)
            best = survivors[0]
        else:
            candidates.sort(key=lambda c: c.verify_score, reverse=True)
            best = candidates[0]

        # --- Commit (full validation eval) ------------------------------
        cand_score, cand_traces = runner(best.workflow)
        accepted = (cand_score > base_score) if accept_if_improves else True
        if accepted:
            workflow = best.workflow
            base_score = cand_score
            no_improve = 0
            pool = CounterexamplePool(capacity=pool.capacity)
            pool.extend(cand_traces)
            # Refresh both regression sets under the new workflow.
            current_pass_ids = [t.instance_id for t in cand_traces if t.is_success]
            if guard_runner is not None:
                g_score_new, g_traces_new = guard_runner(workflow)
                guard_pass_ids = [t.instance_id for t in g_traces_new if t.is_success]
                if verbose:
                    print(f"  guard re-evaluated: score={g_score_new:.4f}  pass_set_size={len(guard_pass_ids)}")
        else:
            no_improve += 1

        reports.append(
            RefinementReport(
                iteration=it,
                base_score=base_score,
                candidate_score=cand_score,
                accepted=accepted,
                chosen_mode_size=chosen_mode.size,
                chosen_mode_keywords=chosen_mode.keywords,
                chosen_edit=best.display,
                edit_rationale=best.rationale,
                notes={"verify_score": round(best.verify_score, 4)},
            )
        )

        if verbose:
            mark = "ACCEPT" if accepted else "reject"
            print(
                f"[iter {it}] {mark}  edit={best.display}  V={best.verify_score:.3f}  "
                f"cand={cand_score:.4f} base={base_score:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"[iter {it}] convergence (no_improve={no_improve}>=patience={patience})")
            break

    return workflow, reports


def _build_candidates(
    *,
    workflow: Workflow,
    mode: FailureMode,
    exemplar_traces: List[ExecutionTrace],
    n: int,
    proposer: Optional[LLMProposer],
    operator_lib: Optional[OperatorLibrary],
    verbose: bool,
) -> List[_Candidate]:
    """Produce up to `n` candidate workflows for verification."""

    out: List[_Candidate] = []

    if proposer is not None:
        edits: List[Edit] = proposer.propose(workflow, mode, exemplar_traces)
        for e in edits:
            try:
                cw = apply_edit(workflow, e)
            except Exception as ex:
                if verbose:
                    print(f"  edit {e.short()} not applicable: {ex}")
                continue
            out.append(_Candidate(display=e.short(), workflow=cw, rationale=e.rationale))
            if verbose:
                print(f"  proposed[{len(out)-1}] {e.short()}  | {e.rationale[:80]}")
    elif operator_lib is not None:
        ops = operator_lib.propose(workflow, mode, top_k=n)
        for op in ops:
            try:
                cw = op.apply(workflow, mode)
            except Exception as ex:
                if verbose:
                    print(f"  legacy op {op.name} not applicable: {ex}")
                continue
            out.append(_Candidate(display=f"legacy:{op.name}", workflow=cw, rationale=f"legacy operator {op.name}"))
            if verbose:
                print(f"  proposed[{len(out)-1}] legacy:{op.name}")

    return out[:n]
