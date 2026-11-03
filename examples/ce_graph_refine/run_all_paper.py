"""CE-Graph reference benchmark driver.

Supports math and code tasks, a BBH reasoning proxy, and AIME 2025.
See docs/REPRODUCTION.md for dataset slices and evaluation limitations.
"""
from __future__ import annotations
import argparse, json, os, sys, time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

# Configure OPENROUTER_API_KEY in your shell; credentials are never bundled.
os.environ.setdefault("OPENROUTER_MODEL", "openai/gpt-4o-mini")

from verl.ce_graph import (
    CounterexamplePool, FailureClustering, LLMProposer, Node, Workflow, refine_workflow,
    EmbeddingGMMClustering, make_openai_embed_fn,
)
from verl.ce_graph.openrouter_llm import OpenRouterClient


BENCHMARKS = ("gsm8k", "math", "multiarith", "humaneval", "mbpp", "gaia", "aime_2025")

# ============ per-benchmark workflow + loader + runner dispatch ============
def get_dispatch(benchmark: str):
    if benchmark == "gsm8k":
        from examples.ce_graph_refine.gsm8k_runner import GSM8KTaskRunner, load_gsm8k
        sys_p = ("You are an expert grade-school math tutor. Solve the problem step by step. "
                  "Show the arithmetic explicitly. End your answer on its own line in the format: "
                  "#### <integer>.")
        usr_p = "{prefix}Question: {question}\nSolve carefully. End with the line: #### <integer answer>."
        return load_gsm8k, GSM8KTaskRunner, sys_p, usr_p
    if benchmark == "math":
        from examples.ce_graph_refine.math_runner import MATHTaskRunner, load_math
        sys_p = ("You are an expert math problem solver. Solve the problem step by step. "
                  "Reason carefully and show all work. Write the final answer at the end in the form "
                  "'Answer: \\boxed{X}'.")
        usr_p = "{prefix}Problem: {question}\nSolve. End with: Answer: \\boxed{...}"
        return load_math, MATHTaskRunner, sys_p, usr_p
    if benchmark == "multiarith":
        from examples.ce_graph_refine.multiarith_runner import MultiArithTaskRunner, load_multiarith
        sys_p = ("You are a careful arithmetic solver. Solve the problem step by step. "
                  "End your answer on its own line in the format: #### <integer>.")
        usr_p = "{prefix}Question: {question}\nSolve carefully. End with: #### <integer answer>."
        return load_multiarith, MultiArithTaskRunner, sys_p, usr_p
    if benchmark == "humaneval":
        from examples.ce_graph_refine.humaneval_runner import HumanEvalTaskRunner, load_humaneval
        sys_p = ("You are an expert Python programmer. Given a function signature and docstring, write the "
                  "complete function implementation. Return only the function body and any helper functions, "
                  "wrapped in a single ```python code block. The function must satisfy all docstring examples.")
        usr_p = ("{prefix}Complete the following Python function. Output ONLY the full function definition in a "
                  "```python code block, nothing else.\n\n{question}")
        return load_humaneval, HumanEvalTaskRunner, sys_p, usr_p
    if benchmark == "mbpp":
        from examples.ce_graph_refine.mbpp_runner import MBPPTaskRunner, load_mbpp
        sys_p = ("You are an expert Python programmer. Write a Python function that satisfies the prompt and "
                  "passes the given assertions. Return only the function definition in a ```python block.")
        usr_p = "{prefix}{question}"
        return load_mbpp, MBPPTaskRunner, sys_p, usr_p
    if benchmark == "gaia":
        from examples.ce_graph_refine.gaia_runner import GAIATaskRunner, load_gaia_proxy
        sys_p = ("You are an expert AI assistant. Solve the user's task step by step. Plan, reason, and answer. "
                  "Final answer must be the exact string the question asks for.")
        usr_p = "{prefix}Task: {question}\nReason step by step. Final answer (exactly):"
        return load_gaia_proxy, GAIATaskRunner, sys_p, usr_p
    if benchmark == "aime_2025":
        from examples.ce_graph_refine.aime2025_runner import AIME2025TaskRunner, load_aime_2025
        sys_p = ("You are an expert competition mathematician. The answer to AIME problems is an integer 0-999. "
                  "Reason carefully step by step. End your answer with: #### N")
        usr_p = "{prefix}Problem (AIME 2025): {question}\nReason carefully, then conclude with: #### <integer 0-999>"
        return load_aime_2025, AIME2025TaskRunner, sys_p, usr_p
    raise ValueError(f"unknown benchmark {benchmark}")


def build_baseline_workflow(name: str, system: str, user_template: str,
                            strong_for_code: bool = False,
                            multi_agent: bool = False,
                            sandbox_retry: bool = False,
                            maas: bool = False) -> Workflow:
    """W0 baseline.

    maas=True for code tasks: build the actual MaAS-aligned pipeline
      MultiGenerateCoT → ScEnsemble → Test (reflection loop, max 3 rounds)
    matching the operator library in bingreeky/MaAS HumanEval/MBPP graph. This
    is an independently reconstructed workflow, not a guaranteed paper score.

    multi_agent=True is the older planner→coder→reviewer pipeline (deprecated;
    reviewer over-revised correct code and dropped W0 vs CoT).
    """
    w = Workflow(metadata={"name": name, "version": "W0"})
    if maas and name in ("humaneval", "mbpp"):
        # MaAS-style W0 — reproduces the operator library used in
        # bingreeky/MaAS HumanEval test graph.
        gen_sys = ("You are an expert Python programmer. Solve the problem with chain-of-thought "
                   "reasoning, then output the complete function definition in a single ```python "
                   "code block. The function must satisfy all visible asserts / docstring examples.")
        gen_user = ("Complete the following Python function. Think step by step about edge cases, "
                    "then output ONLY the full function definition in a ```python code block.\n\n{question}")
        w.add_node(Node(node_id="multi_gen", node_type="multi_generate_cot", config={
            "k": 3, "temperature": 0.5,
            "system": gen_sys, "user_template": gen_user,
        }))
        w.add_node(Node(node_id="sc_pick", node_type="sc_ensemble", config={}))
        w.add_node(Node(node_id="test_loop", node_type="test_reflect_loop", config={"max_rounds": 3}))
        w.add_edge("multi_gen", "sc_pick")
        w.add_edge("sc_pick", "test_loop")
        return w
    if multi_agent and name in ("humaneval", "mbpp"):
        # ---- Agent 1: planner ----
        planner_sys = ("You are an expert software architect. Read the function specification carefully, "
                       "then write a concise step-by-step plan for implementing it. "
                       "Cover: (1) what the function should compute, (2) edge cases to handle "
                       "(empty inputs, single-element, negative, zero, boundary), (3) algorithm sketch. "
                       "Do NOT write code. Output the plan as a numbered list.")
        planner_user = ("{prefix}Function specification:\n{question}\n\nWrite the implementation plan.")
        w.add_node(Node(node_id="planner", node_type="llm",
                         config={"system": planner_sys, "user_template": planner_user, "prompt_prefix": ""}))
        # ---- Agent 2: coder ----
        coder_sys = ("You are an expert Python programmer. Implement the function precisely according to the plan. "
                     "Return ONLY the function definition (and any necessary helper functions) in a single "
                     "```python code block. Do not include test calls or print statements.")
        coder_user = ("{prefix}Function specification:\n{question}\n\n"
                      "Implementation plan from architect:\n{last_output}\n\n"
                      "Write the complete function definition in a ```python block.")
        w.add_node(Node(node_id="coder", node_type="llm",
                         config={"system": coder_sys, "user_template": coder_user, "prompt_prefix": ""}))
        # ---- Agent 3: reviewer (defensive: only revises if a real bug is found) ----
        reviewer_sys = ("You are an expert code reviewer. Examine the candidate function for bugs ONLY. "
                        "If the code is correct, output the original code unchanged. "
                        "Only if you find a clear bug (wrong logic, missed edge case, type error), "
                        "output a corrected complete function in a single ```python block. "
                        "Do NOT make stylistic changes.")
        reviewer_user = ("{prefix}Function specification:\n{question}\n\n"
                         "Candidate implementation:\n{last_output}\n\n"
                         "Output either the original code (if correct) or a bug-fixed version. "
                         "Wrap the final code in a ```python block.")
        w.add_node(Node(node_id="reviewer", node_type="llm",
                         config={"system": reviewer_sys, "user_template": reviewer_user, "prompt_prefix": ""}))
        return w
    # default: simple CoT
    w.add_node(Node(node_id="cot", node_type="llm",
                    config={"system": system, "user_template": user_template, "prompt_prefix": ""}))
    if sandbox_retry and name in ("humaneval", "mbpp"):
        # Add sandbox_test_retry: runs visible assertions, retries via LLM if fail.
        w.add_node(Node(node_id="sandbox_retry", node_type="sandbox_test_retry", config={}))
    if strong_for_code and name in ("humaneval", "mbpp"):
        w.add_node(Node(node_id="reflection_check", node_type="reflection", config={
            "instruction": (
                "Re-examine the drafted code. If you find a real bug, return a corrected complete function "
                "in a single ```python code block. If the code looks correct, return it unchanged."
            ),
        }))
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                    choices=BENCHMARKS)
    p.add_argument("--train-n", type=int, default=100)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--n-proposals", type=int, default=6)
    p.add_argument("--verify-k", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--strong-baseline", action="store_true",
                   help="Use multi-step baseline (CoT + reflection) for code tasks — mimics paper's MaAS-level W0")
    p.add_argument("--multi-agent", action="store_true",
                   help="Use MaAS-style multi-agent W0 (planner→coder→reviewer) for code tasks")
    p.add_argument("--sandbox-retry", action="store_true",
                   help="Add sandbox_test_retry node (run visible asserts, LLM-fix on fail) for code tasks")
    p.add_argument("--maas", action="store_true",
                   help="Use MaAS-aligned W0 (MultiGenerateCoT k=3 -> ScEnsemble -> Test-Reflect-loop) for code tasks")
    p.add_argument("--expose-public-tests", action="store_true",
                   help="Diagnostic only: show grading tests to the model; not held-out benchmark accuracy")
    p.add_argument("--save-report", default="")
    args = p.parse_args()

    for key in ("train_n", "val_n", "n_workers", "max_tokens", "n_proposals", "verify_k", "patience"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.seed_offset < 0 or args.max_iters < 0:
        p.error("--seed-offset and --max-iters must be nonnegative")
    if args.expose_public_tests and args.benchmark not in ("humaneval", "mbpp"):
        p.error("--expose-public-tests applies only to HumanEval and MBPP")
    if not os.environ.get("OPENROUTER_API_KEY"):
        p.error("Set OPENROUTER_API_KEY before running API-backed benchmarks; use the offline demo without a key")

    load_fn, Runner, sys_p, usr_p = get_dispatch(args.benchmark)

    total = args.train_n + args.val_n + args.seed_offset
    all_examples = load_fn(n=total)
    if len(all_examples) < total:
        p.error(f"Requested {total} examples including offset, but the loader returned {len(all_examples)}; reduce the slice sizes")
    off = args.seed_offset
    train_examples = all_examples[off : off + args.train_n]
    val_examples = all_examples[off + args.train_n : off + args.train_n + args.val_n]
    print(f"[{args.benchmark}] loaded train={len(train_examples)} val={len(val_examples)} total={len(all_examples)}")

    llm = OpenRouterClient(model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    common = dict(llm=llm, max_tokens=args.max_tokens, temperature=0.0, n_workers=args.n_workers)
    # Architecture and test visibility are independent choices. Full scoring
    # tests are never shown to the model unless explicitly requested.
    if args.expose_public_tests:
        print("WARNING: test-visible diagnostic run; do not report as held-out benchmark accuracy")
        common["expose_public_tests"] = True
    train_runner = Runner(examples=train_examples, **common)
    val_runner = Runner(examples=val_examples, **common)

    w0 = build_baseline_workflow(args.benchmark, sys_p, usr_p,
                                  strong_for_code=args.strong_baseline,
                                  multi_agent=args.multi_agent,
                                  sandbox_retry=args.sandbox_retry,
                                  maas=args.maas)

    print(f"\n[{args.benchmark}] === baseline W0 on val ===")
    t0 = time.time()
    base_score, _ = val_runner(w0)
    base_time = time.time() - t0
    print(f"[{args.benchmark}] W0 val acc = {base_score:.4f}  ({base_time:.1f}s)")

    print(f"\n[{args.benchmark}] === CE-Graph refinement (paper-aligned) ===")
    proposer = LLMProposer(llm=llm, n_candidates=args.n_proposals)
    _eb = os.environ.get('EMBED_BACKEND', 'openrouter')
    if _eb == 'openai':
        embed_fn = make_openai_embed_fn(model=os.environ.get('EMBED_MODEL','text-embedding-3-small'), base_url=os.environ.get('EMBED_BASE_URL','https://api.openai.com/v1/embeddings'))
    elif _eb == 'local':
        from examples.ce_graph_refine._local_embed import local_embed_fn
        embed_fn = local_embed_fn
    else:
        embed_fn = make_openai_embed_fn()  # OpenRouter text-embedding-3-small (paper-faithful default)
    clustering = EmbeddingGMMClustering(embed_fn=embed_fn, max_modes=6, min_cluster_size=2, min_for_gmm=5)
    pool = CounterexamplePool(capacity=2000)

    t0 = time.time()
    w_star, reports = refine_workflow(
        workflow=w0, runner=train_runner, llm=llm, proposer=proposer,
        pool=pool, clustering=clustering,
        max_iters=args.max_iters, n_proposals=args.n_proposals,
        verify_k=args.verify_k, patience=args.patience, verbose=True,
    )
    refine_time = time.time() - t0

    print(f"\n[{args.benchmark}] === refined W* on val ===")
    t0 = time.time()
    star_score, _ = val_runner(w_star)
    star_time = time.time() - t0
    print(f"[{args.benchmark}] W* val acc = {star_score:.4f}  ({star_time:.1f}s)")

    print(f"\n[{args.benchmark}] === Summary ===")
    print(f"  W0 val acc:  {base_score:.4f}")
    print(f"  W* val acc:  {star_score:.4f}   Δ {star_score-base_score:+.4f}")
    print(f"  refinement wall: {refine_time:.1f}s")

    if args.save_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_report)), exist_ok=True)
        json.dump({
            "benchmark": args.benchmark,
            "args": vars(args),
            "base_score": base_score, "star_score": star_score,
            "delta": star_score - base_score,
            "refine_time_s": refine_time,
            "base_eval_time_s": base_time,
            "star_eval_time_s": star_time,
            "n_iters": len(reports),
            "evaluation_protocol": "test-visible-diagnostic" if args.expose_public_tests else "scoring-tests-hidden",
            "model": llm.model,
            "embedding_backend": _eb,
            "train_ids": [ex.qid for ex in train_examples],
            "val_ids": [ex.qid for ex in val_examples],
            "workflow": w_star.to_dict(),
            "refinement_reports": [vars(report) for report in reports],
        }, open(args.save_report, "w"), indent=2)


if __name__ == "__main__":
    main()
