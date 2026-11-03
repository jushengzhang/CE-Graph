# Reproduction guide

This is the CE-Graph reference release accompanying [Failure-Driven Workflow Refinement](https://proceedings.mlr.press/v306/zhang26f.html). The supplied code reconstructs the core loop and a MaAS-style baseline; it does not bundle the original experiment environment, checkpoints, datasets, or measured result logs.

## Installation and offline checks

From a fresh virtual environment:

```bash
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
python -m examples.ce_graph_refine.offline_demo
python -m examples.ce_graph_refine.run_all_paper --help
```

The offline demo uses a deterministic synthetic task. It exercises failure collection, mode identification, candidate verification, and acceptance without an API key. It is not an accuracy experiment. The test suite runs without API credentials; `bash scripts/check.sh` runs the complete local check sequence.

## API-backed runs

Export your own API key; do not put it in source or committed configuration:

```bash
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export EMBED_BACKEND="local"     # Hashing features; no remote embedding calls
# For semantic embeddings instead:
# export EMBED_BACKEND="openrouter"
```

Commands below are starting configurations, not a claim of matching the published table. API requests may incur provider charges.

```bash
python -m examples.ce_graph_refine.run_all_paper --benchmark gsm8k \
  --train-n 50 --val-n 200 --max-iters 8 --n-proposals 4 --verify-k 6 \
  --patience 2 --n-workers 8 --max-tokens 2048 --save-report runs/gsm8k.json

python -m examples.ce_graph_refine.run_all_paper --benchmark math \
  --train-n 100 --val-n 200 --max-iters 10 --n-proposals 4 --verify-k 10 \
  --patience 3 --n-workers 8 --max-tokens 2048 --save-report runs/math.json

python -m examples.ce_graph_refine.run_all_paper --benchmark multiarith \
  --train-n 50 --val-n 100 --max-iters 10 --n-proposals 4 --verify-k 10 \
  --patience 3 --n-workers 8 --max-tokens 2048 --save-report runs/multiarith.json

python -m examples.ce_graph_refine.run_all_paper --benchmark humaneval --maas \
  --train-n 50 --val-n 114 --max-iters 10 --n-proposals 4 --verify-k 10 \
  --patience 3 --n-workers 4 --max-tokens 2048 --save-report runs/humaneval.json

python -m examples.ce_graph_refine.run_all_paper --benchmark mbpp --maas \
  --train-n 100 --val-n 157 --max-iters 10 --n-proposals 4 --verify-k 10 \
  --patience 3 --n-workers 4 --max-tokens 2048 --save-report runs/mbpp.json

# BBH reasoning proxy, not the GAIA benchmark:
python -m examples.ce_graph_refine.run_all_paper --benchmark gaia \
  --train-n 30 --val-n 50 --max-iters 10 --n-proposals 4 --verify-k 10 \
  --patience 3 --n-workers 4 --max-tokens 2048 --save-report runs/bbh-proxy.json

# Small AIME extension; the two source sets contain 30 problems in total:
python -m examples.ce_graph_refine.run_all_paper --benchmark aime_2025 \
  --train-n 10 --val-n 20 --max-iters 2 --n-proposals 2 --verify-k 4 \
  --patience 2 --n-workers 2 --max-tokens 2048 --save-report runs/aime2025.json
```

## Data splits and test visibility

The driver selects disjoint contiguous refinement and evaluation slices from the chosen loader split. `--seed-offset` shifts the starting index; it is **not** a random-number seed. The release currently defaults loaders to their test split (AIME uses its available train split). If your protocol needs official train/test separation, supply separate runners and loaders rather than treating this default as such a protocol.

`--maas` builds the reconstructed MultiGenerateCoT → ScEnsemble → Test/Reflect workflow. By default, the full scoring tests are withheld. HumanEval retains its problem docstring; MBPP retains one example assertion from the supplied runner.

`--expose-public-tests` is an explicit diagnostic option for code tasks. It reveals the same tests used for scoring, prints a warning, and marks the JSON report `test-visible-diagnostic`. Such a run must not be reported as held-out benchmark accuracy. The supplied archive previously coupled this behavior to `--maas`; the public release separates the two.

Code tasks and Python repair nodes execute generated code in subprocesses with timeouts, not a security sandbox. Use a disposable environment without unrelated credentials. Do not run unreviewed repair programs on a workstation with valuable data.

## Scope and differences

- **Executor:** nodes run in list order; graph edges are editable metadata, not a parallel DAG scheduler.
- **Clustering:** semantic embeddings plus GMM/BIC when configured; rule-based failure buckets for small batches or embedding errors. Local hashing embeddings are useful for debugging, not equivalent to the paper's semantic embedding setup.
- **Math:** four MATH subjects, balanced loading, normalized string matching. The grader does not establish symbolic equivalence; inspect normalization when comparing protocols.
- **QA:** `gaia` dispatches to selected BBH tasks. It does not implement GAIA's tool-using/file-based evaluation.
- **Backbone:** configurable public API models; the default is `openai/gpt-4o-mini`. Provider versions and generation randomness affect results.
- **Baseline provenance:** the supplied archive attributes its MaAS-style operator reconstruction to [bingreeky/MaAS](https://github.com/bingreeky/MaAS). Keep that attribution when extending these operators.
- **Optional extensions:** regression/guard checks and code synthesis are available through `refine_workflow`; the standard CLI does not enable all of them.

Reports include the actual split IDs, model, embedding backend, workflow, and per-iteration edit records. They intentionally contain no API key. Record provider/model revision, all seeds, token budget and test-visibility mode for any numerical comparison. Report all planned seeds; choosing the best seed on the evaluation set changes the evaluation protocol.

## Public-release maintenance

- Corrected the incompatible `datasets` / `huggingface-hub` dependency constraints.
- Replaced the legacy private-gateway default with the configurable public API client while keeping the `AIDPClient` import alias.
- Removed local evaluation application headers and unverified hard-coded benchmark target scores.
- Separated scoring-test exposure from the MaAS architecture switch.
- Added checks for invalid or undersized data slices and saved full workflow/refinement metadata in reports.
- Added offline checks, the original-paper overview and publication metadata.

The original release's optional format-normalization helpers remain in place. No paid API benchmark suite was rerun as part of preparing the repository.
