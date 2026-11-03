from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

# NOTE: We intentionally avoid heavy clustering backends by default.
# Some environments can hang due to OpenMP/threading interactions in certain
# sklearn clusterers. CE-Graph does not require a specific clustering algorithm;
# it only requires *consistent* bucketing into recurring failure modes.

import re

from .counterexamples import ExecutionTrace


_EXCEPTION_RE = re.compile(
    r"\b("
    r"IndentationError|SyntaxError|NameError|TypeError|ValueError|KeyError|AttributeError|"
    r"IndexError|ZeroDivisionError|RecursionError|AssertionError|ImportError|ModuleNotFoundError|"
    r"FileNotFoundError|StopIteration|MemoryError|OverflowError|RuntimeError|"
    r"UnboundLocalError|NotImplementedError|TimeoutError"
    r")\b"
)


def _extract_error_signature(error_msg: str) -> str:
    """Coarse but useful failure signature.

    For sandbox failures the literal first line varies per example (it contains
    file paths, line numbers, source snippets). Bucketing on the literal text
    produces singleton clusters. Bucketing on the *exception class* (or a
    well-known semantic tag like `parse_failure:no_boxed`) produces dense
    clusters that the Proposer can actually attack.
    """
    if not error_msg:
        return "no_error"
    msg = error_msg.lower()
    # Semantic tags we emit ourselves (math/code parsers, format checks)
    for tag in (
        "parse_failure:no_boxed",
        "parse_failure:no_number",
        "parse_failure:no_choice",
        "invalid_json",
        "wrong_answer",
        "verifier:missing_integer",
        "verifier:invalid_json",
        "sandbox:empty_completion",
        "sandbox:timeout",
        "aidp",
    ):
        if tag in msg:
            return tag
    # Python exception class in a sandbox traceback
    m = _EXCEPTION_RE.search(error_msg)
    if m:
        return f"py_exception:{m.group(1).lower()}"
    # Fallback: first 40 chars of first line
    return error_msg.split("\n", 1)[0].strip().lower()[:40]


@dataclass
class FailureSignature:
    r"""Failure signature \phi(\tau).

    Keep it simple and auditable: a short text string + structured flags.
    """

    text: str
    flags: Dict[str, Any]


@dataclass
class FailureMode:
    """A clustered failure mode b in the paper."""

    mode_id: int
    size: int
    exemplar_indices: List[int]
    keywords: List[str]


class FailureClustering:
    """Cluster execution failures into recurring modes.

    Default implementation groups recurring semantic tags and exception classes.
    This is intentionally lightweight; you can swap it with embedding-based
    clustering if desired.
    """

    def __init__(
        self,
        max_modes: int = 12,
        min_cluster_size: int = 10,
        random_state: int = 0,
    ):
        self.max_modes = int(max_modes)
        self.min_cluster_size = int(min_cluster_size)
        self.random_state = int(random_state)

    def signature(self, trace: ExecutionTrace) -> FailureSignature:
        # Convert trace into a stable, low-variance textual representation.
        # You should customize this to your environments / tools.
        pieces: List[str] = []
        flags: Dict[str, Any] = {}

        err = (trace.error or "").strip()
        if err:
            pieces.append(f"error:{err[:200]}")
            flags["has_error"] = True
        else:
            flags["has_error"] = False

        # Common structure: tool failures / invalid JSON / verifier failed
        tool_errors = []
        invalid_json = 0
        for step in trace.node_outputs:
            if step.get("tool_error"):
                tool_errors.append(str(step.get("tool_error"))[:120])
            if step.get("invalid_json"):
                invalid_json += 1

        if tool_errors:
            pieces.append("tool:" + " | ".join(tool_errors[:3]))
            flags["tool_error"] = True
        else:
            flags["tool_error"] = False

        if invalid_json:
            pieces.append(f"invalid_json:{invalid_json}")
            flags["invalid_json"] = True
        else:
            flags["invalid_json"] = False

        # Add a compact view of scalar metrics
        if trace.metrics:
            for k in sorted(trace.metrics.keys()):
                v = trace.metrics[k]
                if isinstance(v, (int, float)):
                    pieces.append(f"m:{k}={float(v):.3f}")

        # Fallback: include last output snippet for discriminating modes
        if trace.final_output is not None:
            s = str(trace.final_output)
            pieces.append("out:" + s[:200])

        text = " ; ".join(pieces) if pieces else "unknown_failure"
        return FailureSignature(text=text, flags=flags)

    def cluster(self, traces: Sequence[ExecutionTrace]) -> Tuple[List[FailureMode], np.ndarray]:
        """Return (modes, assignments).

        assignments[i] = mode_id for traces[i].
        """
        if len(traces) == 0:
            return [], np.array([], dtype=np.int64)

        sigs = [self.signature(t) for t in traces]
        texts = [s.text for s in sigs]

        # Rule-based bucketing on a *coarse* signature (exception class / semantic
        # tag) so code-benchmark sandbox failures don't all land in singleton
        # clusters because their first lines carry distinct file/line numbers.
        buckets: Dict[Tuple[str, bool, bool], List[int]] = {}
        for i, t in enumerate(traces):
            err_sig = _extract_error_signature(t.error or "")
            s = sigs[i]
            key = (err_sig, bool(s.flags.get("tool_error")), bool(s.flags.get("invalid_json")))
            buckets.setdefault(key, []).append(i)

        # Sort buckets by size and cap to max_modes.
        bucket_items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        bucket_items = bucket_items[: self.max_modes]

        labels = np.full(len(traces), -1, dtype=np.int64)
        modes: List[FailureMode] = []
        for mode_id, (key, idx) in enumerate(bucket_items):
            for i in idx:
                labels[i] = mode_id

            # Keywords: simple token frequency within the bucket.
            kws = self._keywords_from_texts([texts[i] for i in idx], topn=8)
            modes.append(
                FailureMode(
                    mode_id=mode_id,
                    size=len(idx),
                    exemplar_indices=idx[: min(5, len(idx))],
                    keywords=kws,
                )
            )

        if not modes:
            return [self._singleton_mode(texts, 0, list(range(len(texts))))], np.zeros(len(texts), dtype=np.int64)

        return modes, labels

    def _singleton_mode(self, texts: List[str], mode_id: int, indices: List[int]) -> FailureMode:
        keywords = []
        if texts:
            # crude keywords: most common tokens
            toks = " ".join(texts).split()
            freq: Dict[str, int] = {}
            for t in toks:
                t = t.strip().lower()
                if len(t) < 3:
                    continue
                freq[t] = freq.get(t, 0) + 1
            keywords = [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:8]]
        return FailureMode(mode_id=mode_id, size=len(indices), exemplar_indices=indices[: min(5, len(indices))], keywords=keywords)

    def _keywords_from_texts(self, texts: List[str], topn: int = 8) -> List[str]:
        toks = " ".join(texts).replace(";", " ").replace("|", " ").split()
        freq: Dict[str, int] = {}
        for t in toks:
            t = t.strip().lower()
            if len(t) < 4:
                continue
            if t.startswith("m:") or t.startswith("out:"):
                continue
            freq[t] = freq.get(t, 0) + 1
        return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:topn]]

    # Keeping TF-IDF helpers for optional future use.


class EmbeddingGMMClustering(FailureClustering):
    """Paper-aligned clustering: semantic embedding + GMM with BIC model selection.

    Matches CE-Graph (arXiv 2510.10035) §3.2:
      - ψ_sem maps error message → R^d (we use OpenAI text-embedding-3-small, d=1536)
      - Fit GaussianMixture, select n_components by BIC
      - Dominant mode = max mixing coefficient π_k

    Falls back to the rule-based parent class when < min_for_gmm traces are present.
    """

    def __init__(self, *, embed_fn, max_modes: int = 6, min_cluster_size: int = 2,
                 min_for_gmm: int = 5, random_state: int = 0,
                 reg_covar: float = 1e-3):
        super().__init__(max_modes=max_modes, min_cluster_size=min_cluster_size, random_state=random_state)
        self.embed_fn = embed_fn  # callable: list[str] -> np.ndarray (n, d)
        self.min_for_gmm = int(min_for_gmm)
        self.reg_covar = float(reg_covar)

    def cluster(self, traces):
        if len(traces) < self.min_for_gmm:
            return super().cluster(traces)

        sigs = [self.signature(t) for t in traces]
        texts = [s.text for s in sigs]

        try:
            embeddings = self.embed_fn(texts)
            embeddings = np.asarray(embeddings, dtype=np.float64)
            assert embeddings.ndim == 2 and embeddings.shape[0] == len(texts)
        except Exception:
            return super().cluster(traces)

        from sklearn.mixture import GaussianMixture
        n_samples = len(traces)
        ks = list(range(1, min(self.max_modes, n_samples) + 1))
        best_bic = float("inf"); best_gmm = None
        for k in ks:
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type="diag",
                    random_state=self.random_state, max_iter=80, n_init=1,
                    reg_covar=self.reg_covar,
                )
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic; best_gmm = gmm
            except Exception:
                continue
        if best_gmm is None:
            return super().cluster(traces)

        labels = best_gmm.predict(embeddings)
        weights = best_gmm.weights_

        order = np.argsort(-weights)  # densest first
        new_labels = np.full(n_samples, -1, dtype=np.int64)
        modes: List[FailureMode] = []
        for new_id, orig_id in enumerate(order):
            idx = [i for i in range(n_samples) if labels[i] == orig_id]
            if not idx:
                continue
            for i in idx:
                new_labels[i] = new_id
            kws = self._keywords_from_texts([texts[i] for i in idx], topn=8)
            modes.append(FailureMode(
                mode_id=new_id, size=len(idx),
                exemplar_indices=idx[: min(5, len(idx))],
                keywords=kws,
            ))
        if not modes:
            return super().cluster(traces)
        return modes, new_labels


def make_openai_embed_fn(api_key=None, model="openai/text-embedding-3-small",
                          base_url="https://openrouter.ai/api/v1/embeddings", batch=64):
    """Return a callable: list[str] -> np.ndarray (n, d) via OpenRouter."""
    import json as _json, urllib.request, urllib.error, os as _os, time as _time
    key = api_key or _os.environ.get("OPENROUTER_API_KEY", "")

    def _embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            body = _json.dumps({"model": model, "input": chunk}).encode()
            req = urllib.request.Request(
                base_url, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        resp = _json.loads(r.read())
                    out.extend([d["embedding"] for d in resp["data"]])
                    break
                except Exception:
                    if attempt == 2: raise
                    _time.sleep(2 * (attempt + 1))
        return np.array(out, dtype=np.float64)
    return _embed
