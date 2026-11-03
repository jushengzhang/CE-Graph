"""Tiny code-execution sandbox for HumanEval / MBPP grading.

Runs candidate Python code in a subprocess with a wall-clock timeout. Returns
(success, error_message). Stdout/stderr are captured but only error is exposed.
Designed for trusted benchmark grading — not adversarial security isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    ok: bool
    error: str = ""
    stdout: str = ""


def run_python(code: str, timeout_s: float = 8.0) -> SandboxResult:
    """Execute `code` as a standalone Python script in a subprocess."""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode == 0:
            return SandboxResult(ok=True, stdout=proc.stdout)
        # condense stderr
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return SandboxResult(ok=False, error=" | ".join(err[-3:])[:300])
    except subprocess.TimeoutExpired:
        return SandboxResult(ok=False, error="timeout")
    except Exception as e:
        return SandboxResult(ok=False, error=f"{type(e).__name__}: {e!s}"[:300])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def extract_python_block(text: str) -> str:
    """Pull the largest python code block out of `text`.

    Supports ```python / ``` / ``` py fenced blocks; falls back to the whole
    string if no fence is found.

    Trailing whitespace is stripped, but leading-line indentation is preserved
    so the caller can decide how to splice the block onto a host program. A
    bare `.strip()` would erase the first line's indent while leaving later
    lines indented, yielding inconsistent indentation that fails to parse.
    """
    if not text:
        return ""
    import re

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).rstrip()
    return text.strip()
