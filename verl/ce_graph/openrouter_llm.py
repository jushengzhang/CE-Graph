"""Minimal OpenRouter client matching AIDPClient.chat() interface.

Usage:
    llm = OpenRouterClient(api_key=..., model="openai/gpt-4o-mini")
    r = llm.chat([{"role": "user", "content": "..."}], max_tokens=512, temperature=0)
    print(r.text)
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_DEFAULT_BASE = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class _Resp:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int = 1024,
             temperature: float = 0.0, **kwargs) -> _Resp:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jushengzhang/CE-Graph",
            "X-Title": "CE-Graph",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(self.base_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw_body = resp.read().decode("utf-8")
                obj = json.loads(raw_body)
                text = obj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                usage = obj.get("usage") or {}
                return _Resp(
                    text=text,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    raw=obj,
                )
            except urllib.error.HTTPError as e:
                last_err = e
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    err_body = ""
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(self.retry_backoff * attempt)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(self.retry_backoff * attempt)
                continue
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries: {last_err}")
