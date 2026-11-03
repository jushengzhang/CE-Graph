"""Public OpenAI-compatible client interface.

The legacy AIDPClient import is retained for compatibility with the supplied
runners. It now uses the public OpenRouter client; no private gateway is needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from .openrouter_llm import OpenRouterClient

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

# Backwards-compatible name used by the original release's runner annotations.
AIDPClient = OpenRouterClient

def default_client() -> OpenRouterClient:
    return OpenRouterClient()
