from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionTrace:
    """One execution trace for a single input instance.

    This mirrors the paper's execution trace: it stores node activations, intermediate
    outputs, tool errors, and the final verdict.
    """

    instance_id: str
    workflow_version: str
    node_outputs: List[Dict[str, Any]] = field(default_factory=list)
    final_output: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_success: bool = False
    error: Optional[str] = None


class CounterexamplePool:
    """A bounded memory of failed executions.

    The pool is append-only; sampling is stratified by failure mode during
    refinement.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = int(capacity)
        self._traces: List[ExecutionTrace] = []

    def add(self, trace: ExecutionTrace) -> None:
        if trace.is_success:
            return
        self._traces.append(trace)
        if len(self._traces) > self.capacity:
            # FIFO drop
            self._traces = self._traces[-self.capacity :]

    def extend(self, traces: List[ExecutionTrace]) -> None:
        for t in traces:
            self.add(t)

    def all(self) -> List[ExecutionTrace]:
        return list(self._traces)

    def __len__(self) -> int:
        return len(self._traces)
