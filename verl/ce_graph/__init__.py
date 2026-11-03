"""CE-Graph: Counterexample-Guided Workflow Refinement.

This subpackage implements a practical version of the paper's core loop:

    W_{t+1} = Repair(W_t, b_t, O)

where b_t is a clustered failure mode derived from a counterexample pool.

Design goals:
  - minimal dependencies (numpy + sklearn)
  - runnable on a local machine
  - easy to extend with your own task runners and operators
"""

from .counterexamples import CounterexamplePool, ExecutionTrace
from .executor import Executor
from .failures import FailureClustering, FailureMode, FailureSignature, EmbeddingGMMClustering, make_openai_embed_fn
from .llm import AIDPClient, LLMResponse
from .operators import Operator, OperatorLibrary
from .proposer import Edit, LLMProposer, apply_edit
from .refine import RefinementReport, refine_workflow
from .workflow import Node, Workflow

__all__ = [
    "AIDPClient",
    "CounterexamplePool",
    "Edit",
    "Executor",
    "ExecutionTrace",
    "FailureClustering",
    "FailureMode",
    "FailureSignature",
    "EmbeddingGMMClustering",
    "make_openai_embed_fn",
    "LLMProposer",
    "LLMResponse",
    "Node",
    "Operator",
    "OperatorLibrary",
    "RefinementReport",
    "Workflow",
    "apply_edit",
    "refine_workflow",
]
