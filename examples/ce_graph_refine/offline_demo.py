"""Deterministic toy refinement demo; no API key, datasets, or network calls.

Run from the repository root with:
    python -m examples.ce_graph_refine.offline_demo
This demonstrates the control flow, not a measured research benchmark result.
"""
from __future__ import annotations
import json
from verl.ce_graph import ExecutionTrace, FailureClustering, Node, Operator, OperatorLibrary, Workflow, refine_workflow


def toy_runner(workflow: Workflow):
    repaired = any(node.node_type == "verifier" for node in workflow.nodes)
    traces = [ExecutionTrace(
        instance_id=f"toy-{i}", workflow_version=workflow.metadata.get("version", "W0"),
        final_output='{"answer": 42}' if repaired else "answer: 42",
        node_outputs=[{"invalid_json": not repaired}],
        is_success=repaired, error=None if repaired else "invalid_json",
    ) for i in range(4)]
    return sum(trace.is_success for trace in traces) / len(traces), traces


def add_verifier(workflow, mode):
    updated = workflow.copy()
    updated.add_node(Node("format_check", "verifier", {"strict": True}))
    updated.add_edge("answer", "format_check")
    updated.metadata["version"] = "W1"
    return updated


def main():
    initial = Workflow(nodes=[Node("answer", "llm")], metadata={"name": "offline-toy", "version": "W0"})
    operators = OperatorLibrary([Operator("add_format_verifier", lambda w, m: not any(n.node_type == "verifier" for n in w.nodes), add_verifier)])
    final, reports = refine_workflow(initial, toy_runner, operator_lib=operators,
        clustering=FailureClustering(min_cluster_size=1), max_iters=2, verify_k=4, verbose=False)
    result = {"demo": "synthetic control-flow example, not benchmark accuracy", "baseline_score": toy_runner(initial)[0],
              "refined_score": toy_runner(final)[0], "accepted_edits": sum(r.accepted for r in reports), "workflow": final.to_dict()}
    assert result["accepted_edits"] == 1 and result["refined_score"] == 1.0
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
