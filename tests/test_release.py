from __future__ import annotations
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from verl.ce_graph import CounterexamplePool, ExecutionTrace, FailureClustering, Node, Workflow, refine_workflow
from verl.ce_graph.failures import EmbeddingGMMClustering
from verl.ce_graph.llm import AIDPClient, LLMResponse
from verl.ce_graph.openrouter_llm import OpenRouterClient
from verl.ce_graph.proposer import Edit, apply_edit
from examples.ce_graph_refine.offline_demo import add_verifier, toy_runner
from verl.ce_graph.operators import Operator, OperatorLibrary


class ReleaseTests(unittest.TestCase):
    def test_pool_retains_only_failures_and_respects_capacity(self):
        pool = CounterexamplePool(capacity=2)
        pool.extend([ExecutionTrace(str(i), 'W0', is_success=(i == 0)) for i in range(4)])
        self.assertEqual([t.instance_id for t in pool.all()], ['2', '3'])

    def test_prompt_edits_do_not_mutate_the_original(self):
        original = Workflow(nodes=[Node('answer', 'llm', {'nested': {'a': 1}})])
        edited = apply_edit(original, Edit('RevisePrompt', {'node_id': 'answer', 'prompt_prefix': 'Check carefully'}))
        self.assertNotIn('prompt_prefix', original.nodes[0].config)
        edited.nodes[0].config['nested']['a'] = 2
        self.assertEqual(original.nodes[0].config['nested']['a'], 1)
        with self.assertRaises(ValueError):
            apply_edit(original, Edit('RevisePrompt', {'node_id': 'missing', 'system': 'x'}))

    def test_refinement_verifies_and_accepts_a_targeted_repair(self):
        initial = Workflow(nodes=[Node('answer', 'llm')])
        operators = OperatorLibrary([Operator('repair', lambda w, m: True, add_verifier)])
        final, reports = refine_workflow(initial, toy_runner, operator_lib=operators, max_iters=2, verbose=False)
        self.assertEqual(toy_runner(final)[0], 1.0)
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].accepted)
        self.assertEqual(len(initial.nodes), 1)

    def test_refinement_rejects_non_improving_edits(self):
        initial = Workflow(nodes=[Node('answer', 'llm')])
        operators = OperatorLibrary([Operator('noop', lambda w, m: True, lambda w, m: w.copy())])
        final, reports = refine_workflow(initial, toy_runner, operator_lib=operators, patience=1, max_iters=3, verbose=False)
        self.assertEqual(len(reports), 1)
        self.assertFalse(reports[0].accepted)
        self.assertEqual(final.to_dict(), initial.to_dict())

    def test_gmm_and_embedding_failure_fallback(self):
        import numpy as np
        traces = [ExecutionTrace(str(i), 'W0', error='invalid_json' if i < 4 else 'wrong_answer') for i in range(8)]
        clustering = EmbeddingGMMClustering(embed_fn=lambda texts: np.array([[0, i / 100] if i < 4 else [10, i / 100] for i in range(8)]), max_modes=2)
        modes, labels = clustering.cluster(traces)
        self.assertEqual(sum(m.size for m in modes), 8)
        self.assertEqual(len(labels), 8)
        fallback = EmbeddingGMMClustering(embed_fn=lambda texts: (_ for _ in ()).throw(RuntimeError('offline')))
        modes, _ = fallback.cluster(traces)
        self.assertEqual(sorted(m.size for m in modes), [4, 4])

    def test_maas_does_not_reveal_scoring_tests_by_default(self):
        from examples.ce_graph_refine.humaneval_runner import HumanEvalExample, HumanEvalTaskRunner
        from examples.ce_graph_refine.mbpp_runner import MBPPExample, MBPPTaskRunner
        fake_llm = object()
        he = HumanEvalExample('he0', 'def answer():\n    """Return a value."""\n', 'def check(candidate):\n    assert candidate() == 42', 'answer')
        self.assertNotIn('def check', HumanEvalTaskRunner([he], llm=fake_llm)._question_for(he))
        self.assertIn('def check', HumanEvalTaskRunner([he], llm=fake_llm, expose_public_tests=True)._question_for(he))
        mb = MBPPExample('mb0', 'Return 42', ['assert answer() == 42', 'assert answer() > 40'], [], 'assert answer() == 42')
        prompt = MBPPTaskRunner([mb], llm=fake_llm)._question_for(mb)
        self.assertNotIn('assert answer() > 40', prompt)
        self.assertIn('assert answer() > 40', MBPPTaskRunner([mb], llm=fake_llm, expose_public_tests=True)._question_for(mb))

    def test_public_client_compatibility_and_request_contract(self):
        self.assertIs(AIDPClient, OpenRouterClient)
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        def respond(request, timeout):
            self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/chat/completions')
            body = json.loads(request.data)
            self.assertEqual(body['max_tokens'], 2048)
            self.assertEqual(request.headers['Authorization'], 'Bearer test-not-a-real-key')
            return Response(json.dumps({'choices': [{'message': {'content': 'ok'}}], 'usage': {'prompt_tokens': 3}}).encode())
        with patch('urllib.request.urlopen', side_effect=respond):
            result = OpenRouterClient(api_key='test-not-a-real-key').chat([{'role':'user','content':'hello'}], max_tokens=2048)
        self.assertEqual(result.text, 'ok')
        self.assertEqual(result.prompt_tokens, 3)

    def test_cli_help_and_missing_key_do_not_use_network(self):
        command = [sys.executable, '-m', 'examples.ce_graph_refine.run_all_paper']
        result = subprocess.run(command + ['--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--expose-public-tests', result.stdout)
        clean_env = {key: value for key, value in os.environ.items() if key != 'OPENROUTER_API_KEY'}
        result = subprocess.run(command + ['--benchmark', 'gsm8k'], env=clean_env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Set OPENROUTER_API_KEY', result.stderr)


if __name__ == '__main__':
    unittest.main()
