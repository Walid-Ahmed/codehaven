"""The evidence kept for a final answer, and the checks an answer's citations must pass."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import run
from offline_agent.evidence import classify, retain, check_answer, discovery_task
from offline_agent.model import response_metrics
from offline_agent.workspace import Workspace


class EvidenceTests(unittest.TestCase):
    def test_classification_distinguishes_services_tests_stubs_and_docs(self):
        for path, expected in (("svc/app/main.py", "service"), ("tests/test_api.py", "test"),
                               ("scripts/inventory_stub.py", "stub"), ("README.md", "reference")):
            self.assertEqual(classify(path, "app = FastAPI()"), expected)

    def test_bounded_memory_keeps_snippets_and_prioritizes_citations(self):
        matches = [dict(path=f"services/service_{i}/app/main.py", line=10, text="app = FastAPI() " * 30) for i in range(80)]
        evidence, limited = retain([], matches)
        self.assertTrue(limited)
        self.assertLessEqual(len(evidence), 50)
        self.assertLessEqual(len(json.dumps(evidence, ensure_ascii=False).encode()), 4000)
        self.assertTrue(all(row['kind'] == 'service' for row in evidence))
        self.assertTrue(all('snippet' in row for row in evidence))

    def test_supported_citations_missing_coverage_and_single_app_contradiction(self):
        evidence, _ = retain([], [dict(path='a/app/main.py', line=10, text='FastAPI('),
                                   dict(path='b/app/main.py', line=20, text='FastAPI(')])
        good = 'Apps: [a/app/main.py:10] and [repo/b/app/main.py:20].'
        self.assertEqual(check_answer(good, evidence, discovery=True), [])
        issues = check_answer('The FastAPI entry point is a/app/main.py [a/app/main.py:10].', evidence, discovery=True)
        self.assertTrue(any('single' in issue for issue in issues))
        self.assertTrue(any('Missing' in issue for issue in issues))
        self.assertTrue(any('not supported' in issue for issue in check_answer('[a/app/main.py:999]', evidence)))
        self.assertTrue(any('require' in issue for issue in check_answer('It has apps.', evidence)))
        duplicate_labels = [dict(path='repo/a/app/main.py', line=10, kind='service'),
                            dict(path='a/app/main.py', line=10, kind='service')]
        self.assertEqual(check_answer('The FastAPI entry point is [a/app/main.py:10].', duplicate_labels), [])

    def test_urls_and_host_ports_are_not_citations(self):
        evidence = [dict(path='app/main.py', line=3, kind='service', snippet='')]
        answer = 'Service at app/main.py:3; Ollama listens on http://127.0.0.1:11434 or localhost:8000/docs.'
        self.assertEqual(check_answer(answer, evidence), [])

    def test_classification_matches_whole_paths_not_substrings(self):
        evidence = [dict(path='app/main.py', line=3, kind='service', snippet=''),
                    dict(path='tests/app/main.py', line=2, kind='test', snippet='')]
        answer = 'Services:\n- app/main.py:3\nTests:\n- tests/app/main.py:2'
        self.assertEqual(check_answer(answer, evidence, discovery=True), [])
        swapped = 'Services:\n- tests/app/main.py:2\nTests:\n- app/main.py:3'
        self.assertEqual(sum('Classification mismatch' in issue for issue in check_answer(swapped, evidence, discovery=True)), 2)

    def test_discovery_does_not_intercept_edits_or_unrelated_tasks(self):
        self.assertTrue(discovery_task('Find the FastAPI entry points. Do not modify files.'))
        self.assertTrue(discovery_task("Find FastAPI entry points; don't modify anything"))
        self.assertTrue(discovery_task('List FastAPI entry points without changing or deleting files'))
        self.assertFalse(discovery_task("Find FastAPI entry points, don't delete them, but add logging"))
        self.assertFalse(discovery_task('Find and fix FastAPI entry points'))
        self.assertFalse(discovery_task('Explain how FastAPI works'))

    def test_real_model_failure_pattern_is_rejected_and_inventory_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('catalog/app/main.py', 'orders/app/main.py', 'tests/test_api.py'):
                path = Path(root, name); path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('from fastapi import FastAPI\napp = FastAPI()\n')
            bad_answer = ('Service candidates:\n- catalog/app/main.py:2\n- orders/app/main.py:2\n'
                          '- tests/test_api.py:2\n\n' +
                          'Limited coverage; evidence does not include all potential entry points.\n' * 3)
            class Model:
                def chat(self, messages, schema):
                    return dict(action='finish', path='', message='Discovery summary', content=bad_answer)
            output = []
            result = run(Model(), Workspace(root), 'Find all FastAPI entry points', output=output.append)
            self.assertEqual(result, 2)
            self.assertNotIn(bad_answer, output)
            self.assertTrue(any('Classification mismatch' in line for line in output))
            self.assertTrue(any('repeats' in line for line in output))
            self.assertTrue(any('Test candidates:\n- [tests/test_api.py:2]' in line for line in output))

    def test_discovery_searches_once_then_summarizes_without_per_file_reads(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ('one/app/main.py', 'two/app/main.py', 'tests/test_api.py', 'scripts/stub.py'):
                file = Path(root, path); file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('app = FastAPI()\n')
            workspace = Workspace(root)
            calls = []
            class Model:
                def chat(self, messages, schema):
                    calls.append((messages, schema))
                    return dict(action='finish', path='', message='Done',
                                content='Apps: [one/app/main.py:1] [two/app/main.py:1]. Tests and stubs are separate candidates.')
            with patch.object(workspace, 'search', wraps=workspace.search) as search:
                result = run(Model(), workspace, 'Find all FastAPI entry points', max_steps=4, output=lambda _: None)
            self.assertEqual(result, 0)
            search.assert_called_once_with('FastAPI(')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]['properties']['action']['enum'], ['finish'])
            self.assertIn('"kind": "test"', calls[0][0][-1]['content'])
            self.assertIn('"kind": "stub"', calls[0][0][-1]['content'])

    def test_metrics_rates_cache_unavailable_and_zero_duration(self):
        result = response_metrics(dict(load_duration=1_000_000_000, prompt_eval_duration=2_000_000_000,
                                       eval_duration=4_000_000_000, eval_count=20, prompt_eval_count=100,
                                       prompt_eval_cached_count=80))
        self.assertEqual(result['load_seconds'], 1)
        self.assertEqual(result['prompt_seconds'], 2)
        self.assertEqual(result['tokens_per_second'], 5)
        self.assertEqual(result['cached_tokens'], 80)
        self.assertIsNone(response_metrics(dict(eval_count=0, eval_duration=0))['cached_tokens'])
        self.assertNotIn('tokens_per_second', response_metrics(dict(eval_count=0, eval_duration=0)))
        self.assertEqual(response_metrics({}), {})
        self.assertEqual(response_metrics({'eval_duration': 'unknown', 'eval_count': True}), {})

    def test_bad_discovery_answer_is_flagged_and_exits_unverified(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ('a/app/main.py', 'b/app/main.py'):
                file = Path(root, path); file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('app = FastAPI()\n')
            class Model:
                last_metrics = {'load_seconds': 0.1, 'cached_tokens': None}
                def chat(self, messages, schema):
                    return dict(action='finish', path='', message='Done',
                                content='The FastAPI entry point is a/app/main.py [a/app/main.py:999].')
            output = []
            status = run(Model(), Workspace(root), 'Find all FastAPI entry points', output=output.append)
            self.assertEqual(status, 2)
            self.assertTrue(any('unverified' in line for line in output))
            self.assertTrue(any('multiple service' in line for line in output))
            self.assertTrue(any('not supported' in line for line in output))
            self.assertTrue(any('Ollama metrics:' in line for line in output))

    def test_reported_metrics_are_visible_when_response_hits_token_limit(self):
        class Model:
            last_metrics = {'generated_tokens': 1024, 'generation_seconds': 90}
            def chat(self, messages, schema):
                raise ValueError('Model reached the response token limit')
        with tempfile.TemporaryDirectory() as root:
            output = []
            with self.assertRaisesRegex(ValueError, 'token limit'):
                run(Model(), Workspace(root), 'Explain', output=output.append)
            self.assertIn('Ollama metrics: generation 90.0s; 1,024 generated', output)


class EvidenceCoverageTests(unittest.TestCase):
    """Citations are checked against every line observed, and evidence keeps its coverage when it is trimmed."""

    def test_a_citation_may_span_adjacent_passages_but_not_a_gap(self):
        from offline_agent.evidence import covered
        rows = [{"path": "a.py", "line": 1, "end_line": 10}, {"path": "a.py", "line": 11, "end_line": 20},
                {"path": "a.py", "line": 30}, {"path": "repo/b.py", "line": 5, "end_line": 8}]
        for path, start, end, expected in (("a.py", 5, 15, True), ("a.py", 19, 21, False), ("a.py", 30, 30, True),
                                           ("a.py", 25, 30, False), ("b.py", 6, 8, True), ("c.py", 1, 1, False)):
            with self.subTest(path=path, start=start, end=end):
                self.assertEqual(covered(rows, path, start, end), expected)
        self.assertEqual(check_answer("It spans both. [a.py:5-15]", rows), [])

    def test_many_rows_are_merged_into_ranges_instead_of_being_lost(self):
        rows = [{"path": "big.py", "line": n, "end_line": n, "text": "x" * 60} for n in range(1, 200)]
        kept, limited = retain([], rows)
        self.assertEqual(kept, [{"path": "big.py", "line": 1, "end_line": 199, "kind": "source", "snippet": ""}])
        self.assertFalse(limited)  # Every observed line is still covered.

    def test_when_something_must_go_the_oldest_rows_go_first(self):
        rows = [{"path": f"f{n}.py", "line": 1, "text": ""} for n in range(80)]
        kept, limited = retain([], rows)
        self.assertTrue(limited)
        self.assertEqual(kept[-1]["path"], "f79.py")
        self.assertNotIn("f0.py", [row["path"] for row in kept])

