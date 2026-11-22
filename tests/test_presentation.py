"""The short summaries of search and retrieval results shown in the terminal."""
import json
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import format_metrics, format_retrieval, format_search, run
from offline_agent.workspace import Workspace


class PresentationTests(unittest.TestCase):
    def test_search_summary_lists_aligned_matches_and_limits(self):
        data = dict(query="FastAPI(", scope=".", files_searched=40, files_skipped=2, limit_reached=True,
                    candidate_limit_reached=False,
                    matches=[dict(path="a/main.py", line=2, text="app = FastAPI()"),
                             dict(path="services/b/main.py", line=12, text="api   =   FastAPI(title='b')")])
        lines = format_search(data).splitlines()
        self.assertEqual(lines[0], 'Found 2 matches for "FastAPI(" in . (40 files searched, 2 skipped); results limited')
        self.assertEqual(lines[1], "  a/main.py:2            app = FastAPI()")
        self.assertEqual(lines[1].index("app ="), lines[2].index("api ="))
        self.assertEqual(lines[2], "  services/b/main.py:12  api = FastAPI(title='b')")

    def test_search_summary_caps_rows_and_handles_no_matches(self):
        many = dict(query="x", scope="src", files_searched=1, matches=[
            dict(path="a.py", line=n, text="x" * 300) for n in range(1, 31)])
        text = format_search(many)
        self.assertIn("… 10 more matches passed to the model", text)
        self.assertTrue(all(len(line) < 130 for line in text.splitlines()))
        self.assertEqual(format_search(dict(query="x", scope=".", files_searched=3, matches=[])),
                         'Found 0 matches for "x" in . (3 files searched)')

    def test_retrieval_summary_shows_sources_scores_and_dependencies(self):
        data = dict(context_bytes=3812, context_budget_bytes=4000, budget_limited=True, results=[
            dict(source="repo/app.py", start_line=1, end_line=30, kind="match", score=4.2137,
                 symbol="create_app", symbol_complete=True),
            dict(source="repo/db.py", start_line=5, end_line=9, kind="dependency", via="repo/app.py",
                 symbol="connect", symbol_complete=False)])
        self.assertEqual(format_retrieval(data).splitlines(), [
            "Retrieved 2 passages (3,812 of 4,000 bytes); results limited",
            "  repo/app.py:1-30  score 4.21, create_app",
            "  repo/db.py:5-9    dependency via repo/app.py, connect (partial)"])

    def test_metrics_summary_is_one_readable_line(self):
        metrics = dict(load_seconds=0.25, prompt_seconds=3.1, generation_seconds=9.04, total_seconds=12.4,
                       prompt_tokens=2310, cached_tokens=1900, generated_tokens=180, tokens_per_second=19.91)
        self.assertEqual(format_metrics(metrics),
                         "Ollama metrics: 12.4s total (load 0.2s, prompt 3.1s, generation 9.0s); "
                         "2,310 prompt tokens (1,900 cached); 180 generated at 19.9 tok/s")
        self.assertEqual(format_metrics(dict(prompt_tokens=5, cached_tokens=None)), "Ollama metrics: 5 prompt tokens")
        self.assertEqual(format_metrics(dict(cached_tokens=None)), "Ollama metrics: no timings or token counts reported")

    def test_terminal_gets_summary_while_model_gets_full_json(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "code.py").write_text("value = 42\n")
            calls = []
            class Model:
                def chat(self, messages, schema):
                    calls.append(messages)
                    if len(calls) == 1:
                        return dict(action="search", path="", content="value", message="Find value")
                    return dict(action="finish", path="", message="Done", content="It is 42. [code.py:1]")
            output = []
            self.assertEqual(run(Model(), Workspace(root), "What is value?", output=output.append), 0)
            self.assertIn('Found 1 match for "value" in . (1 files searched)\n  code.py:1  value = 42', output)
            self.assertFalse(any('"matches"' in line for line in output))
            tool_message = calls[1][-1]["content"]
            self.assertEqual(json.loads(tool_message.split("\n", 1)[1])["matches"][0]["line"], 1)

    def test_read_errors_are_shown_to_the_user(self):
        with tempfile.TemporaryDirectory() as root:
            replies = iter([dict(action="read", path="missing.py", content="", message="Read it"),
                            dict(action="finish", path="", message="Done", content="The file is missing.")])
            class Model:
                def chat(self, messages, schema):
                    return next(replies)
            output = []
            run(Model(), Workspace(root), "Explain missing.py", output=output.append)
            self.assertTrue(any(line.startswith("Tool error:") for line in output))
