import json
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run
from offline_agent.workspace import Workspace


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)

    def put(self, name, text):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def search(self, query, scope=""):
        return json.loads(self.ws.search(query, scope))

    def test_literal_case_sensitive_matches_with_line_numbers(self):
        self.put("code.py", "needle\nNeedle\nprint('needle.*')\n")
        result = self.search("needle.*")
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["path"], "code.py")
        self.assertEqual(result["matches"][0]["line"], 3)
        self.assertEqual(len(self.search("needle")["matches"]), 2)

    def test_file_and_directory_scope(self):
        self.put("src/a.py", "needle\n")
        self.put("other.py", "needle\n")
        for scope in ("src", "src/a.py"):
            result = self.search("needle", scope)
            self.assertEqual([m["path"] for m in result["matches"]], ["src/a.py"])
            self.assertEqual(result["files_searched"], 1)

    def test_invalid_queries(self):
        for query in ("", "  ", "a\nb", "a\rb", "a\0b", "a" * 257):
            with self.subTest(query=repr(query)), self.assertRaises(ValueError):
                self.ws.search(query)

    def test_path_restrictions(self):
        for scope in ("../outside", "/tmp", ".git", "missing"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.ws.search("needle", scope)
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.ws.search("needle", "link")

    def test_hidden_excluded_binary_and_large_files_skipped(self):
        self.put(".env", "needle")
        self.put("node_modules/a.js", "needle")
        self.put(".git/config", "needle")
        self.put("large.py", "needle" + "x" * 64000)
        (self.root / "binary").write_bytes(b"needle\0")
        (self.root / "invalid").write_bytes(b"needle\xff")
        self.put("visible.py", "needle")
        (self.root / "link.py").symlink_to(self.root / "visible.py")
        result = self.search("needle")
        self.assertEqual([m["path"] for m in result["matches"]], ["visible.py"])
        self.assertEqual(result["files_skipped"], 2)

    def test_no_matches(self):
        self.put("code.py", "unrelated")
        result = self.search("needle")
        self.assertEqual(result["matches"], [])
        self.assertFalse(result["limit_reached"])
        self.assertEqual(result["query"], "needle")
        self.assertEqual(result["scope"], ".")
        self.assertIn("does not establish", result["note"])

    def test_fastapi_literal_is_not_an_extension_filter(self):
        self.put("app/main.py", 'app = FastAPI(title="ShopSphere")\n')
        correct = self.search("FastAPI(")
        self.assertEqual(correct["matches"][0]["path"], "app/main.py")
        malformed = self.search("FastAPI(.py")
        self.assertEqual(malformed["matches"], [])
        self.assertEqual(malformed["query"], "FastAPI(.py")

    def test_match_limit(self):
        self.put("code.py", "needle\n" * 70)
        result = self.search("needle")
        self.assertEqual(len(result["matches"]), 50)
        self.assertTrue(result["limit_reached"])

    def test_unicode_output_and_snippets_bounded(self):
        self.put("code.py", ("x" * 500 + "needle" + "漢" * 500 + "\n") * 30)
        raw = self.ws.search("needle")
        result = json.loads(raw)
        self.assertLessEqual(len(raw.encode()), 16000)
        self.assertTrue(result["limit_reached"])
        self.assertTrue(all(m["snippet_truncated"] and "needle" in m["text"]
                            and len(m["text"]) <= 320 for m in result["matches"]))

    def test_directory_scope_avoids_global_candidate_limit(self):
        for i in range(1000):
            self.put(f"a{i:04}.py", "irrelevant")
        self.put("src/z.py", "needle")
        self.assertTrue(self.search("needle")["candidate_limit_reached"])
        self.assertEqual(len(self.search("needle", "src")["matches"]), 1)

    def test_graph_search_result_reaches_model(self):
        self.put("code.py", "needle\n")
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                first = len(calls) == 1
                return {"action": "search" if first else "finish", "path": "",
                        "content": "needle" if first else "Found needle [code.py:1].", "message": "Find needle"}

        self.assertEqual(run(Model(), self.ws, "Find needle", output=lambda _: None), 0)
        self.assertIn('"path": "code.py"', calls[1][-1]["content"])
        self.assertIn('"line": 1', calls[1][-1]["content"])

    def test_graph_search_failure_reaches_model(self):
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return {"action": "search" if len(calls) == 1 else "finish", "path": "../x",
                        "content": "needle", "message": "Search"}

        run(Model(), self.ws, "Find", output=lambda _: None)
        self.assertIn("Tool error", calls[1][-1]["content"])


class EnclosingPartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)

    def matches(self, query):
        return json.loads(self.ws.search(query))["matches"]

    def test_python_matches_name_the_enclosing_function_and_mark_definitions(self):
        (self.root / "pricing.py").write_text("LIMIT = apply_discount\n\n\ndef apply_discount(p):\n    return p\n\n\n"
                                              "class Cart:\n    def total(self, p):\n        return apply_discount(p)\n")
        found = self.matches("apply_discount")
        self.assertNotIn("in", found[0])
        self.assertEqual((found[1]["in"], found[1]["defines"]), ("apply_discount", True))
        self.assertEqual(found[2]["in"], "Cart.total")
        self.assertNotIn("defines", found[2])

    def test_other_languages_and_broken_python_use_the_nearest_definition(self):
        (self.root / "cart.js").write_text("function total(p) {\n  return discount(p);\n}\n")
        (self.root / "broken.py").write_text("def go(:\n    discount()\n")
        by_path = {match["path"]: match for match in self.matches("discount(")}
        self.assertEqual(by_path["cart.js"]["in"], "total")
        self.assertEqual(by_path["broken.py"]["in"], "go")
