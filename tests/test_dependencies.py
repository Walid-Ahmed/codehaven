"""Python import following: which file imports which, and the symbols and line spans found by parsing."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.dependencies import PythonDependencies
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import READ_BYTES


class ImportResolutionTests(unittest.TestCase):
    def test_absolute_imports_aliases_and_symbols(self):
        files = {"app.py": "import helpers as h\nfrom models import Order as O\n",
                 "helpers.py": "def calculate(): pass\n", "models.py": "class Order: pass\n"}
        graph = PythonDependencies(files)
        self.assertEqual(graph.edges["app.py"], {"helpers.py": [''], "models.py": ['Order']})
        self.assertEqual(graph.definitions["models.py"]["Order"], 1)

    def test_relative_imports_and_package_initializers(self):
        files = {"pkg/__init__.py": "", "pkg/api.py": "from . import helpers\nfrom .models import Item\n",
                 "pkg/helpers.py": "", "pkg/models.py": "class Item: pass"}
        edges = PythonDependencies(files).edges["pkg/api.py"]
        self.assertIn("pkg/helpers.py", edges)
        self.assertEqual(edges["pkg/models.py"], ['Item'])
        self.assertIn("pkg/__init__.py", edges)

    def test_parent_relative_and_src_layout(self):
        files = {"src/pkg/api/main.py": "from ..helpers import compute\nimport pkg.models\n",
                 "src/pkg/helpers.py": "def compute(): pass", "src/pkg/models.py": ""}
        edges = PythonDependencies(files).edges["src/pkg/api/main.py"]
        self.assertEqual(set(edges), {"src/pkg/helpers.py", "src/pkg/models.py"})

    def test_stdlib_external_dynamic_and_invalid_imports_ignored(self):
        files = {"app.py": "import os\nimport requests\n__import__('helpers')\nfrom ... import escape\n",
                 "helpers.py": "raise RuntimeError('never execute')"}
        self.assertEqual(PythonDependencies(files).edges["app.py"], {})

    def test_syntax_error_keeps_other_files_available(self):
        graph = PythonDependencies({"bad.py": "def broken(", "ok.py": "import bad"})
        self.assertEqual(graph.parse_failures, 1)
        self.assertIn("bad.py", graph.edges["ok.py"])


class DependencyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def retrieve(self, query="uniqueentrypoint"):
        return json.loads(KnowledgeBase(workspace=self.root).retrieve(query))

    def test_dependency_included_without_query_overlap(self):
        self.put("app.py", "from helpers import calculate\ndef uniqueentrypoint():\n    return calculate()\n")
        self.put("helpers.py", "def calculate():\n    return 42\n")
        result = self.retrieve()
        dependency = next(r for r in result["results"] if r["source"] == "repo/helpers.py")
        self.assertEqual(dependency["kind"], "dependency")
        self.assertEqual(dependency["via"], "repo/app.py")
        self.assertEqual(dependency["depth"], 1)
        self.assertIn("return 42", dependency["text"])

    def test_named_definition_passage_selected(self):
        self.put("app.py", "from helpers import calculate\ndef uniqueentrypoint(): return calculate()\n")
        self.put("helpers.py", "# filler\n" * 70 + "def calculate():\n    return 42\n")
        result = self.retrieve()
        dependency = next(r for r in result["results"] if r["source"] == "repo/helpers.py")
        self.assertLessEqual(dependency["start_line"], 71)
        self.assertGreaterEqual(dependency["end_line"], 71)
        self.assertIn("def calculate", dependency["text"])

    def test_two_hops_and_depth_limit(self):
        self.put("app.py", "import middle\ndef uniqueentrypoint(): pass\n")
        self.put("middle.py", "import leaf\n")
        self.put("leaf.py", "import beyond\n")
        self.put("beyond.py", "value = 5\n")
        result = self.retrieve()
        sources = [r["source"] for r in result["results"]]
        self.assertIn("repo/middle.py", sources)
        self.assertIn("repo/leaf.py", sources)
        self.assertNotIn("repo/beyond.py", sources)
        self.assertTrue(result["dependency_limit_reached"])

    def test_cycles_deduplicate_passages(self):
        self.put("app.py", "import helpers\ndef uniqueentrypoint(): pass\n")
        self.put("helpers.py", "import app\n")
        result = self.retrieve()
        self.assertEqual(len(result["results"]), 2)
        self.assertFalse(result["dependency_limit_reached"])

    def test_dependency_count_and_json_budget(self):
        self.put("app.py", "\n".join(f"import helper{i}" for i in range(8)) + "\ndef uniqueentrypoint(): pass\n")
        for i in range(8):
            self.put(f"helper{i}.py", "# " + "漢" * 1700)
        raw = KnowledgeBase(workspace=self.root).retrieve("uniqueentrypoint")
        result = json.loads(raw)
        self.assertLessEqual(len([r for r in result["results"] if r["kind"] == "dependency"]), 4)
        self.assertLessEqual(len(raw.encode()), 16000)
        self.assertTrue(result["dependency_limit_reached"])

    def test_unindexed_symlinks_and_oversize_targets_not_followed(self):
        self.put("app.py", "import hidden\nimport large\ndef uniqueentrypoint(): pass\n")
        self.put("outside.txt", "never executed")
        (self.root / "hidden.py").symlink_to(self.root / "outside.txt")
        self.put("large.py", "x" * (READ_BYTES + 1))
        self.assertEqual([r["source"] for r in self.retrieve()["results"]], ["repo/app.py"])

    def test_refresh_rebuilds_import_graph(self):
        self.put("app.py", "import first\ndef uniqueentrypoint(): pass\n")
        self.put("first.py", "value = 1")
        self.put("second.py", "value = 2")
        kb = KnowledgeBase(workspace=self.root)
        self.put("app.py", "import second\ndef uniqueentrypoint(): pass\n")
        kb.refresh()
        sources = [r["source"] for r in json.loads(kb.retrieve("uniqueentrypoint"))["results"]]
        self.assertIn("repo/second.py", sources)
        self.assertNotIn("repo/first.py", sources)

    def test_indexing_never_imports_code_or_uses_network(self):
        self.put("app.py", "import helpers\ndef uniqueentrypoint(): pass\n")
        self.put("helpers.py", "raise RuntimeError('must never execute')\n")
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            result = self.retrieve()
        self.assertIn("repo/helpers.py", [r["source"] for r in result["results"]])
