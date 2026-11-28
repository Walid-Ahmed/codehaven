"""The keyword index kept between runs, re-indexing only changed files, and notes on code that is not in the repository."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run_state
from offline_agent.answers import MISSING_HEADING, missing_code
from offline_agent.dependencies import PythonDependencies
from offline_agent.javascript import JavaScriptDependencies
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace

AUTH = ("import os\nimport requests\nfrom jose import jwt\nfrom .gone import helper\n\n\n"
        "def login(token):\n    claims = jwt.decode(token, os.environ['KEY'])\n    return claims['sub']\n\n\n"
        "def fetch(url):\n    return requests.get(url).json()\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        (self.root / "app").mkdir(parents=True)
        (self.root / "app" / "__init__.py").write_text("")
        (self.root / "app" / "auth.py").write_text(AUTH)
        (self.root / "app" / "store.py").write_text("from .auth import login\n\n\ndef restock(item):\n    return item + 1\n")
        self.cache = Path(self.temp.name) / "index" / "repo.keywords"

    def same_index(self, one, other):
        """Two indexes answer alike: the same passages, outline, imports, and retrieval result."""
        self.assertEqual([(e["source"], e["start_line"], e["text"]) for e in one.entries],
                         [(e["source"], e["start_line"], e["text"]) for e in other.entries])
        self.assertEqual(one.dependencies.edges, other.dependencies.edges)
        self.assertEqual(one.dependencies.spans, other.dependencies.spans)
        self.assertEqual(one.dependencies.missing, other.dependencies.missing)
        self.assertEqual(one.symbols, other.symbols)
        self.assertEqual(one.retrieve("restock login token"), other.retrieve("restock login token"))


class SavedIndexTests(Base):
    def test_a_saved_index_is_reused_and_answers_alike(self):
        first = KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        self.assertTrue(self.cache.exists())
        self.assertEqual(first.reused, 0)
        again = KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        self.assertEqual(again.reused, again.documents)
        self.same_index(again, KnowledgeBase(workspace=Workspace(self.root)))

    def test_the_saved_index_holds_no_code_text_and_is_private(self):
        KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        saved = self.cache.read_text()
        self.assertNotIn("jwt.decode", saved)
        self.assertNotIn("return item + 1", saved)
        if os.name == "posix":
            self.assertEqual(self.cache.stat().st_mode & 0o077, 0)

    def test_a_changed_file_is_indexed_afresh(self):
        KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        (self.root / "app" / "store.py").write_text("def restock(item, count):\n    return item + count + 99\n")
        again = KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        self.assertEqual(again.reused, again.documents - 1)
        self.assertIn("count + 99", again.retrieve("restock count"))
        self.same_index(again, KnowledgeBase(workspace=Workspace(self.root)))

    def test_a_damaged_or_missing_index_means_a_fresh_build(self):
        self.cache.parent.mkdir(parents=True)
        self.cache.write_text("{not json")
        kb = KnowledgeBase(workspace=Workspace(self.root), cache=self.cache)
        self.assertEqual(kb.reused, 0)
        self.same_index(kb, KnowledgeBase(workspace=Workspace(self.root)))
        kb.forget()
        self.assertFalse(self.cache.exists())

    def test_nothing_is_written_without_a_cache(self):
        KnowledgeBase(workspace=Workspace(self.root))
        self.assertFalse(self.cache.parent.exists())
        self.assertEqual(sorted(p.name for p in (self.root / "app").iterdir()), ["__init__.py", "auth.py", "store.py"])


class RefreshTests(Base):
    def test_refresh_reindexes_only_what_changed_and_matches_a_fresh_build(self):
        kb = KnowledgeBase(workspace=Workspace(self.root))
        self.assertFalse(kb.refresh())
        (self.root / "app" / "store.py").write_text("def restock(item):\n    return item + 2\n")
        (self.root / "app" / "new.py").write_text("from .store import restock\n")
        self.assertTrue(kb.refresh())
        self.same_index(kb, KnowledgeBase(workspace=Workspace(self.root)))
        (self.root / "app" / "new.py").unlink()
        self.assertTrue(kb.refresh())
        self.same_index(kb, KnowledgeBase(workspace=Workspace(self.root)))


class MissingCodeTests(Base):
    def test_python_imports_of_code_not_here_are_found_and_the_standard_library_is_not(self):
        found = PythonDependencies({"app/auth.py": AUTH, "app/__init__.py": ""}).missing["app/auth.py"]
        self.assertEqual([(row["module"], row["name"], row["local"]) for row in found],
                         [("requests", "requests", False), ("jose", "jwt", False), (".gone", "helper", True)])

    def test_javascript_imports_of_code_not_here_are_found(self):
        files = {"a.js": "import jwt, { decode as d } from 'jsonwebtoken';\nimport b from './b';\nimport c from './gone';\n"
                         "const fs = require('fs');\nconst _ = require('lodash');\n", "b.js": "export default 1;\n"}
        found = JavaScriptDependencies(files).missing["a.js"]
        self.assertEqual([(row["module"], row["name"], row["local"]) for row in found],
                         [("jsonwebtoken", "jwt", False), ("jsonwebtoken", "d", False), ("./gone", "c", True),
                          ("lodash", "_", False)])

    def test_only_imports_used_on_the_cited_lines_are_noted(self):
        workspace = Workspace(self.root)
        kb = KnowledgeBase(workspace=workspace)
        notes = missing_code("`login` decodes the token (app/auth.py:7-9).", workspace, kb)
        self.assertEqual(len(notes), 1)
        self.assertIn("`jose`", notes[0])
        self.assertIn("line 3 of app/auth.py", notes[0])
        self.assertEqual(missing_code("`restock` adds one (app/store.py:4-5).", workspace, kb), [])

    def test_an_answer_says_what_its_code_depends_on_that_is_not_here(self):
        workspace = Workspace(self.root)
        kb = KnowledgeBase(workspace=workspace)

        class Model:
            prompt_budget = 12_000

            def chat(self, messages, schema, on_text=None):
                return {"action": "finish", "path": "", "message": "Done",
                        "content": "`fetch` returns the JSON body of a GET request (app/auth.py:12-13)."}

        lines = []
        state = run_state(Model(), workspace, "What does fetch in app/auth.py return?", knowledge=kb, output=lines.append)
        self.assertIn(MISSING_HEADING, state["final_text"])
        self.assertIn("`requests`", state["final_text"])
        self.assertNotIn("`jose`", state["final_text"])
        # The note is not evidence, so it cannot make a checked answer unverified.
        self.assertEqual(state["issues"], [])
        self.assertEqual(json.loads(kb.retrieve("fetch"))["dependency_parse_failures"], 0)


if __name__ == "__main__":
    unittest.main()
