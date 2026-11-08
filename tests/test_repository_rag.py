"""Repository source retrieval, CLI defaults, and refresh behavior."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import run
from offline_agent.cli import main
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace


class RepositoryRagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = Workspace(self.root)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_source_code_and_documentation_have_repository_citations(self):
        self.put("src/orders.py", "def verify_basket(basket):\n    return basket is not None\n")
        self.put("README.md", "Basket verification architecture")
        kb = KnowledgeBase(workspace=self.workspace)
        result = json.loads(kb.retrieve("verify_basket"))
        self.assertEqual(result["results"][0]["source"], "repo/src/orders.py")
        self.assertEqual(result["results"][0]["start_line"], 1)
        self.assertEqual(result["results"][0]["end_line"], 2)
        self.assertEqual(result["documents_by_source"]["repo"], 2)

    def test_multiple_languages_and_build_files(self):
        for name in ("app.ts", "api.go", "Cargo.toml", "Dockerfile", "Makefile"):
            self.put(name, "uniqueprojectidentifier")
        kb = KnowledgeBase(workspace=self.root)
        self.assertEqual(kb.documents_by_source["repo"], 5)
        self.assertEqual(len(json.loads(kb.retrieve("uniqueprojectidentifier"))["results"]), 4)

    def test_reference_folder_is_not_double_indexed(self):
        self.put("knowledge/team.md", "uniquecompanypattern")
        self.put("main.py", "value = 1")
        kb = KnowledgeBase(self.root / "knowledge", workspace=self.workspace)
        matches = json.loads(kb.retrieve("uniquecompanypattern"))["results"]
        self.assertEqual([r["source"] for r in matches], ["user/team.md"])
        self.assertEqual(kb.documents_by_source["repo"], 1)

    def test_repository_filtering(self):
        for name in (".env", ".git/config", "node_modules/app.js", ".venv/lib/code.py"):
            self.put(name, "uniquerepotoken")
        self.put("large.py", "uniquerepotoken" + "x" * 64000)
        self.put("image.png", "uniquerepotoken")
        (self.root / "binary.py").write_bytes(b"uniquerepotoken\0")
        self.put("good.py", "uniquerepotoken")
        (self.root / "linked.py").symlink_to(self.root / "good.py")
        result = json.loads(KnowledgeBase(workspace=self.root).retrieve("uniquerepotoken"))
        self.assertEqual([r["source"] for r in result["results"]], ["repo/good.py"])

    def test_refresh_replaces_modified_and_deleted_files(self):
        self.put("code.py", "oldrepopattern")
        kb = KnowledgeBase(workspace=self.root)
        self.put("code.py", "newrepopattern")
        kb.refresh()
        self.assertEqual(json.loads(kb.retrieve("oldrepopattern"))["results"], [])
        self.assertTrue(json.loads(kb.retrieve("newrepopattern"))["results"])
        (self.root / "code.py").unlink()
        kb.refresh()
        self.assertEqual(json.loads(kb.retrieve("newrepopattern"))["results"], [])

    def test_cli_automatically_indexes_workspace(self):
        self.put("src/code.py", "uniquerepositoryterm")
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent:
            with contextlib.redirect_stdout(io.StringIO()) as display:
                self.assertEqual(main(["--workspace", str(self.root), "Explain uniquerepositoryterm"]), 0)
        kb = agent.call_args.kwargs["knowledge"]
        self.assertIn("repo/src/code.py", kb.retrieve("uniquerepositoryterm"))
        self.assertIn("1 repository files", display.getvalue())

    def test_indexing_repository_does_not_use_network(self):
        self.put("main.rs", "fn uniquerustpattern() {}")
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            kb = KnowledgeBase(workspace=self.root)
            result = kb.retrieve("uniquerustpattern")
        self.assertIn("repo/main.rs", result)
