import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import run
from offline_agent.knowledge import KnowledgeBase, chunks, tokens
from offline_agent.workspace import Workspace


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, text):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def test_bundled_langgraph_guide(self):
        result = json.loads(KnowledgeBase().retrieve("StateGraph conditional edges"))
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["source"], "builtin/langgraph.md")
        self.assertIn("StateGraph", result["results"][0]["text"])

    def test_custom_reference_ranking_and_citation(self):
        self.put("policies.md", "# Policy\n\nShopSphere orders require validation.\n")
        self.put("other.txt", "An unrelated gardening document.")
        result = json.loads(KnowledgeBase(self.root).retrieve("ShopSphere orders validation"))
        passage = result["results"][0]
        self.assertEqual(passage["source"], "user/policies.md")
        self.assertEqual(passage["start_line"], 1)
        self.assertEqual(passage["end_line"], 3)
        self.assertGreater(passage["score"], 0)

    def test_no_match_and_invalid_query(self):
        kb = KnowledgeBase()
        self.assertEqual(json.loads(kb.retrieve("xyzzyunmatched"))["results"], [])
        for query in ("", " ", "x" * 1001):
            with self.assertRaises(ValueError):
                kb.retrieve(query)

    def test_unsupported_hidden_symlink_binary_and_oversize_excluded(self):
        self.put("good.rst", "zqxuniquedoc")
        self.put(".hidden.md", "zqxuniquedoc")
        self.put("ignored.pdf", "zqxuniquedoc")
        self.put("big.md", "zqxuniquedoc" + "x" * 64000)
        (self.root / "bad.md").write_bytes(b"zqxuniquedoc\0")
        (self.root / "link.md").symlink_to(self.root / "good.rst")
        result = json.loads(KnowledgeBase(self.root).retrieve("zqxuniquedoc"))
        self.assertEqual([r["source"] for r in result["results"]], ["user/good.rst"])
        self.assertEqual(result["skipped_documents"], 1)

    def test_chunk_lines_and_long_lines(self):
        text = "\n".join("line" for _ in range(65))
        pieces = list(chunks(text))
        self.assertEqual([(a, b) for a, b, _ in pieces], [(1, 30), (31, 60), (61, 65)])
        pieces = list(chunks("x" * 5000))
        self.assertTrue(all(a == b == 1 and len(t) <= 1800 for a, b, t in pieces))
        self.assertEqual("".join(t for _, _, t in pieces), "x" * 5000)

    def test_bounded_results_and_index_document_limit(self):
        for i in range(25):
            self.put(f"doc{i:03}.md", "needle " + "漢" * 1500)
        with patch("offline_agent.knowledge.MAX_DOCUMENTS", 20):
            result = KnowledgeBase(self.root).retrieve("needle")
        decoded = json.loads(result)
        self.assertLessEqual(len(decoded["results"]), 4)
        self.assertLessEqual(len(result.encode()), 16000)
        self.assertTrue(decoded["results_limited"])
        self.assertTrue(decoded["index_limited"])
        self.assertEqual(decoded["documents"], 20)

    def test_camel_case_and_snake_case_names_match_plain_words(self):
        self.assertEqual(tokens("applyDiscount apply_discount HTTPServer"), ["apply", "discount", "apply", "discount", "httpserver"])

    def test_word_endings_do_not_stop_a_match(self):
        self.put("repo/test_store.py", "def test_restock_increases_quantity():\n    assert True\n")
        self.put("repo/other.py", "def unrelated():\n    return 1\n")
        result = json.loads(KnowledgeBase(workspace=self.root / "repo").retrieve("Which test checks restocking?"))["results"]
        self.assertEqual(result[0]["source"], "repo/test_store.py")
        self.assertEqual(tokens("pricing price"), ["pric", "pric"])

    def test_bundled_notes_do_not_crowd_out_project_code(self):
        self.put("repo/pricing.py", "def apply_discount(price):\n    return price\n")
        kb = KnowledgeBase(workspace=self.root / "repo")
        sources = [row["source"] for row in json.loads(kb.retrieve("Which function calls apply_discount?"))["results"]]
        self.assertEqual(sources, ["repo/pricing.py"])
        sources = [row["source"] for row in json.loads(kb.retrieve("How do LangGraph conditional edges work?"))["results"]]
        self.assertTrue(any(source.startswith("builtin/") for source in sources))

    def test_deleted_reference_not_in_next_index(self):
        self.put("policy.txt", "uniquepolicytoken")
        self.assertTrue(json.loads(KnowledgeBase(self.root).retrieve("uniquepolicytoken"))["results"])
        (self.root / "policy.txt").unlink()
        self.assertFalse(json.loads(KnowledgeBase(self.root).retrieve("uniquepolicytoken"))["results"])

    def test_bad_root_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeBase(self.root / "missing")

    def test_loading_and_retrieval_do_not_open_network_connections(self):
        self.put("guide.md", "offline knowledge")
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            result = KnowledgeBase(self.root).retrieve("offline knowledge")
        self.assertIn("user/guide.md", result)

    def test_graph_initial_context_and_retrieve_tool(self):
        self.put("guide.md", "uniqueprojectpattern means validate input before storage")
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return {"action": "retrieve" if len(calls) == 1 else "finish", "path": "",
                        "content": "uniqueprojectpattern" if len(calls) == 1 else "Validate input before storage [user/guide.md:1].", "message": "Consult references"}

        result = run(Model(), Workspace(self.root), "Explain uniqueprojectpattern",
                     knowledge=KnowledgeBase(self.root), output=lambda _: None)
        self.assertEqual(result, 0)
        self.assertIn("user/guide.md", calls[0][-1]["content"])
        self.assertIn('"start_line": 1', calls[1][-1]["content"])

    def test_model_cannot_select_arbitrary_reference_root(self):
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return {"action": "retrieve" if len(calls) == 1 else "finish", "path": "/tmp",
                        "content": "guide", "message": "Consult references"}

        run(Model(), Workspace(self.root), "Task", knowledge=KnowledgeBase(), output=lambda _: None)
        self.assertIn("path must be empty", calls[1][-1]["content"])

    def test_complete_final_answer_is_displayed(self):
        output = []

        class Model:
            def chat(self, messages, schema):
                return {"action": "finish", "path": "", "content": "Validate orders. [user/team.md:1-1]",
                        "message": "Answer ready"}

        run(Model(), Workspace(self.root), "Question", output=output.append)
        self.assertIn("Validate orders. [user/team.md:1-1]", output)
