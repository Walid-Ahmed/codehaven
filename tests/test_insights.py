"""Test gaps and the project tour: built by parsing, cited by real lines, and never saved as long-term memory."""
import tempfile
import unittest
from pathlib import Path

from offline_agent import insights
from offline_agent.agent import Session
from offline_agent.knowledge import KnowledgeBase
from offline_agent.memory import Memory
from offline_agent.workspace import Workspace


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, _schema, on_text=None):
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        for name, text in {
            "README.md": "# Shop\n",
            "main.py": '"""Start the shop."""\nfrom shop.store import Store\nfrom shop import pricing\n\n\ndef main():\n    return Store()\n',
            "shop/__init__.py": "",
            "shop/store.py": "from shop.pricing import with_tax\n\n\nclass Store:\n    def add(self, item):\n        return item\n\n"
                             "    def remove(self, item):\n        return item\n\n    def _log(self):\n        pass\n",
            "shop/pricing.py": "def with_tax(price):\n    return price * 1.2\n\n\ndef discount(price):\n    return price\n",
            "tests/test_store.py": "from shop.store import Store\n\n\ndef test_add():\n    assert Store().add(1) == 1\n"
                                   "    assert with_tax\n",
        }.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.workspace = Workspace(self.root)
        self.knowledge = KnowledgeBase(workspace=self.workspace, context_budget=4000)

    def test_requests_are_recognised(self):
        for task, kind in [("Which functions have no tests?", "gaps"), ("Show me the test gaps", "gaps"),
                           ("What is untested?", "gaps"), ("Give me a project tour", "tour"),
                           ("Where should I start reading?", "tour"), ("Which files should I read first?", "tour"),
                           ("Explain the tests", None), ("How does Store.add work?", None)]:
            self.assertEqual(insights.requested(task), kind, task)

    def test_gaps_list_functions_no_test_mentions_without_the_model(self):
        memory = Memory(self.workspace, Path(self.temp.name, "memory"))
        session = Session(ScriptedModel([]), self.workspace, knowledge=self.knowledge, memory=memory, output=lambda _l: None)
        self.assertEqual(session.ask("Which functions have no tests?"), 0)
        answer = session.last_state["final_text"]
        self.assertIn("2 of 4 functions and methods are not mentioned by name in any of the 1 test file", answer)
        self.assertIn("- shop/store.py:8-9 `Store.remove`", answer)
        self.assertIn("- shop/pricing.py:5-6 `discount`", answer)
        for name in ("Store.add", "with_tax", "_log", "`main`", "test_add"):
            self.assertNotIn(name, answer)
        self.assertEqual(memory.notes, [])  # A saved list would not notice a new test.

    def test_tour_follows_imports_from_the_entry_point(self):
        order = insights.reading_order(list(insights.code_files(self.workspace)), self.knowledge.dependencies.edges)
        self.assertEqual([path for path, _ in order], ["main.py", "shop/pricing.py", "shop/store.py"])
        self.assertIn("nothing imports it", order[0][1])
        self.assertIn("and 1 other file", order[1][1])
        model = ScriptedModel([{"sentence": "Starts the shop by building a Store."}, ValueError("bad"),
                               {"sentence": "Holds the Store class."}])
        session = Session(model, self.workspace, knowledge=self.knowledge, output=lambda _l: None)
        self.assertEqual(session.ask("Give me a project tour"), 0)
        lines = session.last_state["final_text"].splitlines()
        self.assertTrue(lines[1].startswith("1. README.md:1-1 Start here"))
        self.assertTrue(lines[2].startswith("2. main.py:1-7 Starts the shop by building a Store. Why now: the entry point"))
        self.assertIn("(no description", lines[3])  # pricing.py has no opening comment to fall back on.
        self.assertTrue(lines[4].startswith("4. shop/store.py:1-12 Holds the Store class."))
        self.assertNotIn("tests/", session.last_state["final_text"])
        self.assertIn("Start the shop", model.prompts[0][1]["content"])

    def test_no_imports_means_no_invented_order(self):
        (self.root / "main.py").write_text("print(1)\n")
        (self.root / "shop" / "store.py").write_text("X = 1\n")
        (self.root / "tests" / "test_store.py").write_text("")
        knowledge = KnowledgeBase(workspace=self.workspace, context_budget=4000)
        found = insights.tour(ScriptedModel([]), self.workspace, knowledge, lambda _l: None)
        self.assertIn("could not find imports", found["text"])
        self.assertIn("Start with README.md", found["text"])


if __name__ == "__main__":
    unittest.main()
