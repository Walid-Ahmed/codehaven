"""Reviewing a file: one function per model call, findings cited by looked-up lines, and nothing changed."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import REVIEW, Session
from offline_agent.review import REVIEW_PARTS, review, reviewable
from offline_agent.workspace import Workspace

STORE = ('"""Store."""\n\n\nclass Store:\n    def __init__(self):\n        self.items = {}\n\n'
         "    def restock(self, name, amount):\n        self.items[name] += amount\n\n"
         '    def low_stock(self, limit):\n        """Items at or below the limit."""\n'
         "        return [n for n, c in self.items.items() if c < limit]\n\n\ndef helper():\n    return 1\n")


def verdict(kind, finding=""):
    return {"promise": "p", "does": "d", "edge": "e", "verdict": kind, "finding": finding}


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


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "store.py").write_text(STORE)
        self.workspace = Workspace(self.root)

    def test_review_requests_are_recognised(self):
        for task in ("Review store.py", "please review `store.py`", "Code review of store.py", "Find bugs in store.py"):
            self.assertTrue(REVIEW.search(task), task)
        for task in ("Explain store.py", "How does the review queue in store.py work?"):
            self.assertFalse(REVIEW.search(task), task)

    def test_each_function_is_checked_alone_and_findings_cite_its_lines(self):
        model = ScriptedModel([verdict("ok"), verdict("missing check", "A name not in items raises KeyError."),
                               verdict("bug", "Uses < but promises at or below; use <=."), ValueError("bad reply")])
        session = Session(model, self.workspace, output=lambda _line: None)
        self.assertEqual(session.ask("Review store.py"), 0)
        answer = session.last_state["final_text"].splitlines()
        self.assertIn("2 likely problems found", answer[0])
        self.assertEqual(answer[1], "- store.py:11-13 `Store.low_stock` Bug: Uses < but promises at or below; use <=.")
        self.assertEqual(answer[2], "- store.py:8-9 `Store.restock` Missing check: A name not in items raises KeyError.")
        self.assertIn("`Store.__init__`", answer[3])
        self.assertIn("Could not be checked: `helper`", answer[4])
        self.assertEqual(len(model.prompts), 4)
        shown = model.prompts[2][1]["content"]
        self.assertIn("class Store:", shown)
        self.assertIn("c < limit", shown)
        self.assertNotIn("def restock", shown)  # Only the one function is shown.
        self.assertEqual((self.root / "store.py").read_text(), STORE)

    def test_a_function_is_shown_the_first_lines_of_what_it_calls(self):
        (self.root / "cart.py").write_text(
            'def price_of(name, table):\n    """Price of one item; raises KeyError for an unknown name."""\n    return table[name]\n\n\n'
            "class Cart:\n    def add(self, name):\n        self.items.append(name)\n\n"
            "    def total(self, table, extra):\n        self.add(extra)\n        seen = {}.get(extra)\n"
            "        return sum(price_of(name) for name in self.items)\n\n    def get(self, key):\n        return key\n")
        model = ScriptedModel([verdict("ok")] * 4)
        review(model, self.workspace, "cart.py", output=lambda _line: None)
        alone, total = model.prompts[0][1]["content"], model.prompts[2][1]["content"]
        self.assertNotIn("It calls these functions", alone)  # price_of calls nothing: its prompt is as before.
        self.assertIn("It calls these functions of the same file", total)
        self.assertIn("Cart.add:\n    def add(self, name):", total)
        self.assertIn('price_of:\ndef price_of(name, table):\n    """Price of one item; raises KeyError for an unknown name."""', total)
        self.assertNotIn("return table[name]", total)  # Signatures only, never the bodies.
        self.assertNotIn("Cart.get", total)  # {}.get(extra) is a call on another object.

    def test_a_file_without_functions_is_checked_whole_and_long_files_say_what_was_left_out(self):
        (self.root / "settings.toml").write_text("limit = 3\n")
        found = review(ScriptedModel([verdict("ok")]), self.workspace, "settings.toml", output=lambda _line: None)
        self.assertIn("no likely problems found", found["text"])
        (self.root / "many.py").write_text("".join(f"def f{n}():\n    return {n}\n\n\n" for n in range(REVIEW_PARTS + 3)))
        chosen, total = reviewable("many.py", (self.root / "many.py").read_text())
        self.assertEqual((len(chosen), total), (REVIEW_PARTS, REVIEW_PARTS + 3))
        found = review(ScriptedModel([verdict("ok")] * REVIEW_PARTS), self.workspace, "many.py", output=lambda _line: None)
        self.assertIn(f"Only the first {REVIEW_PARTS} of {REVIEW_PARTS + 3}", found["text"])


if __name__ == "__main__":
    unittest.main()
