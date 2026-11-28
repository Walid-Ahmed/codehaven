"""Changing code: only with an Editor, only after the diff is approved, never outside the rules, and undoable."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent import edits
from offline_agent.agent import run_state
from offline_agent.coder import Editor, requested
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace

STORE = "def low_stock(items, threshold):\n    return [i for i in items if i.count < threshold]\n"


class Model:
    prompt_budget = 12_000

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, schema, on_text=None):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


FIX = {"summary": "Include items exactly at the threshold.",
       "edits": [{"path": "store.py", "old": "i.count < threshold", "new": "i.count <= threshold"}]}


class EditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "repo")
        self.root.mkdir()
        (self.root / "store.py").write_text(STORE)
        (self.root / ".env").write_text("KEY=1\n")
        self.ws = Workspace(self.root)
        self.journal = Path(self.temp.name, "undo", "repo.json")
        self.asked = []

    def editor(self, answer=True):
        return Editor(confirm=lambda diff, summary: self.asked.append(diff) or answer, journal=self.journal)

    def ask(self, model, task, answer=True, **options):
        return run_state(model, self.ws, task, knowledge=KnowledgeBase(workspace=self.ws), editor=self.editor(answer),
                         output=lambda _: None, **options)

    def test_requests_are_recognised(self):
        self.assertEqual(requested("Fix low_stock so items at the threshold count"), "change")
        self.assertEqual(requested("Rename low_stock to below_threshold"), "rename")
        self.assertEqual(requested("Write tests for low_stock"), "tests")
        self.assertEqual(requested("Add docstrings to store.py"), "docstrings")
        self.assertEqual(requested("Fix the code until the tests pass"), "fix_tests")
        self.assertIsNone(requested("Explain low_stock"))

    def test_without_an_editor_nothing_is_changed(self):
        model = Model({"action": "finish", "path": "", "message": "Done", "points": [],
                       "content": "Change < to <= at store.py:2."})
        run_state(model, self.ws, "Fix low_stock in store.py so items at the threshold count", output=lambda _: None)
        self.assertEqual((self.root / "store.py").read_text(), STORE)

    def test_an_approved_change_is_applied_and_can_be_undone(self):
        state = self.ask(Model(FIX), "Fix low_stock in store.py so items at the threshold count")
        self.assertIn("i.count <= threshold", (self.root / "store.py").read_text())
        self.assertIn("-    return [i for i in items if i.count < threshold]", self.asked[0])
        self.assertIn("Changed store.py", state["final_text"])
        self.assertFalse(self.journal.is_relative_to(self.root))
        self.assertEqual(edits.undo(self.ws, self.journal), FIX["summary"])
        self.assertEqual((self.root / "store.py").read_text(), STORE)
        with self.assertRaises(edits.EditError):
            edits.undo(self.ws, self.journal)  # Nothing left to undo.

    def test_a_declined_change_is_not_applied(self):
        state = self.ask(Model(FIX), "Fix low_stock in store.py", answer=False)
        self.assertEqual((self.root / "store.py").read_text(), STORE)
        self.assertIn("you declined it", state["final_text"])

    def test_an_undo_never_discards_a_later_edit(self):
        self.ask(Model(FIX), "Fix low_stock in store.py")
        (self.root / "store.py").write_text("# my own edit\n")
        with self.assertRaisesRegex(edits.EditError, "changed again"):
            edits.undo(self.ws, self.journal)
        self.assertEqual((self.root / "store.py").read_text(), "# my own edit\n")

    def test_edits_must_match_once_and_stay_within_the_rules(self):
        for edit in ({"path": "store.py", "old": "threshold", "new": "x"},           # Occurs twice.
                     {"path": "store.py", "old": "missing text", "new": "x"},
                     {"path": ".env", "old": "KEY=1", "new": "KEY=2"},                # Hidden.
                     {"path": "../outside.py", "old": "", "new": "x"},
                     {"path": "store.py", "old": "", "new": "x"}):                    # Exists, so not created.
            with self.subTest(edit=edit), self.assertRaises(edits.EditError):
                edits.planned(self.ws, [edit])
        self.assertEqual(edits.planned(self.ws, [{"path": "tests/test_store.py", "old": "", "new": "x\n"}])[0][1], None)

    def test_a_proposal_that_does_not_apply_is_sent_back_once(self):
        wrong = {"summary": "s", "edits": [{"path": "store.py", "old": "i.count<threshold", "new": "x"}]}
        model = Model(wrong, FIX)
        self.ask(model, "Fix low_stock in store.py")
        self.assertIn("cannot be applied", model.prompts[1])
        self.assertIn("<= threshold", (self.root / "store.py").read_text())

    def test_rename_is_plain_code_across_files(self):
        (self.root / "app.py").write_text("from store import low_stock\nprint(low_stock([], 1))\n")
        self.ask(Model(), "Rename low_stock to below_threshold")
        self.assertIn("def below_threshold", (self.root / "store.py").read_text())
        self.assertEqual((self.root / "app.py").read_text(), "from store import below_threshold\nprint(below_threshold([], 1))\n")
        edits.undo(self.ws, self.journal)
        self.assertEqual((self.root / "store.py").read_text(), STORE)

    def test_writing_tests_creates_a_new_file_that_undo_removes(self):
        tests = {"summary": "Tests for low_stock.", "edits": [{"path": "tests/test_store.py", "old": "",
                                                              "new": "from store import low_stock\n"}]}
        self.ask(Model(tests), "Write tests for low_stock")
        self.assertTrue((self.root / "tests" / "test_store.py").exists())
        edits.undo(self.ws, self.journal)
        self.assertFalse((self.root / "tests" / "test_store.py").exists())

    def test_fix_until_the_tests_pass_sends_failures_back(self):
        runs = iter(["Exit code: 1\nAssertionError: 3 != 2", "Exit code: 0\nOK"])
        editor = Editor(confirm=lambda *_: True, journal=self.journal, confirm_tests=lambda: True)
        model = Model(FIX)
        with patch("offline_agent.coder.run_tests", side_effect=lambda *a, **k: next(runs)):
            state = run_state(model, self.ws, "Fix store.py until the tests pass", editor=editor, output=lambda _: None)
        self.assertIn("AssertionError: 3 != 2", model.prompts[0])
        self.assertIn("The tests pass", state["final_text"])


if __name__ == "__main__":
    unittest.main()
