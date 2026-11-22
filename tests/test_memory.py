"""Long-term memory: opt-in, checked answers only, and never trusted once the code they cite has changed."""
import os
import contextlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import Session
from offline_agent.cli import main
from offline_agent.memory import NOTES, Memory
from offline_agent.workspace import Workspace

PRICING = "RATE = 0.1\n\n\ndef apply_discount(price):\n    return price * (1 - RATE)\n"


def finish(content, points=()):
    return {"action": "finish", "path": "", "content": content, "message": "Done", "points": list(points)}


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, _schema, on_text=None):
        self.prompts.append(messages)
        return self.replies.pop(0)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        self.root.mkdir()
        (self.root / "pricing.py").write_text(PRICING)
        (self.root / "stock.py").write_text("def restock(item):\n    return item\n")
        self.folder = Path(self.temp.name, "memory")
        self.workspace = Workspace(self.root)

    def memory(self):
        return Memory(self.workspace, self.folder)

    def test_notes_are_saved_outside_the_repository_for_this_user_only(self):
        memory = self.memory()
        self.assertTrue(memory.add("How is the discount applied?", "apply_discount multiplies by 1 - RATE (pricing.py:4-5)."))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["pricing.py", "stock.py"])
        self.assertEqual(memory.path.parent, self.folder)
        if os.name != "nt":  # Windows keeps a user's files private through the profile folder, not file modes.
            self.assertEqual(stat.S_IMODE(memory.path.stat().st_mode), 0o600)
        self.assertEqual([note["question"] for note in self.memory().notes], ["How is the discount applied?"])

    def test_answers_without_readable_citations_are_not_saved(self):
        memory = self.memory()
        for answer in ("It applies a discount.", "See missing.py:1-2.", "See .env:1."):
            with self.subTest(answer=answer):
                self.assertFalse(memory.add("How is the discount applied?", answer))
        self.assertFalse(memory.path.exists())

    def test_recall_needs_shared_words_and_unchanged_files(self):
        memory = self.memory()
        memory.add("How is the discount applied?", "apply_discount multiplies by 1 - RATE (pricing.py:4-5).")
        memory.add("What does restock return?", "restock returns its item (stock.py:1-2).")
        self.assertEqual([note["question"] for note in memory.recall("Where is the discount rate applied to a price?")],
                         ["How is the discount applied?"])
        self.assertEqual(memory.recall("Which tests cover logging?"), [])
        self.assertEqual(memory.recall("How is the discount applied?", skip=["how is the discount applied?"]), [])
        (self.root / "pricing.py").write_text(PRICING.replace("0.1", "0.2"))
        self.assertEqual(memory.recall("Where is the discount rate applied to a price?"), [])
        # The stale note is deleted for good; the note about the unchanged file stays.
        self.assertEqual([note["question"] for note in self.memory().notes], ["What does restock return?"])

    def test_a_repeated_question_replaces_its_note_and_the_oldest_notes_go_first(self):
        memory = self.memory()
        for number in range(NOTES + 2):
            memory.add(f"Question {number}?", f"Answer {number} (stock.py:1).")
        memory.add("question 5?", "A newer answer (stock.py:2).")
        self.assertEqual(len(memory.notes), NOTES)
        self.assertEqual(memory.notes[0]["question"], "Question 2?")
        self.assertEqual([note["answer"] for note in memory.notes if note["question"].lower() == "question 5?"],
                         ["A newer answer (stock.py:2)."])

    def test_damaged_or_foreign_files_start_empty(self):
        memory = self.memory()
        memory.add("What does restock return?", "restock returns its item (stock.py:1-2).")
        memory.path.write_text("{not json")
        self.assertEqual(self.memory().notes, [])
        memory.path.write_text(json.dumps({"repository": "/somewhere/else", "notes": memory.notes}))
        self.assertEqual(self.memory().notes, [])

    def test_a_session_saves_checked_answers_and_recalls_them_next_time(self):
        point = {"path": "pricing.py", "symbol": "apply_discount", "text": "Multiplies the price by 1 - RATE."}
        model = ScriptedModel([finish("The discount is applied in apply_discount.", [point])])
        session = Session(model, self.workspace, memory=self.memory(), output=lambda _line: None)
        self.assertEqual(session.ask("How is the discount applied in pricing.py?"), 0)
        self.assertTrue(session.saved)

        # A new session, as after a restart: the note is shown and its cited lines may be cited again.
        model = ScriptedModel([finish("As noted before, apply_discount applies it.", [point])])
        lines = []
        session = Session(model, self.workspace, memory=self.memory(), output=lines.append)
        self.assertEqual(session.ask("Where is the discount applied?"), 0)
        task = model.prompts[0][1]["content"]
        self.assertIn("Notes saved in earlier sessions", task)
        self.assertIn("pricing.py:4-5", task)
        self.assertIn("Current task: Where is the discount applied?", task)
        self.assertTrue(any("Recalled 1 saved answer" in line for line in lines))

    def test_a_single_run_uses_memory_and_recalled_lines_count_as_evidence(self):
        from offline_agent.agent import run
        memory = self.memory()
        memory.add("How is the discount applied?", "apply_discount multiplies by 1 - RATE (pricing.py:4-5). See also gone.py:1.")
        notes = memory.recall("Where is the discount applied?")
        self.assertEqual(memory.evidence(notes), [{"path": "pricing.py", "line": 4, "end_line": 5,
                                                   "text": "def apply_discount(price):"}])
        self.assertEqual(memory.text(notes), "Q: How is the discount applied?\n"
                                             "A: apply_discount multiplies by 1 - RATE (pricing.py:4-5). See also gone.py:1.")
        # The model cites only what the note cited; without the recalled evidence that would be flagged as unverified.
        model = ScriptedModel([{"action": "finish", "path": "", "content": "The price list is discounted in pricing.py:4-5.",
                                "message": "Done"}])
        self.assertEqual(run(model, self.workspace, "Summarize where the discount gets applied",
                             memory=memory, output=lambda _line: None), 0)
        self.assertIn("Notes saved in earlier sessions", model.prompts[0][1]["content"])
        self.assertEqual(len(self.memory().notes), 2)

    def test_unverified_answers_are_not_saved_and_sessions_without_memory_save_nothing(self):
        model = ScriptedModel([finish("The discount is in pricing.py:40-50.")])
        session = Session(model, self.workspace, memory=self.memory(), output=lambda _line: None)
        self.assertEqual(session.ask("How is the discount applied in pricing.py?"), 2)
        self.assertFalse(session.saved)
        point = {"path": "pricing.py", "symbol": "apply_discount", "text": "Multiplies the price by 1 - RATE."}
        session = Session(ScriptedModel([finish("In apply_discount.", [point])]), self.workspace, output=lambda _line: None)
        self.assertEqual(session.ask("How is the discount applied in pricing.py?"), 0)
        self.assertFalse(session.saved)
        self.assertFalse(self.folder.exists())

    def test_the_command_remembers_by_default_and_no_remember_turns_it_off(self):
        from unittest.mock import patch
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent, \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            main(["--workspace", str(self.root), "--memory-dir", str(self.folder), "Explain"])
            self.assertEqual(agent.call_args.kwargs["memory"].path.parent, self.folder)
            self.assertIn("--no-remember", printed.getvalue())
            main(["--workspace", str(self.root), "--no-remember", "Explain"])
            self.assertIsNone(agent.call_args.kwargs["memory"])
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.gui.serve", return_value=0) as serve, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["--gui", "--no-browser"])
            self.assertTrue(serve.call_args.kwargs["remember"])
            main(["--gui", "--no-browser", "--no-remember"])
            self.assertFalse(serve.call_args.kwargs["remember"])
        self.assertFalse(self.folder.exists())  # Nothing is written until an answer is saved.

    def test_forget_deletes_the_notes_from_the_command_line(self):
        self.memory().add("What does restock return?", "restock returns its item (stock.py:1-2).")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--forget", "--workspace", str(self.root), "--memory-dir", str(self.folder)])
        self.assertEqual(code, 0)
        self.assertIn("Forgot 1 saved answer", output.getvalue())
        self.assertEqual(list(self.folder.iterdir()), [])
        self.assertEqual(self.memory().notes, [])


if __name__ == "__main__":
    unittest.main()
