"""Saved chats: kept privately outside the repository, reopened with their memory, and off with --no-remember."""
import os
import stat
import tempfile
import unittest
from pathlib import Path

from offline_agent.chats import CHATS, Chats, new_id
from offline_agent.gui import App
from offline_agent.workspace import Workspace


def finish(content):
    return {"action": "finish", "path": "", "content": content, "message": "Done", "points": []}


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, _schema, on_text=None):
        self.prompts.append(messages)
        return self.replies.pop(0)


class ChatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        self.root.mkdir()
        (self.root / "pricing.py").write_text("RATE = 0.1\n")
        self.folder = Path(self.temp.name, "chats")

    def app(self, replies, remember=True):
        app = App(ScriptedModel(replies), remember=remember, memory_folder=Path(self.temp.name, "memory"),
                  chats_folder=self.folder)
        self.addCleanup(app.close)
        app.open(str(self.root))
        return app

    def ask(self, app, question):
        return [event for event in app.ask(question)][-1]

    def test_chats_are_saved_privately_outside_the_repository(self):
        chats = Chats(Workspace(self.root), self.folder)
        first = new_id()
        chats.record(first, "What is RATE?", "It is 0.1 (pricing.py:1).", ["an issue"])
        chats.record(first, "And why?", "No reason is given.")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["pricing.py"])
        if os.name != "nt":  # Windows keeps a user's files private through the profile folder, not file modes.
            self.assertEqual(stat.S_IMODE(chats.path.stat().st_mode), 0o600)
        again = Chats(Workspace(self.root), self.folder)
        self.assertEqual([(row["title"], row["questions"]) for row in again.listing()], [("What is RATE?", 2)])
        self.assertEqual(again.get(first)["turns"][0]["issues"], ["an issue"])
        for number in range(CHATS + 2):
            again.record(new_id(), f"Question {number}", "Answer")
        self.assertEqual(len(again.chats), CHATS)  # The oldest go first.
        self.assertNotIn(first, [chat["id"] for chat in again.chats])
        again.forget()
        self.assertFalse(again.path.exists())
        chats.path.parent.mkdir(exist_ok=True)
        chats.path.write_text("{broken")
        self.assertEqual(Chats(Workspace(self.root), self.folder).chats, [])

    def test_a_chat_is_reopened_after_a_restart_with_its_memory(self):
        app = self.app([finish("RATE is 0.1 (pricing.py:1)."), finish("It is a discount rate (pricing.py:1).")])
        done = self.ask(app, "What is RATE?")
        self.assertEqual([chat["title"] for chat in done["chats"]], ["What is RATE?"])
        self.ask(app, "What is it for?")
        self.assertEqual(app.summary()["chats"][0]["questions"], 2)
        app.reset()
        restarted = self.app([finish("Yes (pricing.py:1).")])
        listed = restarted.summary()["chats"]
        self.assertEqual(len(listed), 1)
        opened = restarted.open_chat(listed[0]["id"])
        self.assertEqual([turn["question"] for turn in opened["turns"]], ["What is RATE?", "What is it for?"])
        self.assertEqual(opened["remembered"], 2)
        self.ask(restarted, "Is it constant?")
        self.assertIn("What is it for?", restarted.session.model.prompts[0][1]["content"])  # The follow-up saw the chat.
        self.assertEqual(restarted.summary()["chats"][0]["questions"], 3)  # And it was added to the same chat.
        self.assertEqual(restarted.delete_chat(listed[0]["id"]), {"chats": []})
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            restarted.open_chat(listed[0]["id"])

    def test_new_chat_starts_a_separate_saved_chat_and_forget_deletes_them(self):
        app = self.app([finish("One (pricing.py:1)."), finish("Two (pricing.py:1).")])
        self.ask(app, "First question?")
        app.reset()
        self.ask(app, "Second question?")
        self.assertEqual([chat["title"] for chat in app.summary()["chats"]], ["Second question?", "First question?"])
        self.assertEqual(app.forget()["chats"], [])
        self.assertFalse(any(self.folder.glob("*.json")))

    def test_nothing_is_saved_with_no_remember_or_for_an_uploaded_file(self):
        app = self.app([finish("One (pricing.py:1).")], remember=False)
        done = self.ask(app, "First question?")
        self.assertEqual(done["chats"], [])
        self.assertFalse(self.folder.exists())
        app = self.app([finish("It sets X (notes.py:1).")])
        app.open_file("notes.py", b"X = 1\n")
        self.assertEqual(self.ask(app, "What does this do?")["chats"], [])
        self.assertFalse(self.folder.exists())


if __name__ == "__main__":
    unittest.main()
