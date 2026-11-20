"""Explaining Git changes: read with git only, hidden files left out, and nothing written, not even inside .git."""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from offline_agent import changes
from offline_agent.agent import Session
from offline_agent.memory import Memory
from offline_agent.workspace import Workspace

STORE = ("class Store:\n    def __init__(self):\n        self.items = {}\n\n"
         "    def low_stock(self, limit):\n        return [n for n, c in self.items.items() if c < limit]\n")


def finish(content, points=()):
    return {"action": "finish", "path": "", "content": content, "message": "Done", "points": list(points)}


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts, self.schemas = list(replies), [], []

    def chat(self, messages, schema, on_text=None):
        self.prompts.append(messages)
        self.schemas.append(schema)
        return self.replies.pop(0)


def snapshot(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class ChangesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        self.root.mkdir()
        self.git("init", "-q")
        (self.root / "store.py").write_text(STORE)
        (self.root / "notes.md").write_text("# Notes\n")
        (self.root / ".env").write_text("KEY=1\n")
        self.git("add", "-A", "-f")
        self.commit("Add the store")
        self.workspace = Workspace(self.root)

    def git(self, *arguments):
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *arguments], check=True, capture_output=True)

    def commit(self, message):
        self.git("commit", "-q", "-m", message)

    def test_tasks_about_changes_are_recognised_and_others_are_not(self):
        for task, expected in [("What changed in my last commit?", ("commit", "HEAD")),
                               ("Explain my uncommitted changes", ("uncommitted", None)),
                               ("What have I changed since the last commit?", ("uncommitted", None)),
                               ("Explain the staged changes", ("staged", None)),
                               ("Explain commit 8ec6e4f", ("commit", "8ec6e4f")),
                               ("How does Store.commit work?", None),
                               ("Where are uncommitted transactions rolled back?", None),
                               ("Explain store.py", None)]:
            with self.subTest(task=task):
                self.assertEqual(changes.requested(task), expected)

    def test_uncommitted_changes_are_read_without_writing_anything(self):
        (self.root / "store.py").write_text(STORE.replace("c < limit", "c <= limit"))
        (self.root / ".env").write_text("KEY=2\n")
        (self.root / "new.py").write_text("X = 1\n")
        before = snapshot(self.root)
        report = changes.describe(self.workspace, ("uncommitted", None), 6000)
        self.assertEqual(snapshot(self.root), before)  # Not even git's index file was refreshed.
        self.assertIn("-        return [n for n, c in self.items.items() if c < limit]", report["text"])
        self.assertIn("+        return [n for n, c in self.items.items() if c <= limit]", report["text"])
        self.assertIn("new.py", report["text"])
        self.assertNotIn("KEY", report["text"])
        self.assertNotIn(".env:", report["text"])
        self.assertIn("1 hidden or excluded file", report["text"])
        self.assertEqual(report["paths"], ["store.py"])
        row = report["evidence"][0]
        self.assertTrue(row["path"] == "store.py" and row["line"] <= 6 <= row["end_line"])

    def test_last_commit_and_a_commit_by_name(self):
        (self.root / "store.py").write_text(STORE + "\n    def count(self):\n        return len(self.items)\n")
        (self.root / "notes.md").unlink()
        self.git("add", "-A")
        self.commit("Count items\n\nAlso drop the notes.")
        report = changes.describe(self.workspace, ("commit", "HEAD"), 6000)
        self.assertIn("Count items", report["text"])
        self.assertIn("the last commit", report["text"])
        self.assertIn("+    def count(self):", report["text"])
        self.assertIn("notes.md: deleted", report["text"])
        self.assertEqual(report["paths"], ["store.py"])  # A deleted file cannot be cited.
        first = subprocess.run(["git", "-C", str(self.root), "rev-list", "--max-parents=0", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
        root_commit = changes.describe(self.workspace, ("commit", first[:10]), 6000)
        self.assertIn("store.py: added", root_commit["text"])
        with self.assertRaisesRegex(changes.NoChanges, "no commit named"):
            changes.describe(self.workspace, ("commit", "deadbeef1"), 6000)

    def test_nothing_to_explain_is_said_plainly(self):
        with self.assertRaisesRegex(changes.NoChanges, "no uncommitted changes"):
            changes.describe(self.workspace, ("uncommitted", None), 6000)
        plain = Path(self.temp.name, "plain")
        plain.mkdir()
        (plain / "a.py").write_text("A = 1\n")
        with self.assertRaisesRegex(changes.NoChanges, "not a Git repository"):
            changes.describe(Workspace(plain), ("commit", "HEAD"), 6000)

    def test_a_large_diff_is_cut_to_the_limit_and_says_so(self):
        (self.root / "big.py").write_text("".join(f"VALUE_{n} = {n}\n" for n in range(900)))
        report = changes.describe(self.workspace, ("uncommitted", None), 3000)
        self.assertLessEqual(len(report["text"].encode()), 3300)
        self.assertNotIn("Not fully shown", report["text"])  # big.py is untracked, so it is only named.
        self.git("add", "big.py")
        report = changes.describe(self.workspace, ("uncommitted", None), 3000)
        self.assertIn("Not fully shown: big.py", report["text"])

    def test_answer_cites_changed_lines_and_is_not_remembered(self):
        (self.root / "store.py").write_text(STORE.replace("c < limit", "c <= limit"))
        model = ScriptedModel([finish("low_stock now includes items at the limit.",
                                      [{"path": "store.py", "symbol": "Store.low_stock", "text": "< became <=."}])])
        memory = Memory(self.workspace, Path(self.temp.name, "memory"))
        output = []
        session = Session(model, self.workspace, memory=memory, output=output.append)
        self.assertEqual(session.ask("Explain my uncommitted changes"), 0)
        answer = session.last_state["final_text"]
        self.assertRegex(answer, r"store\.py:\d+(-\d+)? `Store\.low_stock`")
        self.assertEqual(session.last_state["issues"], [])
        self.assertIn("c <= limit", model.prompts[0][-1]["content"])
        self.assertEqual(model.schemas[0]["properties"]["points"]["items"]["properties"]["path"]["enum"], ["store.py"])
        self.assertFalse(session.saved)
        self.assertEqual(memory.notes, [])

    def test_no_changes_is_answered_without_the_model(self):
        output = []
        session = Session(ScriptedModel([]), self.workspace, output=output.append)
        self.assertEqual(session.ask("What are my uncommitted changes?"), 0)
        self.assertIn("no uncommitted changes", session.last_state["final_text"])


if __name__ == "__main__":
    unittest.main()
