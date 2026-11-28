"""Why is this line here? The commits come from git blame; the model only explains what they say."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run_state
from offline_agent.history import requested
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace


def git(root, *arguments):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Dana", "-c", "user.email=dana@example.com",
                    "-c", "commit.gpgsign=false", *arguments], check=True, capture_output=True)


class Model:
    prompt_budget = 12_000

    def __init__(self):
        self.prompts = []

    def chat(self, messages, schema):
        self.prompts.append(messages[-1]["content"])
        return {"answer": "The second commit added the floor so totals never go below zero."}


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        (self.root / "pricing.py").write_text("def total(p):\n    return p\n")
        git(self.root, "add", "pricing.py")
        git(self.root, "commit", "-q", "-m", "Add total")
        (self.root / "pricing.py").write_text("def total(p):\n    return max(0, p)\n")
        git(self.root, "commit", "-q", "-am", "Never return a negative total", "-m", "Refunds made totals negative.")
        self.ws = Workspace(self.root)

    def test_requests_are_recognised(self):
        self.assertTrue(requested("Why is the max() call here in pricing.py:2?"))
        self.assertTrue(requested("Who last changed total?"))
        self.assertFalse(requested("Explain total"))

    def test_the_commit_that_changed_the_lines_is_cited_and_explained(self):
        model = Model()
        before = sorted(path.name for path in (self.root / ".git").rglob("*"))
        state = run_state(model, self.ws, "Why is pricing.py:2 written this way?", knowledge=KnowledgeBase(workspace=self.ws),
                          output=lambda _: None)
        text = state["final_text"]
        self.assertIn("pricing.py:2: ", text)
        self.assertIn("by Dana", text)
        self.assertIn("Never return a negative total", text)
        self.assertIn("floor", text)
        self.assertIn("Refunds made totals negative.", model.prompts[0])
        self.assertIn("max(0, p)", model.prompts[0])
        self.assertEqual(sorted(path.name for path in (self.root / ".git").rglob("*")), before)  # Git only read.

    def test_uncommitted_lines_are_said_to_be_so(self):
        (self.root / "pricing.py").write_text("def total(p):\n    return max(1, p)\n")
        state = run_state(Model(), self.ws, "Who changed pricing.py:2?", output=lambda _: None)
        self.assertIn("not committed yet", state["final_text"])


if __name__ == "__main__":
    unittest.main()
