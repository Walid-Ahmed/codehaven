"""The opt-in run_python action: off unless asked for, approved code only, and the same sandbox as the tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import run_state
from offline_agent.cli import main
from offline_agent.prompts import PYTHON_GUIDE
from offline_agent.test_runner import SNIPPET_BYTES, container_command, run_snippet, snapshot
from offline_agent.workspace import Workspace

IMAGE_ID = "sha256:" + "a" * 64
CODE = "from pricing import total\nprint(total(3))\n"


def action(kind, content="", path=""):
    return {"action": kind, "path": path, "content": content, "message": "Working"}


class Model:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, schema, on_text=None):
        self.prompts.append((messages, schema))
        return self.replies.pop(0)


class RunPythonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pricing.py").write_text("def total(n):\n    return n * 2\n")
        self.ws = Workspace(self.root)

    def run_task(self, replies, **options):
        model, lines = Model(replies), []
        state = run_state(model, self.ws, "Work out what total(3) returns", output=lines.append, **options)
        return model, state, lines

    def test_off_by_default_so_the_prompts_are_unchanged(self):
        model, _, lines = self.run_task([action("run_python", CODE), action("finish", "It returns 6.")])
        messages, schema = model.prompts[0]
        self.assertNotIn(PYTHON_GUIDE, messages[0]["content"])
        self.assertNotIn("run_python", schema["properties"]["action"]["enum"])
        self.assertTrue(any("unknown action 'run_python'" in line for line in lines))

    def test_a_declined_snippet_never_runs(self):
        with patch("offline_agent.agent.run_snippet") as sandbox:
            model, state, lines = self.run_task([action("run_python", CODE), action("finish", "It returns 6.")],
                                                confirm_code=lambda code: False)
        sandbox.assert_not_called()
        self.assertIn(PYTHON_GUIDE, model.prompts[0][0][0]["content"])
        self.assertIn("run_python", model.prompts[0][1]["properties"]["action"]["enum"])
        self.assertTrue(any(CODE.strip() in line for line in lines))  # The user saw the code before deciding.
        self.assertIn("NOT run", model.prompts[1][0][-1]["content"])

    def test_an_approved_snippet_runs_in_the_sandbox_and_its_output_goes_to_the_model(self):
        shown = []
        with patch("offline_agent.agent.run_snippet", return_value="Exit code: 0\n6\n") as sandbox:
            model, state, _ = self.run_task([action("run_python", CODE), action("finish", "It returns 6.")],
                                            confirm_code=lambda code: shown.append(code) or True,
                                            test_options={"image": "local:1"})
        sandbox.assert_called_once()
        self.assertEqual(sandbox.call_args.args[1], CODE)
        self.assertEqual(sandbox.call_args.kwargs["image"], "local:1")
        self.assertIn("Exit code: 0\n6", model.prompts[1][0][-1]["content"])
        self.assertEqual(shown, [CODE])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["pricing.py"])  # Nothing was written.

    def test_the_snippet_runs_in_the_locked_down_container(self):
        line = container_command(["docker"], IMAGE_ID, "name", Path("/tmp/project"), None, Path("/tmp/snippet"))
        self.assertIn("none", line[line.index("--network") + 1])
        self.assertIn("type=bind,source=/tmp/project,target=/workspace,readonly", line)
        self.assertIn("type=bind,source=/tmp/snippet,target=/snippet,readonly", line)
        self.assertEqual(line[-2:], [IMAGE_ID, "/snippet/main.py"])
        self.assertIn("--read-only", line)

    def test_snippets_need_no_tests_folder_and_are_bounded(self):
        destination = Path(self.temp.name, "copy")
        destination.mkdir()
        self.assertEqual(snapshot(self.ws, destination, None), 1)
        for code in ("", "   ", "x" * (SNIPPET_BYTES + 1)):
            with self.subTest(size=len(code)), self.assertRaises(ValueError):
                run_snippet(self.ws, code)

    def test_the_cli_offers_it_only_with_allow_python(self):
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent, \
                patch("offline_agent.cli.print"):
            main(["--workspace", str(self.root), "--no-remember", "Explain"])
            self.assertIsNone(agent.call_args.kwargs["confirm_code"])
            main(["--workspace", str(self.root), "--no-remember", "--allow-python", "Explain"])
            self.assertTrue(callable(agent.call_args.kwargs["confirm_code"]))


if __name__ == "__main__":
    unittest.main()


class RetryTests(unittest.TestCase):
    """An answer that fails its checks is sent back once, and whichever answer passes more checks is kept."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pricing.py").write_text("RATE = 0.25\n\n\ndef total(n):\n    return n * (1 + RATE)\n")
        self.ws = Workspace(self.root)

    def run_task(self, replies, retry=True):
        model, lines = Model(replies), []
        state = run_state(model, self.ws, "What rate does total in pricing.py use?", output=lines.append, retry_unverified=retry)
        return model, state, lines

    def test_a_corrected_answer_that_passes_replaces_the_first(self):
        model, state, lines = self.run_task([action("finish", "It adds a tax (pricing.py:90)."),
                                             action("finish", "It adds RATE, 25% (pricing.py:1 and pricing.py:4-5).")])
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("did not pass CodeHaven's checks", model.prompts[1][0][-2]["content"])
        self.assertEqual(state["issues"], [])
        self.assertEqual(state["exit_code"], 0)
        self.assertIn("pricing.py:4-5", state["final_text"])

    def test_a_correction_that_is_no_better_keeps_the_first_answer(self):
        model, state, lines = self.run_task([action("finish", "It adds a tax (pricing.py:90)."),
                                             action("finish", "It adds a tax (pricing.py:91).")])
        self.assertIn("pricing.py:90", state["final_text"])
        self.assertEqual(state["exit_code"], 2)
        self.assertTrue(any("first answer is kept" in line for line in lines))

    def test_only_once_and_only_when_asked(self):
        model, state, _ = self.run_task([action("finish", "It adds a tax (pricing.py:90).")], retry=False)
        self.assertEqual(len(model.prompts), 1)
        self.assertEqual(state["exit_code"], 2)
