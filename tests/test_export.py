"""Answers saved as Markdown: only when asked, to a new file, with citations and failed checks kept."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.cli import interactive, main
from offline_agent.export import markdown, write


class Session:
    def __init__(self, root):
        self.workspace = type("W", (), {"root": root})()
        self.model = type("M", (), {"model": "qwen2.5-coder:7b"})()
        self.last_state = {}

    def ask(self, task):
        self.last_state = {"final_text": f"{task} is at pkg/a.py:3.", "issues": ["Citation not supported: pkg/a.py:3-3."]}
        return 2


class ExportTests(unittest.TestCase):
    def test_markdown_keeps_citations_and_checks(self):
        text = markdown([("Where is x?", "At pkg/a.py:3.", ["Citation not supported"])], "/repo", "m")
        self.assertIn("## Where is x?", text)
        self.assertIn("At pkg/a.py:3.", text)
        self.assertIn("> **Unverified**\n> - Citation not supported", text)

    def test_an_existing_file_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder, "notes.md")
            target.write_text("mine")
            with self.assertRaises(FileExistsError):
                write(target, "new")
            self.assertEqual(target.read_text(), "mine")
            self.assertTrue(write(folder, "new").name.startswith("codehaven-"))  # A folder gets a new file inside it.

    def test_interactive_export_saves_only_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder, "chat.md")
            inputs = iter(["/export " + str(target), "Where is x?", "/export " + str(target), "quit"])
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                interactive(Session(folder), prompt=lambda _: next(inputs))
            self.assertIn("Nothing to export yet", out.getvalue())
            self.assertIn("Where is x? is at pkg/a.py:3.", target.read_text())
            self.assertEqual([path.name for path in Path(folder).iterdir()], ["chat.md"])

    def test_the_command_line_export(self):
        with tempfile.TemporaryDirectory() as folder, patch("offline_agent.cli.Ollama") as ollama, \
                patch("offline_agent.cli.Session", side_effect=lambda *a, **k: Session(folder)), \
                contextlib.redirect_stdout(io.StringIO()):
            ollama.return_value.model = "m"
            ollama.return_value.other_loaded_models.return_value = []
            target = Path(folder, "answer.md")
            main(["--workspace", folder, "--no-remember", "--no-semantic", "--export", str(target), "Where is x?"])
            self.assertIn("## Where is x?", target.read_text())


if __name__ == "__main__":
    unittest.main()
