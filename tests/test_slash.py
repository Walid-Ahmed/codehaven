"""Slash commands: short forms that are written out in full, in the terminal and on the page."""
import contextlib
import io
import unittest

from offline_agent.cli import interactive
from offline_agent.slash import expand, help_text


class SlashTests(unittest.TestCase):
    def test_commands_are_written_out(self):
        self.assertEqual(expand("/trace checkout_total"), "Trace checkout_total")
        self.assertEqual(expand("/impact Store.add"), "What breaks if I change Store.add?")
        self.assertEqual(expand("/deps"), "Which libraries does this project use?")
        self.assertEqual(expand("/diagram pay"), "Draw a Mermaid diagram of the calls of pay")
        self.assertEqual(expand("What is /trace?"), "What is /trace?")  # Not a command: left as typed.
        self.assertIsNone(expand("/help"))
        for wrong in ("/trace", "/nonsense"):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                expand(wrong)
        self.assertIn("/review <name>", help_text())

    def test_the_terminal_asks_the_written_out_request(self):
        asked = []

        class Session:
            def ask(self, task):
                asked.append(task)
        inputs = iter(["/trace pay", "/help", "/oops", "quit"])
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            interactive(Session(), prompt=lambda _: next(inputs))
        self.assertEqual(asked, ["Trace pay"])
        self.assertIn("/impact <name>", out.getvalue())
        self.assertIn("Unknown command /oops", err.getvalue())


if __name__ == "__main__":
    unittest.main()
