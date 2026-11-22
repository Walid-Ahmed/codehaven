"""Running tests from the graph: approval is needed, results reach the model, and nothing else can be run."""
import tempfile
import unittest
from unittest.mock import patch

from offline_agent.agent import run
from offline_agent.workspace import Workspace


class GraphRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)

    def model(self, path="", content=""):
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return {"action": "run_tests" if len(calls) == 1 else "finish",
                        "path": path if len(calls) == 1 else "", "content": content,
                        "message": "Request tests"}

        return Model(), calls

    @patch("offline_agent.agent.run_tests")
    def test_no_approval_never_executes(self, runner):
        for consent in (None, lambda: False):
            model, calls = self.model()
            run(model, self.ws, "Test", confirm_tests=consent, output=lambda _: None)
            self.assertIn("NOT run", calls[1][-1]["content"])
        runner.assert_not_called()

    @patch("offline_agent.agent.run_tests", return_value="Exit code: 1\nFAILED")
    def test_approved_result_reaches_model_without_apply(self, runner):
        model, calls = self.model()
        run(model, self.ws, "Test", confirm_tests=lambda: True, output=lambda _: None)
        runner.assert_called_once_with(self.ws)
        self.assertIn("Exit code: 1\nFAILED", calls[1][-1]["content"])

    @patch("offline_agent.agent.run_tests")
    def test_custom_command_rejected(self, runner):
        model, calls = self.model(content="rm -rf anything")
        run(model, self.ws, "Test", confirm_tests=lambda: True, output=lambda _: None)
        runner.assert_not_called()
        self.assertIn("no path, content, or custom command", calls[1][-1]["content"])

    @patch("offline_agent.agent.run_tests", return_value="Isolation: Docker; runner: pytest\nExit code: 0\n1 passed")
    def test_configured_pytest_is_used_without_model_command_arguments(self, runner):
        model, calls = self.model()
        options = {"runner": "pytest", "image": "project-tests:local", "docker_socket": None}
        run(model, self.ws, "Run pytest", confirm_tests=lambda: True,
            test_options=options, output=lambda _: None)
        runner.assert_called_once_with(self.ws, **options)
        self.assertIn("runner: pytest", calls[1][-1]["content"])
