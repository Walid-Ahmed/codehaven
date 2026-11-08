"""CodeHaven always runs offline: no tracing flags, clients, or env-var opt-in."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langsmith import get_tracing_context
from langsmith.utils import tracing_is_enabled

import offline_agent
from offline_agent.agent import run
from offline_agent.cli import main
from offline_agent.workspace import Workspace

TRACING_ENV = {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true",
               "LANGSMITH_API_KEY": "test-key", "LANGCHAIN_API_KEY": "test-key"}


class OfflineModel:
    """Stands in for local Ollama and records whether tracing was active during the call."""
    def __init__(self):
        self.observed = []

    def check_model(self):
        pass

    def other_loaded_models(self):
        return []

    def chat(self, messages, schema, on_text=None):
        self.observed.append((get_tracing_context()["enabled"], tracing_is_enabled()))
        return dict(action="finish", path="", content="Done", message="Done")


class OfflineTests(unittest.TestCase):
    def test_removed_tracing_flags_are_rejected(self):
        for flag in ("--langsmith-tracing", "--langsmith-project=x", "--langsmith-include-content"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as exit_:
                main([flag, "Explain"])
            self.assertEqual(exit_.exception.code, 2)

    def test_run_no_longer_accepts_a_tracing_client(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(TypeError):
            run(OfflineModel(), Workspace(root), "Explain", tracing_client=object(),  # pyright: ignore[reportCallIssue]
                output=lambda _: None)

    @patch.dict(os.environ, TRACING_ENV)
    def test_tracing_environment_cannot_enable_uploads_during_a_cli_run(self):
        model = OfflineModel()
        with tempfile.TemporaryDirectory() as root, \
             patch("offline_agent.cli.Ollama", return_value=model), \
             patch("langsmith.client.Client", side_effect=AssertionError("tracing client created")) as client, \
             contextlib.redirect_stdout(io.StringIO()):
            # 2 only means the stub answer cites no evidence; 1 would be an error.
            self.assertNotEqual(main(["--workspace", root, "Explain"]), 1)
        self.assertEqual(model.observed, [(False, False)])
        client.assert_not_called()

    def test_package_source_has_no_tracing_client_or_tracer(self):
        package = Path(offline_agent.__file__).parent
        for source in package.glob("*.py"):
            text = source.read_text()
            for banned in ("LangChainTracer", "langsmith import Client", "langsmith.Client"):
                self.assertNotIn(banned, text, f"{source.name} references {banned}")
