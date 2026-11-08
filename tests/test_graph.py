"""Behavioral coverage for the LangGraph migration."""
import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from langsmith import get_tracing_context

from offline_agent.agent import build_graph, run
from offline_agent.cli import main
from offline_agent.workspace import Workspace


def action(kind, path="", content=""):
    return dict(action=kind, path=path, content=content, message=kind)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)

    def test_wait_message_precedes_model_call_and_reporter_stops_on_error(self):
        output = []
        threads_before = set(threading.enumerate())

        class Model:
            def chat(self, messages, schema):
                self_check = output[-1]
                if "waiting for the local model" not in self_check:
                    raise AssertionError("Missing progress before model call")
                raise ValueError("Local model timed out")

        with self.assertRaisesRegex(ValueError, "Local model timed out"):
            run(Model(), self.ws, "Explain", output=output.append)
        self.assertEqual(set(threading.enumerate()), threads_before)

    def test_read_result_reaches_next_model_call(self):
        Path(self.temp.name, "code.py").write_text("value = 42\n")
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return action("read", "code.py") if len(calls) == 1 else action("finish", content="Value is 42 [code.py:1].")

        self.assertEqual(run(Model(), self.ws, "Inspect code", output=lambda _: None), 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("value = 42", calls[1][-1]["content"])
        self.assertEqual(calls[1][-2]["role"], "assistant")

    def test_thirty_model_actions_stop_without_recursion_error(self):
        class Model:
            calls = 0

            def chat(self, messages, schema):
                self.calls += 1
                return action("read", f"missing{self.calls}.py")

        model = Model()
        self.assertEqual(run(model, self.ws, "Task", max_steps=30, output=lambda _: None), 2)
        self.assertEqual(model.calls, 30)

    def test_repeated_lookup_skips_tool_and_requests_final_answer(self):
        Path(self.temp.name, "code.py").write_text("value = 42\n")
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append((messages, schema))
                if len(calls) <= 2:
                    return action("read", "code.py", f"unused {len(calls)}")
                return action("finish", content="The value is 42. [code.py:1]")

        output = []
        with patch.object(self.ws, "read", wraps=self.ws.read) as read:
            result = run(Model(), self.ws, "Explain value", max_steps=4, output=output.append)
        self.assertEqual(result, 0)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(calls[-1][1]["properties"]["action"]["enum"], ["finish"])
        self.assertTrue(any(": read code.py (read)" in line for line in output))
        self.assertIn("The value is 42. [code.py:1]", output)

    def test_model_ignoring_loop_recovery_stops_without_repeating_tools(self):
        class Model:
            calls = 0

            def chat(self, messages, schema):
                self.calls += 1
                return action("search", content="FastAPI(")

        model = Model()
        with patch.object(self.ws, "search", return_value='{"matches": []}') as search:
            result = run(model, self.ws, "Find entry points", max_steps=12, output=lambda _: None)
        self.assertEqual(result, 2)
        self.assertEqual(model.calls, 3)
        search.assert_called_once()

    def test_final_answer_retains_all_search_paths_and_is_displayed_once(self):
        for name in ("service_a/main.py", "service_b/main.py"):
            target = Path(self.temp.name, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("app = FastAPI()\n")
        captured = []
        class Model:
            def chat(self, messages, schema):
                captured.append(messages)
                if len(captured) == 1:
                    return action("search", content="FastAPI(")
                if len(captured) == 2:
                    return action("read", "service_b/main.py")
                return dict(action="finish", path="service_b/main.py", content="Both services expose apps. [service_a/main.py:1] [service_b/main.py:1]",
                            message="Both services expose apps with constructors.")
        output = []
        result = run(Model(), self.ws, "Find all entry points", max_steps=3, output=output.append)
        self.assertEqual(result, 0)
        final_context = captured[-1][-1]["content"]
        self.assertIn("service_a/main.py", final_context)
        self.assertIn("service_b/main.py", final_context)
        self.assertIn("ALL relevant discovered paths", final_context)
        self.assertEqual(sum("Both services" in line for line in output), 1)

    def test_finish_on_last_allowed_action_succeeds(self):
        class Model:
            def chat(self, messages, schema):
                return action("finish")

        self.assertEqual(run(Model(), self.ws, "Task", max_steps=1, output=lambda _: None), 0)

    def test_last_action_is_reserved_for_answer_and_cannot_write(self):
        class Model:
            def chat(self, messages, schema):
                return action("write", "new.py", "answer = 42\n")

        self.assertEqual(run(Model(), self.ws, "Task", max_steps=1, apply=True,
                             confirm=lambda _: True, output=lambda _: None), 2)
        self.assertFalse((Path(self.temp.name) / "new.py").exists())

    def test_four_step_inspection_reserves_last_call_for_answer(self):
        calls = []
        for name in ("first.py", "second.py", "third.py"):
            Path(self.temp.name, name).write_text("app = FastAPI()\n")

        class Model:
            def chat(self, messages, schema):
                calls.append((messages, schema))
                if schema["properties"]["action"]["enum"] == ["finish"]:
                    return action("finish", content="Entry points include first.py and second.py. [first.py:1] [second.py:1]")
                if len(calls) == 1:
                    return action("search", content="FastAPI(")
                return action("read", ("first.py", "second.py", "third.py")[len(calls)-2])

        output = []
        with patch.object(self.ws, "read", wraps=self.ws.read) as read:
            result = run(Model(), self.ws, "Find entry points", max_steps=4, output=output.append)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1][1]["properties"]["action"]["enum"], ["finish"])
        self.assertIn("no tool calls remain", calls[-1][0][-1]["content"])
        self.assertIn("Entry points include first.py and second.py. [first.py:1] [second.py:1]", output)

    @patch.dict("os.environ", {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"})
    def test_parent_tracing_is_disabled_for_model_execution(self):
        observed = []

        class Model:
            def chat(self, messages, schema):
                observed.append(get_tracing_context()["enabled"])
                return action("finish")

        run(Model(), self.ws, "Task", output=lambda _: None)
        self.assertEqual(observed, [False])

    def test_invalid_limit_rejected_before_model_execution(self):
        for limit in (0, 31):
            with self.assertRaises(ValueError):
                build_graph(None, self.ws, max_steps=limit)

    def test_cli_doctor_uses_fixed_model(self):
        with patch("offline_agent.cli.Ollama") as model, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--doctor"]), 0)
        model.assert_called_once_with(endpoint="http://127.0.0.1:11434", max_output_tokens=1024, prompt_budget=12000)
        model.return_value.check_model.assert_called_once_with()
