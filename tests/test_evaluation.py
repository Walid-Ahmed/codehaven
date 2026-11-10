"""The local eval harness scores scripted agents correctly and its sample repo keeps its planted facts."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.evaluation import EVALS, evaluate_task, load_tasks, main, score

TASKS = {task["id"]: task for task in load_tasks(EVALS / "tasks.json")}
REPO = EVALS / "sample_repo"


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies = iter(replies)
        self.last_metrics = {}

    def check_model(self):
        pass

    def other_loaded_models(self):
        return []

    def chat(self, messages, schema):
        self.last_metrics = {"prompt_tokens": 100, "generated_tokens": 10, "total_seconds": 0.5}
        return next(self.replies)


class EvaluationTests(unittest.TestCase):
    def test_task_file_is_valid_and_sample_repo_has_planted_facts(self):
        self.assertGreaterEqual(len(TASKS), 10)
        store = (REPO / "inventory/store.py").read_text()
        self.assertIn("item.quantity < threshold", store)  # The bug low-stock-bug asks the agent to explain.
        self.assertIn("raise KeyError", store)
        catalog = (REPO / "inventory/catalog.py").read_text()
        self.assertGreater(len(catalog.encode()), 12_000)  # Large enough that reading it needs paging.
        self.assertTrue(catalog.rstrip().endswith('("SKU-0600", "Titanium Spork", 42.0),\n]'))
        for task in TASKS.values():
            with self.subTest(task=task["id"]):
                self.assertTrue(score(task, ""), "each task must fail before the agent acts")

    def test_second_task_set_is_valid_and_fails_before_the_agent_acts(self):
        for task in load_tasks(EVALS / "bookings_tasks.json"):
            with self.subTest(task=task["id"]):
                self.assertTrue(score(task, ""), "each task must fail before the agent acts")

    def test_correct_answer_passes_and_records_metrics(self):
        model = ScriptedModel([action("read", "inventory/store.py"),
                               action("finish", content="It raises KeyError. [inventory/store.py:18-20]")])
        result = evaluate_task(TASKS["remove-missing"], model, REPO)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual((result["exit_code"], result["steps"]), (0, 2))
        self.assertEqual((result["prompt_tokens"], result["generated_tokens"]), (200, 20))

    def test_follow_ups_run_in_one_session_and_are_scored_per_turn(self):
        task = TASKS["follow-discount"]
        prompts = []

        class Recording(ScriptedModel):
            def chat(self, messages, schema):
                prompts.append(messages[1]["content"])
                return super().chat(messages, schema)
        model = Recording([action("finish", content="checkout_total calls it. [inventory/pricing.py:17]"),
                           action("finish", content="It adds 0.08 tax.")])
        result = evaluate_task(task, model, REPO)
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["failures"], ["turn 2: answer lacks /config\\.py/"])
        self.assertIn("Q2: What tax rate does that function add", result["answer"])
        # The follow-up's prompt carries the first question and answer, so "that function" can be resolved.
        self.assertIn("Q: Which function calls apply_discount?\nA: checkout_total calls it.", prompts[1])

    def test_any_changed_file_fails_a_task(self):
        from offline_agent.evaluation import changed_files
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary, "repo")
            shutil.copytree(REPO, copy)
            self.assertEqual(changed_files(REPO, copy), [])
            (copy / "config.py").write_text("changed = True\n")
            (copy / "extra.py").write_text("x = 1\n")
            self.assertEqual(changed_files(REPO, copy), ["config.py", "extra.py"])

    def test_task_files_cannot_ask_for_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "tasks.json")
            path.write_text(json.dumps([{"id": "x", "task": "Fix it", "checks": {"files": {"a.py": ["y"]}}}]))
            with self.assertRaisesRegex(ValueError, "never changes files"):
                load_tasks(path)

    def test_reloads_between_steps_are_counted(self):
        class Reloading(ScriptedModel):
            def chat(self, messages, schema):
                reply = super().chat(messages, schema)
                self.last_metrics["load_seconds"] = 50
                return reply
        model = Reloading([action("read", "inventory/store.py"), action("finish", content="KeyError [inventory/store.py:18-20]")])
        self.assertEqual(evaluate_task(TASKS["remove-missing"], model, REPO)["reloads"], 1)  # The first load is expected.

    def test_wrong_answer_and_errors_fail(self):
        result = evaluate_task(TASKS["tax-rate"], ScriptedModel([action("finish", content="It is 0.08.")]), REPO)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], ["answer lacks /config\\.py/"])
        result = evaluate_task(TASKS["tax-rate"], ScriptedModel([{"action": "x"}, {"action": "x"}]), REPO)
        self.assertEqual(result["exit_code"], 1)
        self.assertTrue(result["failures"][0].startswith("error: Model returned an invalid action twice"))

    def test_cli_writes_report_and_rejects_unknown_ids(self):
        model = ScriptedModel([action("finish", content="It raises KeyError.")])
        with tempfile.TemporaryDirectory() as temporary, patch("offline_agent.evaluation.Ollama", return_value=model), \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            output = Path(temporary, "results.json")
            self.assertEqual(main(["--only", "remove-missing", "--output", str(output)]), 0)
            report = json.loads(output.read_text())
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["results"][0]["id"], "remove-missing")
        self.assertTrue(report["results"][0]["log"])  # Kept so a failed task can be diagnosed afterward.
        self.assertIn("Passed 1/1", printed.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--only", "no-such-task"])
