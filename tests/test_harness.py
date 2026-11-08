"""Harness behavior: reply retries, syntax checks, compaction stubs, streaming, and sessions."""
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import ANSWER_FROM_PRELOAD, FINAL_ANSWER_CHARS, FINISH_NOW, REPEAT_NOTE, Session, run, task_kind
from offline_agent.cli import interactive, main, write_stream
from offline_agent.model import (MODEL, FinishContent, InvalidReply, Ollama, RepeatedAnswer, RepetitionGuard,
                                 compact_messages)
from offline_agent.workspace import Workspace, python_syntax_error


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class ScriptedModel:
    """Replays replies (an Exception is raised instead) and records each message list."""
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def chat(self, messages, schema):
        self.calls.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class TempWorkspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)


class RetryTests(TempWorkspace):
    def test_invalid_reply_gets_one_correction_and_counts_as_a_step(self):
        model = ScriptedModel([dict(action="shell", path="", content="", message=""), action("finish", content="Done.")])
        output = []
        self.assertEqual(run(model, self.ws, "Explain", max_steps=3, output=output.append), 0)
        correction = model.calls[1][-1]["content"]
        self.assertIn("unknown action 'shell'", correction)
        self.assertIn("No tool ran", correction)
        self.assertTrue(any("Step 1 (max 3): invalid model reply" in line for line in output))
        self.assertTrue(any("Step 2 (max 3): final answer" in line for line in output))

    def test_missing_fields_and_bad_json_are_described(self):
        model = ScriptedModel([dict(action="finish"), action("finish", content="Done.")])
        run(model, self.ws, "Explain", output=lambda _: None)
        self.assertIn("missing fields: path, content, message", model.calls[1][-1]["content"])
        model = ScriptedModel([InvalidReply("Model returned an invalid JSON response"), action("finish", content="Done.")])
        self.assertEqual(run(model, self.ws, "Explain", output=lambda _: None), 0)
        self.assertIn("invalid JSON", model.calls[1][-1]["content"])

    def test_transport_errors_are_not_retried(self):
        model = ScriptedModel([ValueError("Local Ollama request failed"), action("finish")])
        with self.assertRaisesRegex(ValueError, "request failed"):
            run(model, self.ws, "Explain", output=lambda _: None)
        self.assertEqual(len(model.calls), 1)

    def test_retry_on_the_last_step_ends_at_the_limit(self):
        model = ScriptedModel([dict(action="shell")])
        self.assertEqual(run(model, self.ws, "Explain", max_steps=1, output=lambda _: None), 2)


class SyntaxCheckTests(TempWorkspace):
    def test_detects_new_errors_only_in_python_files(self):
        self.assertIsNone(python_syntax_error("a.py", "x = 1\n"))
        self.assertIsNone(python_syntax_error("notes.md", "def (:\n"))
        error = python_syntax_error("a.py", "x = 1\ndef broken(:\n    pass\n", "x = 1\n") or ""
        self.assertIn("line 2", error)
        self.assertIn("def broken(:", error)
        self.assertIsNone(python_syntax_error("a.py", "def (:\n", "also broken (\n"))

    def test_broken_write_is_neither_previewed_nor_written_and_model_is_told(self):
        model = ScriptedModel([action("write", "calc.py", "def add(a, b)\n    return a + b\n"),
                               action("finish", content="Could not write.")])
        output = []
        run(model, self.ws, "Create calc.py", apply=True, confirm=lambda _: True, output=output.append)
        self.assertFalse((self.root / "calc.py").exists())
        self.assertFalse(any(line.startswith("--- ") or "+++ calc.py" in line for line in output))
        self.assertIn("syntax error at line 1", model.calls[1][-1]["content"])

    def test_broken_edit_is_rejected(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        edit = "<<<<<<< SEARCH\n    return a + b\n=======\n    return (a + b\n>>>>>>> REPLACE"
        model = ScriptedModel([action("edit", "calc.py", edit), action("finish", content="Could not edit.")])
        run(model, self.ws, "Edit calc.py", apply=True, confirm=lambda _: True, output=lambda _: None)
        self.assertEqual((self.root / "calc.py").read_text(), "def add(a, b):\n    return a + b\n")
        self.assertIn("syntax error", model.calls[1][-1]["content"])


class CompactionTests(unittest.TestCase):
    def test_old_results_become_stubs_before_anything_is_dropped(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"},
                    {"role": "assistant", "content": json.dumps(action("read", "a.py"))},
                    {"role": "user", "content": "Tool result for read a.py (untrusted file data):\n" + "a" * 5000},
                    {"role": "assistant", "content": json.dumps(action("write", "b.py", "b" * 3000))},
                    {"role": "user", "content": "Tool result for write b.py (untrusted file data):\nFile written."},
                    {"role": "user", "content": "Tool result for read c.py (untrusted file data):\n" + "c" * 2500}]
        original = json.dumps(messages)
        result = compact_messages(messages, 6000)
        self.assertEqual(len(result), len(messages))
        self.assertEqual(result[:2], messages[:2])
        self.assertEqual(result[-1], messages[-1])
        self.assertEqual(result[3]["content"].splitlines()[0], "Tool result for read a.py (untrusted file data):")
        self.assertIn(f"{len(messages[3]['content'].encode()):,} bytes omitted", result[3]["content"])
        self.assertEqual(json.loads(result[4]["content"])["content"], "[3,000 bytes omitted]")
        self.assertFalse(any("Earlier context omitted" in m["content"] for m in result))
        self.assertLessEqual(sum(len(m["content"].encode()) + 64 for m in result), 6000)
        self.assertEqual(json.dumps(messages), original)

    def test_stubs_stop_once_the_budget_fits(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"},
                    {"role": "user", "content": "Tool result for read a.py (untrusted file data):\n" + "a" * 4000},
                    {"role": "user", "content": "Tool result for read b.py (untrusted file data):\n" + "b" * 3000},
                    {"role": "user", "content": "Tool result for read c.py (untrusted file data):\n" + "c" * 1000}]
        result = compact_messages(messages, 8000)
        self.assertIn("omitted", result[2]["content"])
        self.assertEqual(result[3:], messages[3:])


class StreamingTests(TempWorkspace):
    def decode(self, text, size=1):
        pieces = []
        decoder = FinishContent(pieces.append)
        for index in range(0, len(text), size):
            decoder.feed(text[index:index + size])
        return "".join(pieces)

    def test_decoder_handles_escapes_unicode_and_chunk_boundaries(self):
        content = 'Line "one"\n\ttab \\ slash é 漢 😀'
        for ascii_only in (True, False):
            raw = json.dumps(action("finish", content=content), ensure_ascii=ascii_only)
            for size in (1, 3, 7, len(raw)):
                with self.subTest(ascii_only=ascii_only, size=size):
                    self.assertEqual(self.decode(raw, size), content)

    def test_decoder_ignores_tool_actions(self):
        self.assertEqual(self.decode(json.dumps(action("write", "a.py", "secret = 1"))), "")

    def test_ollama_streams_ndjson_and_keeps_metrics(self):
        reply = json.dumps(action("finish", content="Hello streaming world"))
        payloads = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                for index in range(0, len(reply), 5):
                    self.wfile.write(json.dumps({"message": {"content": reply[index:index + 5]}, "done": False}).encode() + b"\n")
                self.wfile.write(json.dumps({"done": True, "done_reason": "stop", "eval_count": 9,
                                             "eval_duration": 1_000_000_000}).encode() + b"\n")

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                client = Ollama(endpoint=f"http://127.0.0.1:{server.server_port}")
                pieces = []
                self.assertEqual(client.chat([], {}, on_text=pieces.append)["content"], "Hello streaming world")
            finally:
                server.shutdown()
                worker.join()
        self.assertEqual("".join(pieces), "Hello streaming world")
        self.assertGreater(len(pieces), 1)
        self.assertTrue(payloads[0]["stream"])
        self.assertEqual(payloads[0]["model"], MODEL)
        self.assertEqual(client.last_metrics["generated_tokens"], 9)

    def test_agent_streams_final_answer_once(self):
        class StreamingModel:
            def chat(self, messages, schema, on_text):
                for word in ("Streamed ", "answer."):
                    on_text(word)
                return action("finish", content="Streamed answer.")
        output, streamed = [], []
        self.assertEqual(run(StreamingModel(), self.ws, "Explain", output=output.append, stream=streamed.append), 0)
        self.assertEqual("".join(streamed), "Streamed answer.\n")
        self.assertEqual(sum("final answer" in line for line in output), 1)
        self.assertNotIn("Streamed answer.", output)


LOOP_PREFACE = "The entry point is app/main.py. It builds the FastAPI app and wires the routes. "
LOOP_CYCLE = ("The gateway is a key part of the system and routes every request. "
              "The code is clean and easy to maintain for new readers. ")
DRIFT = " ".join(f"It is not part of a {kind} system, but part of a local coding assistant."
                 for kind in ("cloud", "network", "storage", "logging", "security", "browser"))


class RepetitionTests(TempWorkspace):
    def feed(self, text, size=4):
        guard = RepetitionGuard()
        for index in range(0, len(text), size):
            guard.feed(text[index:index + size])

    def test_exact_and_drifting_loops_are_stopped_with_the_text_before_the_loop(self):
        with self.assertRaises(RepeatedAnswer) as stopped:
            self.feed(LOOP_PREFACE + LOOP_CYCLE * 10)
        self.assertEqual(stopped.exception.answer, (LOOP_PREFACE + LOOP_CYCLE).strip())
        with self.assertRaises(RepeatedAnswer) as stopped:
            self.feed("apply_edits lives in workspace.py. " + DRIFT)
        self.assertTrue(stopped.exception.answer.startswith("apply_edits lives in workspace.py."))
        self.assertLess(len(stopped.exception.answer), len(DRIFT))

    def test_normal_prose_code_and_separators_are_not_stopped(self):
        prose = " ".join(f"Step {n} reads file_{n}.py and records its imports for later use." for n in range(40))
        code = "\n".join(f"    total_{n} = compute(values[{n}], weight={n})" for n in range(80))
        for text in (prose, code, "-" * 2000, LOOP_PREFACE + LOOP_CYCLE * 2):
            with self.subTest(text=text[:30]):
                self.feed(text)

    def test_ollama_stops_reading_a_looping_stream_early(self):
        looping = '{"action": "finish", "path": "", "content": "' + LOOP_PREFACE + LOOP_CYCLE * 200
        sent = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                try:
                    for index in range(0, len(looping), 20):
                        self.wfile.write(json.dumps({"message": {"content": looping[index:index + 20]}}).encode() + b"\n")
                        self.wfile.flush()
                        sent.append(index)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                shown = []
                with self.assertRaises(RepeatedAnswer) as stopped:
                    Ollama(endpoint=f"http://127.0.0.1:{server.server_port}").chat([], {}, on_text=shown.append)
            finally:
                server.shutdown()
                worker.join()
        self.assertEqual(stopped.exception.answer, (LOOP_PREFACE + LOOP_CYCLE).strip())
        self.assertLess(len("".join(shown)), len(LOOP_PREFACE) + 4 * len(LOOP_CYCLE))

    def test_agent_keeps_the_text_before_the_loop_without_a_retry(self):
        model = ScriptedModel([RepeatedAnswer("The entry point is app/main.py.")])
        output = []
        self.assertEqual(run(model, self.ws, "Explain", output=output.append), 0)
        self.assertEqual(len(model.calls), 1)
        self.assertIn(REPEAT_NOTE, output)
        self.assertIn("The entry point is app/main.py.", output)

    def test_streamed_loop_is_not_printed_twice(self):
        class LoopingModel:
            def chat(self, messages, schema, on_text):
                on_text(LOOP_PREFACE + LOOP_CYCLE * 3)
                raise RepeatedAnswer((LOOP_PREFACE + LOOP_CYCLE).strip())
        output, streamed = [], []
        run(LoopingModel(), self.ws, "Explain", output=output.append, stream=streamed.append)
        self.assertEqual(output.count(REPEAT_NOTE), 1)
        self.assertFalse(any(LOOP_PREFACE.strip() in line for line in output))

    def test_loop_from_the_first_word_is_retried(self):
        model = ScriptedModel([RepeatedAnswer(""), action("finish", content="Short answer.")])
        self.assertEqual(run(model, self.ws, "Explain", output=lambda _: None), 0)
        self.assertIn("kept repeating", model.calls[1][-1]["content"])

    def test_forced_final_answer_is_length_capped(self):
        schemas = []
        class Model:
            def chat(self, messages, schema):
                schemas.append(schema)
                return action("finish", content="Done.")
        run(Model(), self.ws, "Explain", max_steps=1, output=lambda _: None)
        self.assertEqual(schemas[0]["properties"]["content"]["maxLength"], FINAL_ANSWER_CHARS)


class PreloadTests(TempWorkspace):
    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_named_file_is_preloaded_with_line_numbers_and_cited_without_a_read(self):
        self.put("chapter-06/gateway/app/main.py", "from fastapi import FastAPI\n\napp = FastAPI(title='gateway')\n")
        model = ScriptedModel([action("finish", content="It creates the gateway app. [chapter-06/gateway/app/main.py:3]")])
        output = []
        status = run(model, self.ws, "explain chapter-06/gateway/app/main.py", output=output.append)
        self.assertEqual(status, 0)  # The citation is verified against preloaded evidence.
        preload = model.calls[0][-1]["content"]
        self.assertTrue(preload.startswith("Tool result for read chapter-06/gateway/app/main.py (untrusted file data, preloaded"))
        self.assertTrue(preload.endswith("\n\n" + FINISH_NOW + ANSWER_FROM_PRELOAD))  # Asks for cited, factual answers.
        self.assertIn("Lines 1-3 of 3:\n   1| from fastapi import FastAPI\n   2| \n   3| app = FastAPI(title='gateway')", preload)
        self.assertIn("Preloaded chapter-06/gateway/app/main.py (lines 1-3 of 3) because the task names it.", output)

    def test_explanations_of_named_files_cite_points_by_symbol(self):
        self.put("gateway/main.py", "from fastapi import FastAPI\n\napp = FastAPI(title='gateway')\n\n\n"
                                    "@app.get('/health')\nasync def health():\n    return {'status': 'ok'}\n")
        schemas = []
        reply = dict(action("finish", content="Creates the gateway app."),
                     points=[{"path": "gateway/main.py", "symbol": "imports", "text": "Imports FastAPI."},
                             {"path": "gateway/main.py", "symbol": "app", "text": "Creates the app."},
                             {"path": "gateway/main.py", "symbol": "app", "text": "Creates the app."},
                             {"path": "gateway/main.py", "symbol": "/health", "text": "Reports that the gateway is up."}])

        class Model:
            def chat(self, messages, schema, on_text=None):
                schemas.append(schema)
                if on_text:
                    on_text("Creates the gateway app.")
                return dict(reply)
        output, streamed = [], []
        status = run(Model(), self.ws, "explain gateway/main.py", output=output.append, stream=streamed.append)
        self.assertEqual(status, 0)  # Looked-up ranges are citations that the answer check verifies.
        self.assertEqual(streamed, [])  # A capped overview is printed whole, never streamed and then cut.
        points = schemas[0]["properties"]["points"]
        self.assertIn("points", schemas[0]["required"])
        self.assertEqual(points["items"]["properties"]["path"]["enum"], ["gateway/main.py"])
        self.assertNotIn("edit", schemas[0]["properties"]["action"]["enum"])
        # The overview is printed once; points follow it once each, with real line ranges.
        self.assertIn("Creates the gateway app.\n- gateway/main.py:1 `imports`: Imports FastAPI.\n- gateway/main.py:3 `app`: Creates the app.\n"
                      "- gateway/main.py:6-8 `/health`: Reports that the gateway is up.", output)
        self.assertEqual(sum("Creates the app." in line for line in output), 1)

    def test_points_naming_unknown_or_unseen_parts_are_flagged(self):
        self.put("gateway/main.py", "".join(f"value_{n} = {n}\n" for n in range(1, 1001)))
        model = ScriptedModel([dict(action("finish", content="Sets values."),
                                    points=[{"path": "gateway/main.py", "symbol": "main", "text": "Starts a server."},
                                            {"path": "gateway/main.py", "symbol": "value_900", "text": "Sets 900."}])])
        output = []
        self.assertEqual(run(model, self.ws, "explain gateway/main.py", output=output.append), 2)
        self.assertIn("- gateway/main.py `main` (not found in the file): Starts a server.", "\n".join(output))
        self.assertIn("Answer check: No part named 'main' was found in gateway/main.py; that point is uncited.", output)
        # value_900 exists but lies past the preloaded lines, so the model never saw it.
        self.assertIn("Answer check: Citation not supported by retained evidence: gateway/main.py:900-900.", output)

    def test_preloaded_file_survives_compaction_beside_large_retrieval(self):
        from offline_agent.model import compact_messages
        self.put("big.py", "".join(f"value_{n} = {n}\n" for n in range(1, 1001)))

        class Knowledge:
            def retrieve(self, _query):
                return json.dumps({"results": [{"source": "notes.md", "start_line": 1, "end_line": 1, "text": "x" * 5000}]})
        model = ScriptedModel([action("finish", content="Done.")])
        run(model, self.ws, "explain big.py", knowledge=Knowledge(), output=lambda _: None)
        sent = json.dumps(compact_messages(model.calls[0], 12_000))
        self.assertIn("value_100 = 100", sent)
        self.assertIn(ANSWER_FROM_PRELOAD, sent)

    def test_change_requests_preload_plain_text_for_exact_search_blocks(self):
        self.put("calc.py", "def add(a, b):\n    return a + b\n")
        model = ScriptedModel([action("finish", content="Done."), action("finish", content="Done.")])
        run(model, self.ws, "Fix add in calc.py", output=lambda _: None)
        preload = model.calls[0][-1]["content"]
        self.assertIn("Lines 1-2 of 2:\ndef add(a, b):\n    return a + b\n", preload)
        self.assertNotIn("1| ", preload)
        self.assertNotIn(ANSWER_FROM_PRELOAD, json.dumps(model.calls[0]))

    def test_unique_file_names_resolve_and_ambiguous_missing_or_hidden_ones_do_not(self):
        from offline_agent.agent import named_files
        files = ["inventory/store.py", "a/main.py", "b/main.py", "README.md"]
        self.assertEqual(named_files("explain `store.py`, then README.md.", files), ["inventory/store.py", "README.md"])
        self.assertEqual(named_files("explain main.py and missing.py", files), [])
        self.assertEqual(named_files("compare a/main.py b/main.py README.md", files), ["a/main.py", "b/main.py"])
        self.put(".env", "SECRET=1\n")
        model = ScriptedModel([action("finish", content="Done.")])
        run(model, self.ws, "explain .env", output=lambda _: None)
        self.assertNotIn("SECRET", json.dumps(model.calls[0]))

    def test_large_file_shows_its_end_and_the_lines_the_task_names(self):
        self.put("catalog.py", "".join(f"row_{n} = {n}\n" for n in range(1, 1001)))
        model = ScriptedModel([action("finish", content="Done.")])
        output = []
        run(model, self.ws, "What is row_500 in catalog.py, and what is the last row?", output=output.append)
        preload = model.calls[0][-1]["content"]
        self.assertIn(" 500| row_500 = 500", preload)
        self.assertIn("1000| row_1000 = 1000", preload)
        self.assertIn("  1| row_1 = 1".strip(), preload)
        self.assertLess(len(preload.encode()), 7000)
        self.assertTrue(any(line.startswith("Preloaded catalog.py (lines 1-") and " and " in line for line in output))

    def test_change_request_shows_the_named_part_of_a_large_file_as_plain_text(self):
        self.put("catalog.py", "".join(f'("SKU-{n:04}", "Widget {n}", 1.5),\n' for n in range(1, 601)))
        model = ScriptedModel([action("finish", content="Done."), action("finish", content="Done.")])
        run(model, self.ws, "In catalog.py, change the price of SKU-0400 to 2.5", output=lambda _: None)
        preload = model.calls[0][-1]["content"]
        self.assertIn('("SKU-0400", "Widget 400", 1.5),', preload)
        self.assertNotIn("| (", preload)

    def test_large_named_file_is_paged_and_names_the_next_range(self):
        self.put("big.py", "".join(f"value_{n} = {n}\n" for n in range(1, 1001)))
        model = ScriptedModel([action("finish", content="Done.")])
        run(model, self.ws, "explain big.py", output=lambda _: None)
        preload = model.calls[0][-1]["content"]
        self.assertRegex(preload, r"Lines 1-\d+ of 1000; \d+ more lines, request content \"\d+-\d+\"")
        self.assertLess(len(preload.encode()), 6500)


class FinalAnswerPromptTests(TempWorkspace):
    def final_prompt(self, task):
        model = ScriptedModel([action("finish", content="Done.")])
        run(model, self.ws, task, max_steps=1, output=lambda _: None)
        return model.calls[0][-1]["content"]

    def test_entry_point_wording_only_for_entry_point_questions(self):
        prompt = self.final_prompt("explain chapter-06/services/gateway/app/main.py")
        self.assertTrue(prompt.startswith("Final-answer call"))
        self.assertNotIn("entry point", prompt)
        self.assertNotIn("services", prompt)
        self.assertIn("Cite the lines you rely on", prompt)
        self.assertIn("only entry point", self.final_prompt("Which services and entry points does this repo have?"))


class SessionTests(TempWorkspace):
    class Knowledge:
        refreshes = 0

        def retrieve(self, query):
            return json.dumps({"results": []})

        def refresh(self):
            self.refreshes += 1

    def test_follow_ups_see_earlier_answers_and_carried_evidence(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        model = ScriptedModel([action("read", "calc.py"), action("finish", content="add sums. [calc.py:1-2]"),
                               action("finish", content="It returns a + b. [calc.py:2]")])
        knowledge = self.Knowledge()
        session = Session(model, self.ws, knowledge=knowledge, output=lambda _: None)
        self.assertEqual(session.ask("What does add do?"), 0)
        self.assertEqual(session.ask("What does it return?"), 0)
        follow_up = model.calls[2][1]["content"]
        self.assertTrue(follow_up.startswith("Earlier in this session"))
        self.assertIn("Q: What does add do?\nA: add sums. [calc.py:1-2]", follow_up)
        self.assertIn("Current task: What does it return?", follow_up)
        self.assertEqual(len(model.calls[2]), 3)  # System, task, and retrieval: no earlier tool transcripts.
        self.assertEqual(knowledge.refreshes, 1)
        session.reset()
        self.assertEqual(session.history(), "")

    def test_repl_keeps_going_after_errors_and_supports_reset(self):
        class FakeSession:
            def __init__(self):
                self.asked, self.resets = [], 0

            def ask(self, task):
                self.asked.append(task)
                if task == "boom":
                    raise ValueError("model failed")
                return 0

            def reset(self):
                self.resets += 1

        session = FakeSession()
        inputs = iter(["  ", "/reset", "boom", "second", "quit"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(interactive(session, "first", prompt=lambda _: next(inputs)), 0)
        self.assertEqual(session.asked, ["first", "boom", "second"])
        self.assertEqual(session.resets, 1)
        self.assertIn("Error: model failed", errors.getvalue())

        def end_of_input(_):
            raise EOFError
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(interactive(FakeSession(), prompt=end_of_input), 0)

    def test_cli_streams_by_default_and_supports_interactive_mode(self):
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--workspace", str(self.root), "Explain"]), 0)
            self.assertIs(agent.call_args.kwargs["stream"], write_stream)
            main(["--workspace", str(self.root), "--no-stream", "Explain"])
            self.assertIsNone(agent.call_args.kwargs["stream"])
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.interactive", return_value=0) as repl, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--workspace", str(self.root), "-i"]), 0)
        session, first_task = repl.call_args.args
        self.assertIsInstance(session, Session)
        self.assertIsNone(first_task)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--workspace", str(self.root)])

    def test_history_is_bounded(self):
        session = Session(ScriptedModel([]), self.ws)
        session.turns = [(f"q{n}", "a" * 5000) for n in range(Session.TURNS)]
        self.assertLess(len(session.history().encode()), Session.TURNS * 1600)


class TaskKindTests(TempWorkspace):
    def test_tasks_are_classified_conservatively(self):
        for task, kind in (("Which function calls apply_discount?", "question"), ("Explain inventory/store.py", "question"),
                           ("Find all FastAPI entry points. Do not modify files.", "question"),
                           ("What does fix_order do?", "question"), ("How do I add a route?", "other"),
                           ("Fix the bug in store.py", "change"), ("Can you fix the bug?", "change"),
                           ("Please make retries configurable", "change"), ("Find and fix the bug", "other"),
                           ("Run the tests", "other"), ("Task", "other")):
            with self.subTest(task=task):
                self.assertEqual(task_kind(task), kind)

    def test_question_cannot_edit_files(self):
        (self.root / "pricing.py").write_text("def apply_discount(p):\n    return p\n")
        model = ScriptedModel([action("edit", "pricing.py", "x = 1\n"), action("finish", content="Nobody calls it.")])
        schemas = []
        chat = model.chat
        model.chat = lambda messages, schema: (schemas.append(schema), chat(messages, schema))[1]
        output = []
        run(model, self.ws, "Which function calls apply_discount?", apply=True, confirm=lambda _: True, output=output.append)
        self.assertNotIn("edit", schemas[0]["properties"]["action"]["enum"])
        self.assertIn("points", schemas[0]["required"])
        self.assertIn("only asks a question", model.calls[1][-1]["content"])
        self.assertEqual((self.root / "pricing.py").read_text(), "def apply_discount(p):\n    return p\n")

    def test_answers_to_any_question_can_cite_by_symbol(self):
        (self.root / "pricing.py").write_text("def apply_discount(p):\n    return p\n\n\ndef total(p):\n    return apply_discount(p)\n")
        model = ScriptedModel([action("read", "pricing.py"),
                               dict(action("finish", content="total calls it."),
                                    points=[{"path": "pricing.py", "symbol": "total", "text": "Calls apply_discount."}])])
        output = []
        self.assertEqual(run(model, self.ws, "Which function calls apply_discount?", output=output.append), 0)
        self.assertIn("total calls it.\n- pricing.py:5-6 `total`: Calls apply_discount.", output)

    def test_names_in_a_question_are_searched_before_the_model_is_asked(self):
        (self.root / "pricing.py").write_text("def apply_discount(p):\n    return p\n\n\ndef checkout_total(p):\n    return apply_discount(p)\n")
        model = ScriptedModel([action("finish", content="checkout_total calls it. [pricing.py:6]")])
        self.assertEqual(run(model, self.ws, "Which function calls apply_discount?", output=lambda _: None), 0)
        first = json.dumps(model.calls[0])
        self.assertIn("pricing.py:1 (defines it)", first)
        self.assertIn("pricing.py:6 (inside checkout_total)", first)

    def test_answer_cut_by_its_length_limit_ends_at_its_last_full_sentence(self):
        from offline_agent.agent import LIMIT_NOTE, OVERVIEW_CHARS, trim_at_limit
        (self.root / "worker.py").write_text("def review():\n    return 1\n")
        # The real failure: four sentences, cut mid-word at exactly 600 characters with a stray Chinese token.
        overview = ("It reads the queue. " * 40)[:OVERVIEW_CHARS - 9] + " saving评估"
        overview = overview[:OVERVIEW_CHARS]
        model = ScriptedModel([dict(action("finish", content=overview),
                                    points=[{"path": "worker.py", "symbol": "review", "text": "Returns 1."}])])
        output = []
        self.assertEqual(run(model, self.ws, "explain worker.py", output=output.append), 0)
        printed = "\n".join(output)
        self.assertNotIn("评估", printed)
        self.assertIn("It reads the queue.\n- worker.py:1-2 `review`: Returns 1.", printed)
        self.assertIn(LIMIT_NOTE, output)
        self.assertEqual(trim_at_limit("Short answer", 600), ("Short answer", False))
        self.assertEqual(trim_at_limit("no sentence end here at all", 10), ("no sentence end here at …", True))

    def test_streamed_answer_at_its_limit_is_not_printed_twice(self):
        from offline_agent.agent import FINAL_ANSWER_CHARS, LIMIT_NOTE
        text = ("Every step is local. " * 200)[:FINAL_ANSWER_CHARS]

        class Model:
            def chat(self, messages, schema, on_text=None):
                on_text(text)
                return action("finish", content=text)
        output, streamed = [], []
        run(Model(), self.ws, "Explain the design", output=output.append, stream=streamed.append)
        self.assertEqual("".join(streamed), text + "\n")
        self.assertIn(LIMIT_NOTE, output)
        self.assertFalse(any("Every step is local." in line for line in output))

    def test_streamed_answer_with_trailing_space_is_not_printed_twice(self):
        # The real failure: the streamed answer ended with a space, the points were joined to the stripped text,
        # and the whole answer was printed a second time.
        (self.root / "worker.py").write_text("def review():\n    return 1\n")

        class Model:
            def chat(self, messages, schema, on_text=None):
                on_text("It reviews rewrites. ")
                return dict(action("finish", content="It reviews rewrites. "),
                            points=[{"path": "worker.py", "symbol": "review", "text": "Returns 1."}])
        output, streamed = [], []
        run(Model(), self.ws, "What does the review function do?", output=output.append, stream=streamed.append)
        self.assertEqual("".join(streamed), "It reviews rewrites. \n")
        self.assertFalse(any("It reviews rewrites." in line for line in output))
        self.assertIn("- worker.py:1-2 `review`: Returns 1.", output)

    def test_message_field_is_bounded_in_every_schema(self):
        from offline_agent.agent import SCHEMA, cited_schema
        for schema in (SCHEMA, cited_schema(), cited_schema(["a.py"])):
            self.assertEqual(schema["properties"]["message"]["maxLength"], 160)

    def test_model_is_told_when_every_named_file_is_done(self):
        (self.root / "calc.py").write_text("x = 1\n")
        fix = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
        model = ScriptedModel([action("edit", "calc.py", fix), action("finish", content="Done.")])
        self.assertEqual(run(model, self.ws, "Fix calc.py", apply=True, confirm=lambda _: True, output=lambda _: None), 0)
        self.assertIn("File written. Every file the task names is now done.", model.calls[1][-1]["content"])

    def test_citing_the_function_around_a_search_match_points_at_the_match(self):
        (self.root / "cart.js").write_text("function total(p) {\n\n\n  return discount(p);\n}\n")
        model = ScriptedModel([dict(action("finish", content="total calls it."),
                                    points=[{"path": "cart.js", "symbol": "total", "text": "Calls discount."}])])
        output = []
        self.assertEqual(run(model, self.ws, "Which function calls discount_v2 or discount(?", output=output.append), 2)
        (self.root / "cart.js").write_text("function total(p) {\n\n\n  return apply_rate(p);\n}\n")
        model = ScriptedModel([dict(action("finish", content="total calls it."),
                                    points=[{"path": "cart.js", "symbol": "total", "text": "Calls it."}])])
        output = []
        self.assertEqual(run(model, self.ws, "Which function calls apply_rate?", output=output.append), 0)
        self.assertIn("total calls it.\n- cart.js:4 `total`: Calls it.", output)

    def test_write_may_remove_lines_when_the_task_says_so(self):
        (self.root / "calc.py").write_text("# Old header\nx = 1\ny = 2\nz = 3\n")
        model = ScriptedModel([action("write", "calc.py", "x = 1\ny = 2\nz = 3\n"), action("finish", content="Removed.")])
        run(model, self.ws, "Remove the header comment from calc.py", apply=True, confirm=lambda _: True, output=lambda _: None)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 1\ny = 2\nz = 3\n")

    def test_ollama_repeat_abort_gets_one_correction(self):
        model = ScriptedModel([InvalidReply("the reply kept repeating the same text"), action("finish", content="Short.")])
        self.assertEqual(run(model, self.ws, "Explain", output=lambda _: None), 0)
        self.assertIn("kept repeating", model.calls[1][-1]["content"])

    def test_change_task_finished_without_a_change_is_nudged_once_then_flagged(self):
        (self.root / "calc.py").write_text("x = 1\n")
        model = ScriptedModel([action("finish", content="x = 2"), action("finish", content="x = 2")])
        output = []
        self.assertEqual(run(model, self.ws, "Fix calc.py", apply=True, confirm=lambda _: True, output=output.append), 2)
        self.assertIn("without changing any file", model.calls[1][-1]["content"])
        self.assertIn("Answer check: The task asks for a change, but no file was changed.", output)

    def test_change_report_needs_no_line_citation(self):
        (self.root / "calc.py").write_text("x = 1\n")
        model = ScriptedModel([action("read", "calc.py"), action("write", "new.py", "y = 2\n"), action("finish", content="Created new.py.")])
        self.assertEqual(run(model, self.ws, "Create new.py", apply=True, confirm=lambda _: True, output=lambda _: None), 0)


class LoopTests(TempWorkspace):
    def test_alternating_lookups_are_caught(self):
        for name in ("a.py", "b.py"):
            (self.root / name).write_text("x = 1\n")
        model = ScriptedModel([action("read", "a.py"), action("read", "b.py"), action("read", "a.py"), action("finish", content="Done.")])
        with patch.object(self.ws, "read", wraps=self.ws.read) as read:
            run(model, self.ws, "Summarize things", output=lambda _: None)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(len(model.calls), 4)

    def test_repeated_failed_edit_stops_instead_of_using_every_step(self):
        (self.root / "calc.py").write_text("x = 1\n")
        bad = action("edit", "calc.py", "<<<<<<< SEARCH\nzzz\n=======\ny\n>>>>>>> REPLACE")
        model = ScriptedModel([bad, bad, action("finish", content="Could not edit.")])
        output = []
        self.assertEqual(run(model, self.ws, "Fix calc.py", apply=True, confirm=lambda _: True, output=output.append), 2)
        self.assertEqual(len(model.calls), 3)

    def test_same_read_is_allowed_after_a_write(self):
        (self.root / "calc.py").write_text("x = 1\n")
        fix = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
        model = ScriptedModel([action("read", "calc.py"), action("edit", "calc.py", fix), action("read", "calc.py"),
                               action("finish", content="Now x = 2.")])
        self.assertEqual(run(model, self.ws, "Fix calc.py", apply=True, confirm=lambda _: True, output=lambda _: None), 0)
        self.assertIn("x = 2", model.calls[3][-1]["content"])

    def test_read_may_be_repeated_once_its_result_was_compacted(self):
        (self.root / "big.py").write_text("x = 1\n" * 900)
        (self.root / "other.py").write_text("y = 2\n" * 900)
        model = ScriptedModel([action("read", "big.py"), action("read", "other.py"), action("read", "big.py"), action("finish", content="Done.")])
        model.prompt_budget = 8000
        with patch.object(self.ws, "read", wraps=self.ws.read) as read:
            run(model, self.ws, "Summarize things", output=lambda _: None)
        self.assertEqual(read.call_count, 3)


class ReloadWarningTests(TempWorkspace):
    def test_reload_after_the_first_step_warns_once(self):
        class Reloading(ScriptedModel):
            last_metrics = {"load_seconds": 60.0, "total_seconds": 70.0}
        (self.root / "a.py").write_text("x = 1\n")
        model = Reloading([action("read", "a.py"), action("read", "a.py", "1-1"), action("finish", content="Done.")])
        output = []
        run(model, self.ws, "Summarize things", output=output.append)
        self.assertEqual(sum(line.startswith("Warning: Ollama reloaded") for line in output), 1)

    def test_doctor_names_other_loaded_models(self):
        with patch("offline_agent.cli.Ollama") as ollama, contextlib.redirect_stdout(io.StringIO()) as printed:
            ollama.return_value.other_loaded_models.return_value = ["qwen2.5:14b"]
            self.assertEqual(main(["--doctor"]), 0)
        self.assertIn("qwen2.5:14b", printed.getvalue())
