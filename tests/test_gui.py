"""The local GUI server: only this machine and the printed link may use it, and chat works end to end."""
import contextlib
import http.client
import io
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.cli import main
from offline_agent.gui import PAGE, App, directories, make_server


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies = list(replies)
        self.release = threading.Event()
        self.release.set()

    def chat(self, _messages, _schema, on_text=None):
        self.release.wait(5)
        reply = self.replies.pop(0)
        if on_text and reply.get("action") == "finish":
            on_text(reply["content"])
        return reply


class GuiServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name, "repo")
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "pkg" / "pricing.py").write_text("def apply_discount(p):\n    return p\n\n\ndef checkout_total(p):\n"
                                                      "    return apply_discount(p)\n")
        (self.repo / ".env").write_text("SECRET=1\n")
        (self.repo / ".gitignore").write_text("private/\n")
        (self.repo / "private").mkdir()
        (self.repo / "private" / "keys.py").write_text("KEY = 'x'\n")
        self.model = ScriptedModel([])
        self.app = App(self.model)
        self.server, self.token, self.url = make_server(self.app)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), thread.join()))

    def request(self, method, path, body=None, *, token=True, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{self.port}", "Content-Type": "application/json"}
        if token:
            headers["X-CodeHaven-Token"] = self.token
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def json(self, method, path, body=None, **options):
        response, data = self.request(method, path, body, **options)
        return response.status, json.loads(data)

    def open_repo(self):
        status, summary = self.json("POST", "/api/open", {"path": str(self.repo)})
        self.assertEqual(status, 200, summary)
        return summary

    def test_page_needs_the_token_and_loads_nothing_from_the_internet(self):
        response, _ = self.request("GET", "/", token=False)
        self.assertEqual(response.status, 403)
        response, page = self.request("GET", "/?token=" + self.token, token=False)
        self.assertEqual(response.status, 200)
        policy = response.getheader("Content-Security-Policy") or ""
        self.assertIn("default-src 'none'", policy)
        found = re.search(r"'nonce-([^']+)'", policy)
        assert found is not None
        nonce = found[1]
        self.assertIn(f'<script nonce="{nonce}">'.encode(), page)
        self.assertNotRegex(page.decode(), r"https?://")  # No CDN, font, or other remote resource.
        self.assertTrue(self.url.startswith(f"http://127.0.0.1:{self.port}/?token="))

    def test_other_hosts_and_missing_tokens_are_refused(self):
        self.assertEqual(self.json("GET", "/api/status", token=False)[0], 403)
        # A DNS-rebinding page reaches 127.0.0.1 under its own name; the Host header gives it away.
        self.assertEqual(self.json("GET", "/api/status", host=f"evil.example:{self.port}")[0], 403)
        self.assertEqual(self.json("GET", "/api/status", host=f"localhost:{self.port}")[0], 200)
        self.assertEqual(self.json("POST", "/api/open", {"path": str(self.repo)}, token=False)[0], 403)

    def test_indexing_reports_counts_and_rejects_bad_paths(self):
        summary = self.open_repo()
        self.assertEqual(summary["repository"], str(self.repo.resolve()))
        self.assertEqual(summary["files"], 1)  # The ignored private/ folder and hidden .env are left out.
        status, error = self.json("POST", "/api/open", {"path": "relative/path"})
        self.assertEqual(status, 400)
        self.assertIn("full path", error["error"])
        self.assertEqual(self.json("POST", "/api/open", {"path": str(self.repo / "missing")})[0], 400)

    def test_question_streams_text_then_a_checked_answer(self):
        self.open_repo()
        self.model.replies = [dict(action("finish", content="checkout_total calls it."),
                                   points=[{"path": "pkg/pricing.py", "symbol": "checkout_total", "text": "Calls apply_discount."}])]
        response, data = self.request("POST", "/api/ask", {"question": "Which function calls apply_discount?"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/x-ndjson; charset=utf-8")
        events = [json.loads(line) for line in data.decode().splitlines()]
        self.assertIn({"type": "text", "text": "checkout_total calls it."}, events)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["complete"])
        self.assertEqual(done["answer"], "checkout_total calls it.\n- pkg/pricing.py:6 `checkout_total`: Calls apply_discount.")
        self.assertEqual((done["exit_code"], done["issues"]), (0, []))
        self.assertTrue(any(event["type"] == "log" and "Step 1 (max 12)" in event["text"] for event in events))

    def test_follow_ups_remember_the_conversation_until_a_new_chat(self):
        self.open_repo()
        self.model.replies = [action("finish", content="First."), action("finish", content="Second.")]
        self.request("POST", "/api/ask", {"question": "What is in pkg?"})
        session = self.app.session
        assert session is not None
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(self.json("POST", "/api/reset", {})[0], 200)
        self.assertEqual(session.turns, [])

    def test_a_second_question_waits_for_the_first(self):
        self.open_repo()
        self.model.replies = [action("finish", content="Slow answer.")]
        self.model.release.clear()
        first = threading.Thread(target=self.request, args=("POST", "/api/ask", {"question": "What is in pkg?"}))
        first.start()
        for _ in range(100):
            if self.app.busy:
                break
            threading.Event().wait(0.02)
        status, error = self.json("POST", "/api/ask", {"question": "And now?"})
        self.assertEqual(status, 409)
        self.assertIn("still working", error["error"])
        self.model.release.set()
        first.join(5)
        self.assertFalse(self.app.busy)

    def test_model_errors_are_reported_and_the_server_keeps_going(self):
        self.open_repo()
        self.model.replies = []  # pop from an empty list raises inside the agent
        _, data = self.request("POST", "/api/ask", {"question": "What is in pkg?"})
        self.assertEqual(json.loads(data.decode().splitlines()[-1])["type"], "error")
        self.assertFalse(self.app.busy)
        self.assertEqual(self.json("GET", "/api/status")[0], 200)

    def test_questions_need_a_repository_and_text(self):
        self.assertEqual(self.json("POST", "/api/ask", {"question": "Hi?"})[1]["error"], "Choose a repository first.")
        self.open_repo()
        self.assertEqual(self.json("POST", "/api/ask", {"question": "   "})[0], 400)

    def test_cited_lines_follow_the_same_rules_as_the_agent(self):
        self.open_repo()
        status, data = self.json("GET", "/api/file?path=repo/pkg/pricing.py&start=5&end=6")
        self.assertEqual(status, 200)
        self.assertEqual((data["path"], data["first"], data["total"]), ("pkg/pricing.py", 1, 6))
        self.assertEqual(data["lines"][4], "def checkout_total(p):")
        for path in ("../repo/pkg/pricing.py", ".env", "private/keys.py", str(self.repo / "pkg" / "pricing.py")):
            with self.subTest(path=path):
                self.assertEqual(self.json("GET", "/api/file?path=" + path)[0], 400)

    def test_folder_browser_lists_visible_folders_and_marks_repositories(self):
        (self.repo / ".git").mkdir()
        listing = directories(str(self.repo))
        self.assertTrue(listing["repository"])
        self.assertEqual([entry["name"] for entry in listing["entries"]], ["pkg", "private"])
        parent = directories(self.temp.name)
        self.assertEqual([(entry["name"], entry["repository"]) for entry in parent["entries"]], [("repo", True)])
        with self.assertRaises(ValueError):
            directories("relative")
        status, data = self.json("GET", "/api/dirs?path=" + str(self.repo))
        self.assertEqual((status, data["path"]), (200, str(self.repo.resolve())))


class ModelPickerTests(unittest.TestCase):
    def test_picker_lists_models_and_switches_between_questions(self):
        class Pickable(ScriptedModel):
            model = "qwen2.5-coder:7b"

            def local_models(self):
                return [{"name": "qwen2.5-coder:7b"}, {"name": "qwen2.5:14b"}]

            def use(self, name):
                if name not in ("qwen2.5-coder:7b", "qwen2.5:14b"):
                    raise ValueError(f"Model {name!r} is not installed.")
                self.model = name
        app = App(Pickable([]))
        self.assertEqual(app.models()["current"], "qwen2.5-coder:7b")
        self.assertEqual([m["name"] for m in app.models()["models"]], ["qwen2.5-coder:7b", "qwen2.5:14b"])
        self.assertEqual(app.use_model("qwen2.5:14b")["current"], "qwen2.5:14b")
        with self.assertRaisesRegex(ValueError, "not installed"):
            app.use_model("missing:latest")
        self.assertEqual(app.summary()["model"], "qwen2.5:14b")
        self.assertFalse(app.busy)


class GuiCliTests(unittest.TestCase):
    def test_gui_starts_with_another_local_model_when_the_default_is_missing(self):
        with patch("offline_agent.cli.Ollama") as ollama, patch("offline_agent.gui.serve", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            client = ollama.return_value
            client.check_model.side_effect = ValueError("not installed")
            client.local_models.return_value = [{"name": "llama3:latest"}]
            client.other_loaded_models.return_value = []
            self.assertEqual(main(["--gui", "--no-browser"]), 0)
        client.use.assert_called_once_with("llama3:latest")
        self.assertIn("starting with llama3:latest", printed.getvalue())

    def test_gui_flag_starts_the_server_without_a_task(self):
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.gui.serve", return_value=0) as serve, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--gui", "--no-browser", "--port", "8765"]), 0)
        self.assertEqual(serve.call_args.kwargs["port"], 8765)
        self.assertFalse(serve.call_args.kwargs["open_browser"])
        self.assertIsNone(serve.call_args.kwargs["repository"])

    def test_page_file_is_packaged(self):
        self.assertTrue(PAGE.is_file())
        self.assertIn("gui.html", Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text())
