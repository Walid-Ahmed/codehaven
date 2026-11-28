"""The local GUI server: only this machine and the printed link may use it, and chat works end to end."""
import contextlib
import http.client
import io
import json
import re
import subprocess
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from offline_agent.cli import main
from offline_agent.gui import CLONE_OFF, PAGE, PAGE_PARTS, App, directories, github_repository, make_server, page_text, upload_name


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
        self.assertEqual((done["remembered"], done["memory"]), (1, 8))
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
        with contextlib.redirect_stderr(io.StringIO()) as log:
            _, data = self.request("POST", "/api/ask", {"question": "What is in pkg?"})
        last = json.loads(data.decode().splitlines()[-1])
        self.assertEqual(last["type"], "error")
        # An unexpected problem is put in plain words on the page; its traceback goes to the server's terminal.
        self.assertIn("did not expect (IndexError", last["message"])
        self.assertIn("Your files were not changed", last["message"])
        self.assertIn("Traceback", log.getvalue())
        self.assertFalse(self.app.busy)
        self.assertEqual(self.json("GET", "/api/status")[0], 200)

    def test_an_unexpected_problem_gets_a_plain_message_not_a_dropped_connection(self):
        with patch.object(self.app, "open", side_effect=KeyError("pkg")), contextlib.redirect_stderr(io.StringIO()):
            status, data = self.json("POST", "/api/open", {"path": str(self.repo)})
        self.assertEqual(status, 500)
        self.assertIn("did not expect (KeyError", data["error"])
        with patch.object(self.app, "summary", side_effect=RuntimeError("boom")), contextlib.redirect_stderr(io.StringIO()):
            status, data = self.json("GET", "/api/status")
        self.assertEqual((status, "RuntimeError: boom" in data["error"]), (500, True))
        self.assertIn("error", self.json("GET", "/api/files")[1])  # The server keeps going.

    def test_questions_need_a_repository_and_text(self):
        self.assertEqual(self.json("POST", "/api/ask", {"question": "Hi?"})[1]["error"], "Choose a repository or upload a file first.")
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

    def test_a_map_is_drawn_from_the_code_without_the_model(self):
        self.assertEqual(self.json("POST", "/api/map", {"paths": ["pkg/pricing.py"]})[1]["error"],
                         "Choose a repository or upload a file first.")
        self.open_repo()
        status, listing = self.json("GET", "/api/files")
        self.assertEqual((status, listing["files"], listing["limited"]), (200, ["pkg/pricing.py"], False))
        status, data = self.json("POST", "/api/map", {"paths": ["repo/pkg/pricing.py"]})
        self.assertEqual(status, 200, data)
        parts = data["files"][0]["parts"]
        self.assertEqual([(part["name"], part["start"], part["end"], part["calls"]) for part in parts],
                         [("apply_discount", 1, 2, []), ("checkout_total", 5, 6, ["apply_discount"])])
        self.assertEqual(self.model.replies, [])  # No model call was needed.
        for paths in ([], [".env"], ["private/keys.py"], ["../repo/pkg/pricing.py"], "pkg/pricing.py"):
            with self.subTest(paths=paths):
                self.assertEqual(self.json("POST", "/api/map", {"paths": paths})[0], 400)
        self.assertEqual(self.request("POST", "/api/map", {"paths": ["pkg/pricing.py"]}, token=False)[0].status, 403)

    def test_flashcards_stream_cards_with_looked_up_lines(self):
        self.open_repo()
        self.model.replies = [{"cards": [{"question": "What does checkout_total call?", "answer": "apply_discount.",
                                          "symbol": "checkout_total"}]}]
        response, data = self.request("POST", "/api/flashcards", {"paths": ["pkg/pricing.py"], "count": 3})
        self.assertEqual(response.status, 200)
        events = [json.loads(line) for line in data.decode().splitlines()]
        self.assertEqual([event["type"] for event in events], ["file", "cards", "done"])
        self.assertEqual((events[1]["cards"][0]["start"], events[1]["cards"][0]["end"]), (5, 6))
        self.assertFalse(self.app.busy)
        self.assertEqual(self.json("POST", "/api/flashcards", {"paths": [".env"]})[0], 400)
        self.assertEqual(self.json("POST", "/api/flashcards", {"paths": ["pkg/pricing.py"], "count": 99})[0], 400)
        self.assertFalse(self.app.busy)

    def test_flashcards_wait_for_a_question_in_progress(self):
        self.open_repo()
        self.model.replies = [action("finish", content="Slow answer.")]
        self.model.release.clear()
        first = threading.Thread(target=self.request, args=("POST", "/api/ask", {"question": "What is in pkg?"}))
        first.start()
        for _ in range(100):
            if self.app.busy:
                break
            threading.Event().wait(0.02)
        self.assertEqual(self.json("POST", "/api/flashcards", {"paths": ["pkg/pricing.py"]})[0], 409)
        self.assertEqual(self.json("POST", "/api/map", {"paths": ["pkg/pricing.py"]})[0], 200)  # The map needs no model.
        self.model.release.set()
        first.join(5)

    def test_study_aids_default_to_the_uploaded_file(self):
        self.assertEqual(self.upload("notes.py", b"def greet(name):\n    return name\n")[0], 200)
        status, data = self.json("POST", "/api/map", {"paths": []})
        self.assertEqual((status, data["files"][0]["path"], data["files"][0]["parts"][0]["name"]), (200, "notes.py", "greet"))
        self.model.replies = [{"cards": [{"question": "What does greet return?", "answer": "Its name argument.", "symbol": "greet"}]}]
        _, data = self.request("POST", "/api/flashcards", {})
        cards = json.loads(data.decode().splitlines()[1])["cards"]
        self.assertEqual((cards[0]["path"], cards[0]["start"], cards[0]["end"]), ("notes.py", 1, 2))

    def test_a_negative_request_length_is_refused_not_waited_on(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest("POST", "/api/reset")
        connection.putheader("X-CodeHaven-Token", self.token)
        connection.putheader("Content-Length", "-1")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 400)
        connection.close()

    def test_long_term_memory_is_off_unless_asked_for_and_can_be_forgotten(self):
        self.open_repo()
        answer = "checkout_total calls apply_discount (pkg/pricing.py:5-6)."
        self.model.replies = [action("read", "pkg/pricing.py"), action("finish", content=answer)]
        self.request("POST", "/api/ask", {"question": "List what checkout_total calls"})
        self.assertEqual((self.json("GET", "/api/status")[1]["remember"], self.app.memory), (False, None))

        folder = Path(self.temp.name, "memory")
        self.app.remember, self.app.memory_folder = True, folder
        self.open_repo()
        self.model.replies = [action("read", "pkg/pricing.py"), action("finish", content=answer)]
        _, data = self.request("POST", "/api/ask", {"question": "List what checkout_total calls"})
        done = json.loads(data.decode().splitlines()[-1])
        self.assertEqual((done["exit_code"], done["kept"], done["saved"]), (0, True, 1))
        self.assertEqual(len(list(folder.iterdir())), 1)
        self.assertEqual(sorted(path.name for path in self.repo.iterdir()), [".env", ".gitignore", "pkg", "private"])
        self.assertEqual(self.open_repo()["saved"], 1)  # Still there after reopening.
        self.assertEqual(self.request("POST", "/api/forget", {}, token=False)[0].status, 403)
        status, summary = self.json("POST", "/api/forget", {})
        self.assertEqual((status, summary["saved"], list(folder.iterdir())), (200, 0, []))

    def test_uploaded_files_are_never_remembered(self):
        self.app.remember, self.app.memory_folder = True, Path(self.temp.name, "memory")
        status, summary = self.upload("notes.py", b"def greet(name):\n    return name\n")
        self.assertEqual((status, summary["remember"], summary["saved"], self.app.memory), (200, True, 0, None))

    def upload(self, name, data, *, token=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/octet-stream"}
        if token:
            headers["X-CodeHaven-Token"] = self.token
        connection.request("POST", "/api/upload?name=" + quote(name), body=data, headers=headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def test_an_uploaded_file_is_chatted_about_without_a_repository(self):
        original = Path(self.temp.name, "notes v2.py")
        original.write_text("def total(items):\n    return sum(items)\n")
        status, summary = self.upload(str(original), original.read_bytes())
        self.assertEqual(status, 200, summary)
        self.assertEqual((summary["upload"], summary["repository"], summary["files"]), ("notes_v2.py", None, 1))
        folder = self.app.upload_folder
        assert folder is not None
        self.assertEqual((folder / "notes_v2.py").read_bytes(), original.read_bytes())
        self.assertEqual(self.json("GET", "/api/status")[1]["upload"], "notes_v2.py")
        self.model.replies = [action("finish", content="It adds the items.")]
        _, data = self.request("POST", "/api/ask", {"question": "What does this file do?"})
        events = [json.loads(line) for line in data.decode().splitlines()]
        self.assertEqual(events[-1]["type"], "done")
        # Unnamed in the question, the only file is still preloaded for the model.
        self.assertTrue(any(e["type"] == "log" and e["text"].startswith("Preloaded notes_v2.py") and "this chat is about" in e["text"] for e in events))
        self.assertEqual(self.json("GET", "/api/file?path=notes_v2.py&start=2")[1]["lines"][1], "    return sum(items)")
        self.open_repo()  # Opening something else deletes the temporary copy, never the original.
        self.assertFalse(folder.exists())
        self.assertTrue(original.exists())
        self.assertIsNone(self.json("GET", "/api/status")[1]["upload"])

    def test_an_upload_keeps_the_conversation_and_a_repository_starts_a_new_one(self):
        self.open_repo()
        self.model.replies = [action("finish", content="It applies a discount.")]
        self.request("POST", "/api/ask", {"question": "What does pkg/pricing.py do?"})
        self.assertEqual(self.upload("notes.py", b"def greet(name):\n    return name\n")[0], 200)
        session = self.app.session
        self.assertEqual([question for question, _ in session.turns], ["What does pkg/pricing.py do?"])
        self.assertIn("What does pkg/pricing.py do?", session.history())
        self.assertEqual(session.evidence, [])  # It quoted files that are no longer open.
        self.model.replies = [action("finish", content="It returns the name.")]
        self.request("POST", "/api/ask", {"question": "And this file?"})
        self.assertEqual(len(self.app.session.turns), 2)
        self.json("POST", "/api/reset", {})
        self.assertEqual(self.app.session.turns, [])
        self.model.replies = [action("finish", content="It returns the name.")]
        self.request("POST", "/api/ask", {"question": "What does it do?"})
        self.open_repo()
        self.assertEqual(self.app.session.turns, [])

    def test_uploads_must_be_small_named_utf8_text_and_need_the_token(self):
        self.assertEqual(self.upload("a.py", b"x = 1\n", token=False)[0], 403)
        for name, data, message in [("big.txt", b"a" * 64_001, "64 KB"), ("image.png", b"\x89PNG\0\0", "binary"),
                                    ("latin.txt", "caf\xe9".encode("latin-1"), "UTF-8"), (".env", b"SECRET=1\n", "Hidden"),
                                    ("", b"x\n", "needs a name")]:
            with self.subTest(name=name):
                status, error = self.upload(name, data)
                self.assertEqual(status, 400)
                self.assertIn(message, error["error"])
        self.assertIsNone(self.app.upload_folder)
        self.assertFalse(self.app.busy)

    def test_upload_names_keep_only_a_safe_file_name(self):
        self.assertEqual(upload_name("C:\\Users\\me\\report (final).md"), "report__final_.md")
        self.assertEqual(upload_name("../../etc/passwd"), "passwd")
        self.assertEqual(upload_name("café.py"), "caf_.py")
        with self.assertRaises(ValueError):
            upload_name("../")

    def test_excluded_uploads_are_refused_and_the_copy_removed(self):
        app = App(self.model, excludes=("*.log",))
        with self.assertRaisesRegex(ValueError, "excluded"):
            app.open_file("server.log", b"started\n")
        self.assertIsNone(app.upload_folder)
        app.open_file("server.txt", b"started\n")
        folder = app.upload_folder
        assert folder is not None
        app.close()
        self.assertFalse(folder.exists())

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


class FakeGit:
    """Stands in for `git clone`: records the command and writes a small repository, with no network."""

    def __init__(self, returncode=0, stderr=b"", files=None):
        self.returncode, self.stderr = returncode, stderr
        self.files = {"app.py": "def main():\n    return 1\n"} if files is None else files
        self.commands, self.envs = [], []

    def __call__(self, command, *, stderr, env, **_options):
        self.commands.append(command)
        self.envs.append(env)
        target = Path(command[-1])
        if self.returncode == 0:
            for name, text in self.files.items():
                (target / name).write_text(text)
        stderr.write(self.stderr)
        process = unittest.mock.Mock(returncode=self.returncode)
        process.poll.return_value = self.returncode
        return process


class GitHubCloneTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name, "repos")
        self.app = App(ScriptedModel([]), allow_clone=True, clone_folder=self.folder)

    def test_cloning_is_off_unless_the_server_was_started_with_allow_clone(self):
        app = App(ScriptedModel([]), clone_folder=self.folder)
        fake = FakeGit()
        with patch("offline_agent.gui.subprocess.Popen", fake), self.assertRaisesRegex(ValueError, "--allow-clone"):
            app.clone("https://github.com/octo/hello")
        self.assertEqual(fake.commands, [])  # Nothing was run, so nothing touched the network.
        self.assertFalse(self.folder.exists())
        self.assertFalse(app.summary()["clone"])
        self.assertIn("--allow-clone", CLONE_OFF)

    def test_addresses_must_name_a_github_repository_without_secrets(self):
        for address, expected in [("https://github.com/octo/hello", ("octo", "hello", False)),
                                  ("github.com/octo/hello.git", ("octo", "hello", False)),
                                  ("https://github.com/octo/hello/tree/main/src", ("octo", "hello", False)),
                                  ("git@github.com:octo/hello.git", ("octo", "hello", True))]:
            with self.subTest(address=address):
                self.assertEqual(github_repository(address), expected)
        for address, message in [("https://user:ghp_secret@github.com/octo/hello", "without a password"),
                                 ("https://gitlab.com/octo/hello", "Only github.com"),
                                 ("https://github.com:8443/octo/hello", "Only github.com"),
                                 ("file:///etc/octo/hello", "GitHub address"),
                                 ("https://github.com/octo", "owner and the repository"),
                                 ("https://github.com/-octo/hello", "does not look like"),
                                 ("https://github.com/octo/..", "does not look like")]:
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, message):
                github_repository(address)

    def test_a_clone_downloads_one_snapshot_without_credentials_then_indexes_it(self):
        fake = FakeGit()
        with patch("offline_agent.gui.subprocess.Popen", fake):
            summary = self.app.clone("https://github.com/octo/hello")
        target = self.folder / "octo" / "hello"
        self.assertEqual((summary["repository"], summary["files"]), (str(target.resolve()), 1))
        self.assertEqual((summary["cloned"], summary["source"], summary["clone"]), (True, "github.com/octo/hello", True))
        command, env = fake.commands[0], fake.envs[0]
        self.assertEqual(command[-2:], ["https://github.com/octo/hello.git", command[-1]])
        for option in ("--depth", "--single-branch", "--no-tags", "--template=", "protocol.allow=never",
                       "core.symlinks=false"):
            self.assertIn(option, command)
        self.assertEqual((env["GIT_TERMINAL_PROMPT"], env["GIT_LFS_SKIP_SMUDGE"]), ("0", "1"))
        self.assertEqual([p.name for p in (self.folder / "octo").iterdir()], ["hello"])  # The partial folder is gone.
        self.assertFalse(self.app.busy)

    def test_a_repository_cloned_before_opens_from_disk_without_git_clone(self):
        target = self.folder / "octo" / "hello"
        target.mkdir(parents=True)
        (target / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", "git@github.com:Octo/Hello.git"], check=True)
        with patch("offline_agent.gui.download") as download:
            summary = self.app.clone("https://github.com/octo/hello")
            self.assertEqual(summary["cloned"], False)
            other = self.folder / "octo" / "other"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.app.clone("github.com/octo/other")
        download.assert_not_called()

    def test_private_or_missing_repositories_explain_the_sign_in_and_leave_nothing(self):
        fake = FakeGit(128, b"remote: Repository not found.\nfatal: repository 'https://github.com/octo/secret.git/' not found\n")
        with patch("offline_agent.gui.subprocess.Popen", fake), \
             self.assertRaisesRegex(ValueError, "never asks for your GitHub password"):
            self.app.clone("https://github.com/octo/secret")
        self.assertEqual(list((self.folder / "octo").iterdir()), [])
        self.assertIsNone(self.app.workspace)
        self.assertFalse(self.app.busy)

    def test_downloads_over_the_size_limit_are_stopped_and_removed(self):
        fake = FakeGit(files={"big.bin": "x" * 2000})
        fake_process = fake.__call__

        def running(*args, **kwargs):
            process = fake_process(*args, **kwargs)
            process.poll.return_value = None  # Still downloading when the size is checked.
            return process
        with patch("offline_agent.gui.subprocess.Popen", side_effect=running), \
             patch("offline_agent.gui.CLONE_BYTES", 1000), self.assertRaisesRegex(ValueError, "larger than"):
            self.app.clone("https://github.com/octo/huge")
        self.assertEqual(list((self.folder / "octo").iterdir()), [])

    def test_the_clone_endpoint_needs_the_token(self):
        server, token, _ = make_server(self.app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        port = server.server_address[1]
        for headers, expected in [({}, 403), ({"X-CodeHaven-Token": token}, 400)]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            connection.request("POST", "/api/clone", body=json.dumps({"address": "https://gitlab.com/a/b"}),
                               headers={"Host": f"127.0.0.1:{port}", **headers})
            self.assertEqual(connection.getresponse().status, expected)
            connection.close()


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
        self.assertFalse(serve.call_args.kwargs["allow_clone"])  # Fully offline unless asked for.

    def test_allow_clone_flag_reaches_the_server(self):
        with patch("offline_agent.cli.Ollama"), patch("offline_agent.gui.serve", return_value=0) as serve, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--gui", "--no-browser", "--allow-clone", "--clone-dir", "/tmp/codehaven-repos"]), 0)
        self.assertTrue(serve.call_args.kwargs["allow_clone"])
        self.assertEqual(serve.call_args.kwargs["clone_folder"], "/tmp/codehaven-repos")

    def test_page_files_are_packaged_and_served_as_one_page(self):
        packaged = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
        for name in ["gui.html"] + [name for _, name in PAGE_PARTS]:
            self.assertTrue(PAGE.with_name(name).is_file(), name)
            self.assertIn(f'"{name}"', packaged)
        page = page_text()
        for mark, _ in PAGE_PARTS:
            self.assertNotIn(mark, page)  # Every part was put in its place.
        self.assertIn('"use strict";', page.split("<script", 1)[1][:60])  # Still the script's first statement.
        self.assertNotIn("<script src", page)
        self.assertNotIn("<link", page)
