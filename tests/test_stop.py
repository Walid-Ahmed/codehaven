"""The Stop button: ending an answer that is waiting for the local model or being written."""
import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from offline_agent.gui import page_text, App, make_server
from offline_agent.model import Ollama, Stopped


def wait_until(check, seconds=5):
    """Poll until check() is true; the tests then assert on what they waited for."""
    deadline = time.monotonic() + seconds
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return check()


class OllamaStopTests(unittest.TestCase):
    """A loopback server stands in for Ollama, so the real connection is the thing being stopped."""

    def serve(self, pieces):
        """Start a fake Ollama that sends pieces of a reply and then never finishes it."""
        self.arrived, self.hold = threading.Event(), threading.Event()
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if pieces:
                    self.send_response(200)
                    self.end_headers()
                    for piece in pieces:
                        self.wfile.write(json.dumps({"message": {"content": piece}}).encode() + b"\n")
                    self.wfile.flush()
                test.arrived.set()
                test.hold.wait(10)  # A model that is still loading, or still writing.

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.hold.set(), server.shutdown(), server.server_close(), thread.join()))
        return Ollama(endpoint=f"http://127.0.0.1:{server.server_port}")

    def stopped_chat(self, client):
        """Run a chat on another thread, stop it once the server has the request, and return how it ended."""
        outcome = []

        def chat():
            try:
                outcome.append(client.chat([{"role": "user", "content": "hello"}], {"type": "object"}))
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=chat, daemon=True)
        thread.start()
        self.assertTrue(self.arrived.wait(5))
        started = time.monotonic()
        client.stop()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "the chat kept waiting after stop()")
        self.assertLess(time.monotonic() - started, 3)
        return outcome[0]

    def test_stop_ends_the_wait_for_a_model_that_has_not_replied_yet(self):
        client = self.serve([])
        self.assertIsInstance(self.stopped_chat(client), Stopped)

    def test_stop_ends_an_answer_that_is_being_written(self):
        client = self.serve(['{"action": "finish", ', '"content": "Half an ans'])
        self.assertIsInstance(self.stopped_chat(client), Stopped)

    def test_chats_are_refused_after_a_stop_until_allowed_again(self):
        client = self.serve([])
        client.stop()
        with self.assertRaises(Stopped):
            client.chat([{"role": "user", "content": "hello"}], {"type": "object"})
        self.assertFalse(self.arrived.is_set())  # Nothing was sent to the model.
        client.allow()
        self.assertIsInstance(self.stopped_chat(client), Stopped)  # It reached the server this time.

    def test_a_failed_request_is_still_an_error_when_nobody_stopped_it(self):
        client = Ollama(endpoint="http://127.0.0.1:9")  # Nothing listens on the discard port.
        with self.assertRaisesRegex(ValueError, "Local Ollama request failed"):
            client.chat([{"role": "user", "content": "hello"}], {"type": "object"})


class StoppableModel:
    """A scripted model that waits like a slow local model, and can be stopped the way Ollama can."""
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies = list(replies)
        self.waiting, self.stopping = threading.Event(), threading.Event()
        self.slow = True

    def chat(self, _messages, _schema, on_text=None):
        if self.slow:
            self.waiting.set()
            self.stopping.wait(5)
        if self.stopping.is_set():
            raise Stopped()
        return self.replies.pop(0)

    def stop(self):
        self.stopping.set()

    def allow(self):
        self.stopping.clear()


class GuiStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name, "repo")
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "pkg" / "pricing.py").write_text("def apply_discount(p):\n    return p\n")
        self.model = StoppableModel([])
        self.app = App(self.model)
        self.server, self.token, _ = make_server(self.app)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.model.stop(), self.server.shutdown(), self.server.server_close(), thread.join()))
        self.assertEqual(self.post("/api/open", {"path": str(self.repo)})[0], 200)

    def post(self, path, body=None):
        """POST to the server and return the status and each JSON line of the reply."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", path, body=json.dumps(body or {}),
                           headers={"Content-Type": "application/json", "X-CodeHaven-Token": self.token})
        response = connection.getresponse()
        lines = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        connection.close()
        return response.status, lines

    def start(self, path, body):
        """Begin a streamed request on another thread and wait until the model is working on it."""
        result = []
        thread = threading.Thread(target=lambda: result.append(self.post(path, body)), daemon=True)
        thread.start()
        self.assertTrue(self.model.waiting.wait(5))
        return thread, result

    def test_stop_ends_the_answer_and_nothing_is_remembered(self):
        thread, result = self.start("/api/ask", {"question": "What is in pkg?"})
        self.assertEqual(self.post("/api/stop"), (200, [{"stopped": True}]))
        thread.join(5)
        status, events = result[0]
        self.assertEqual(status, 200)
        self.assertEqual(events[-1], {"type": "stopped", "remembered": 0, "memory": 8})
        self.assertEqual(self.app.session.turns, [])
        self.assertTrue(wait_until(lambda: not self.app.busy))

    def test_the_next_question_is_answered_after_a_stop(self):
        thread, _ = self.start("/api/ask", {"question": "What is in pkg?"})
        self.post("/api/stop")
        thread.join(5)
        self.assertTrue(wait_until(lambda: not self.app.busy))
        self.model.slow = False
        self.model.replies = [dict(action="finish", path="", content="A discount helper.", message="Done")]
        _, events = self.post("/api/ask", {"question": "What is in pkg?"})
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["answer"], "A discount helper.")
        self.assertEqual(len(self.app.session.turns), 1)

    def test_stop_with_nothing_running_does_nothing(self):
        self.assertEqual(self.post("/api/stop"), (200, [{"stopped": False}]))
        self.assertFalse(self.model.stopping.is_set())  # So it cannot end the question asked next.

    def test_stop_needs_the_token(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", "/api/stop", body="{}", headers={"Content-Type": "application/json"})
        self.assertEqual(connection.getresponse().status, 403)
        connection.close()

    def test_stop_ends_flashcards(self):
        thread, result = self.start("/api/flashcards", {"paths": ["pkg/pricing.py"], "count": 3})
        self.assertEqual(self.post("/api/stop")[1], [{"stopped": True}])
        thread.join(5)
        self.assertEqual([event["type"] for event in result[0][1]], ["file", "stopped"])
        self.assertTrue(wait_until(lambda: not self.app.busy))

    def test_the_page_has_a_stop_button(self):
        page = page_text()
        self.assertIn('"/api/stop"', page)
        self.assertIn('t(stopping ? "Stopping…" : "Stop")', page)



class IndexStopTests(unittest.TestCase):
    """Stop also ends indexing a repository or downloading one, and keeps whatever was open before."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.first, self.second = Path(self.temp.name, "first"), Path(self.temp.name, "second")
        for folder in (self.first, self.second):
            folder.mkdir()
            (folder / "app.py").write_text("def main():\n    return 1\n")
        self.folder = Path(self.temp.name, "clones")
        self.app = App(StoppableModel([]), allow_clone=True, clone_folder=self.folder)

    def press_stop_during(self, name):
        """Patch name in offline_agent.gui so that, when the work calls it, Stop is pressed first."""
        import offline_agent.gui as gui
        real = getattr(gui, name)

        def pressed(*args, **kwargs):
            self.assertEqual(self.app.stop(), {"stopped": True})
            return real(*args, **kwargs)
        return unittest.mock.patch(f"offline_agent.gui.{name}", side_effect=pressed)

    def test_a_stopped_index_keeps_the_repository_open_before(self):
        self.app.open(str(self.first))
        with self.press_stop_during("KnowledgeBase"):
            summary = self.app.open(str(self.second))
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["repository"], str(self.first.resolve()))
        self.assertEqual(self.app.workspace.root, self.first.resolve())
        self.assertFalse(self.app.busy)
        # The next index is not stopped by the earlier press.
        self.assertEqual(self.app.open(str(self.second))["repository"], str(self.second.resolve()))

    def test_a_stopped_download_leaves_nothing_behind(self):
        def git(command, **_options):
            Path(command[-1], "app.py").write_text("x = 1\n")
            self.app.stop()
            return unittest.mock.Mock(returncode=None, poll=unittest.mock.Mock(return_value=None))
        with unittest.mock.patch("offline_agent.gui.subprocess.Popen", side_effect=git):
            summary = self.app.clone("https://github.com/octo/hello")
        self.assertEqual((summary["stopped"], summary["repository"]), (True, None))
        self.assertEqual(list((self.folder / "octo").iterdir()), [])  # Neither the clone nor its partial folder.
        self.assertFalse(self.app.busy)

    def test_the_index_stops_between_two_files(self):
        from offline_agent.knowledge import KnowledgeBase
        from offline_agent.workspace import Workspace
        stop = threading.Event()
        stop.set()
        with self.assertRaises(Stopped):
            KnowledgeBase(workspace=Workspace(self.first), cancel=stop)
        self.assertEqual(KnowledgeBase(workspace=Workspace(self.first), cancel=threading.Event()).documents_by_source["repo"], 1)

    def test_the_page_turns_index_and_download_into_stop_buttons(self):
        page = page_text()
        self.assertIn('indexing === "index" ? stopLabel', page)
        self.assertIn('indexing === "clone" ? stopLabel', page)
        self.assertIn("async function stopIndexing()", page)


if __name__ == "__main__":
    unittest.main()
