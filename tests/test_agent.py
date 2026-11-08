import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from offline_agent.agent import run
from offline_agent.model import MODEL, Ollama
from offline_agent.workspace import Workspace


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, schema):
        return next(self.replies)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)

    def test_reject_escapes_and_hidden_paths(self):
        for path in ["../outside", "/tmp/outside", ".git/config", ".env", "a/../../x", ""]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.ws.path(path)

    def test_symlinks_blocked(self):
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.ws.read("link/example.py")
        self.assertEqual(self.ws.files(), [])

    def test_size_and_binary_limits(self):
        (self.root / "big").write_bytes(b"x" * 64001)
        (self.root / "binary").write_bytes(b"\0")
        for name in ["big", "binary"]:
            with self.assertRaises(ValueError):
                self.ws.read(name)

    def test_codehaven_cannot_write_files(self):
        # Read-only by design: a model asking to write or edit gets a correction, and no file appears.
        self.assertFalse(hasattr(self.ws, "write") or hasattr(self.ws, "propose"))
        output = []
        model = FakeModel([action("write", "hello.py", "print('hello')\n"), action("edit", "hello.py", "x"), action("finish")])
        with self.assertRaisesRegex(ValueError, "unknown action 'edit'"):
            run(model, self.ws, "Create hello.py", output=output.append)
        self.assertFalse((self.root / "hello.py").exists())
        self.assertTrue(any("unknown action 'write'" in line for line in output))

    def test_step_limit_and_invalid_action(self):
        self.assertEqual(run(FakeModel([action("read", "missing")]), self.ws,
                             "task", max_steps=1, output=lambda _: None), 2)
        # One invalid reply is corrected; a second invalid reply in a row stops the task.
        with self.assertRaisesRegex(ValueError, "twice in a row"):
            run(FakeModel([{"action": "shell"}, {"action": "shell"}]), self.ws, "task", output=lambda _: None)


class TransportTests(unittest.TestCase):
    def test_remote_endpoints_rejected(self):
        for endpoint in ["https://example.com", "http://localhost:11434", "http://10.0.0.1",
                         "http://127.0.0.1/evil", "http://user@127.0.0.1", "http://127.0.0.1:bad"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                Ollama(endpoint=endpoint)
        with self.assertRaises(ValueError):
            Ollama("model:cloud")

    def test_any_local_model_name_but_never_a_cloud_one(self):
        self.assertEqual(Ollama().model, "qwen2.5-coder:7b")  # The tested default.
        self.assertEqual(Ollama("qwen2.5:14b").model, "qwen2.5:14b")
        for name in ["gpt-oss:120b-cloud", "qwen3-coder:cloud", "", "bad name", "x" * 201]:
            with self.subTest(model=name), self.assertRaises(ValueError):
                Ollama(name)

    def test_model_list_shows_local_chat_models_default_first(self):
        tags = {"models": [
            {"name": "qwen2.5:14b", "capabilities": ["completion"], "details": {"parameter_size": "14.8B", "family": "qwen2"}},
            {"name": "embeddinggemma:latest", "capabilities": ["embedding"], "details": {}},
            {"name": "gpt-oss:120b-cloud", "capabilities": ["completion"], "details": {}},
            {"name": "remote:latest", "remote_host": "https://ollama.com", "details": {}},
            {"name": "qwen2.5-coder:7b", "capabilities": ["completion"], "details": {"parameter_size": "7.6B"}, "size": 4},
            {"name": "llama3:latest", "details": {}}]}
        with patch.object(Ollama, "request", return_value=tags):
            names = [m["name"] for m in Ollama().local_models()]
        self.assertEqual(names, ["qwen2.5-coder:7b", "llama3:latest", "qwen2.5:14b"])

    def test_switching_to_a_model_that_cannot_chat_keeps_the_current_one(self):
        client = Ollama()
        tags = {"models": [{"name": "embeddinggemma:latest"}, {"name": "qwen2.5:14b"}]}
        def request(path, payload=None):
            if path == "/api/tags":
                return tags
            return {"capabilities": ["embedding"]} if payload["model"].startswith("embedding") else {"capabilities": ["completion"]}
        with patch.object(client, "request", side_effect=request):
            with self.assertRaisesRegex(ValueError, "cannot chat"):
                client.use("embeddinggemma:latest")
            self.assertEqual(client.model, "qwen2.5-coder:7b")
            client.use("qwen2.5:14b")
            self.assertEqual(client.model, "qwen2.5:14b")
            with self.assertRaises(ValueError):
                client.use("missing:latest")
            self.assertEqual(client.model, "qwen2.5:14b")

    def test_real_http_contract(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_payload({"models": [{"name": MODEL}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, payload))
                if self.path == "/api/show":
                    self.send_payload({})
                else:
                    self.send_payload({"message": {"content": json.dumps(action("finish"))}})

            def send_payload(self, value):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                client = Ollama(endpoint=f"http://127.0.0.1:{server.server_port}")
                client.check_model()
                self.assertEqual(client.chat([], {} )["action"], "finish")
                self.assertTrue(calls[-1][1]["stream"])
                self.assertTrue(all(payload["model"] == MODEL for _, payload in calls))
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
