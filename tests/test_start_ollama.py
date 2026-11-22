"""The script that starts a loopback-only Ollama server with cloud features off."""
import contextlib
import io
import unittest
from unittest.mock import patch

from scripts.start_ollama import existing_ollama, main


class LauncherTests(unittest.TestCase):
    @patch("scripts.start_ollama.http.client.HTTPConnection")
    def test_probe_validates_response_and_closes_connection(self, factory):
        connection = factory.return_value
        response = connection.getresponse.return_value
        for status, body, expected in (
            (200, b'{"models": []}', True),
            (302, b'{"models": []}', False),
            (200, b'not json', False),
            (200, b'[]', False),
            (200, b'{"models": "wrong"}', False),
            (200, b'x' * 64_001, False),
        ):
            with self.subTest(status=status, expected=expected, size=len(body)):
                response.status = status
                response.read.return_value = body
                self.assertEqual(existing_ollama(), expected)
        factory.assert_called_with("127.0.0.1", 11434, timeout=2)
        connection.request.assert_called_with("GET", "/api/tags")
        self.assertEqual(connection.close.call_count, 6)

    @patch("scripts.start_ollama.http.client.HTTPConnection")
    def test_probe_connection_failure_is_handled(self, factory):
        connection = factory.return_value
        connection.request.side_effect = OSError("unavailable")
        self.assertFalse(existing_ollama())
        connection.close.assert_called_once()

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([])

    @patch("scripts.start_ollama.shutil.which", return_value=None)
    @patch("scripts.start_ollama.subprocess.run")
    def test_missing_binary(self, run, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()

    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection")
    @patch("scripts.start_ollama.subprocess.run")
    @patch("scripts.start_ollama.existing_ollama", return_value=True)
    def test_existing_server_exits_successfully_without_starting(self, probe, run, connect, which):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("already running", output.getvalue())
        self.assertIn("has not been verified", output.getvalue())
        run.assert_not_called()

    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection")
    @patch("scripts.start_ollama.subprocess.run")
    @patch("scripts.start_ollama.existing_ollama", return_value=False)
    def test_unrecognized_listener_is_not_reused(self, probe, run, connect, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()

    @patch.dict("os.environ", {"OLLAMA_NO_CLOUD": "0", "OLLAMA_HOST": "0.0.0.0:1234"})
    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection", side_effect=ConnectionRefusedError)
    @patch("scripts.start_ollama.subprocess.run")
    def test_overrides_settings_and_preserves_exit_status(self, run, connect, which):
        run.return_value.returncode = 7
        self.assertEqual(self.invoke(), 7)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/bin/ollama", "serve"])
        self.assertEqual(kwargs["env"]["OLLAMA_NO_CLOUD"], "1")
        self.assertEqual(kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")

    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection", side_effect=PermissionError)
    @patch("scripts.start_ollama.subprocess.run")
    def test_port_check_failure_does_not_start_server(self, run, connect, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()
