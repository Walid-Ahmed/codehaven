"""Start-up help: what is missing and how to get it, and starting a local Ollama server for one run."""
import contextlib
import io
import unittest
from unittest.mock import patch

from offline_agent import doctor
from offline_agent.cli import main
from offline_agent.gui import page_text


class FakeServer:
    """Stands in for the Ollama client: reachable or not, with a list of installed models."""
    endpoint, model = "http://127.0.0.1:11434", "qwen2.5-coder:7b"

    def __init__(self, up=True, models=()):
        self.up, self.models = up, list(models)

    def request(self, _path):
        if not self.up:
            raise ValueError("Local Ollama request failed")
        return {"models": [{"name": name} for name in self.models]}

    def local_models(self):
        return [{"name": name} for name in self.models]


class Process:
    """A started server that is still running until terminated."""
    def __init__(self):
        self.returncode, self.stopped = None, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stopped, self.returncode = True, 0

    def wait(self, _seconds=None):
        return 0


class DoctorTests(unittest.TestCase):
    def test_each_missing_piece_names_the_command_that_fixes_it(self):
        with patch("offline_agent.doctor.shutil.which", return_value=None):
            self.assertIn("https://ollama.com/", doctor.missing_server(FakeServer(up=False))[0])
        with patch("offline_agent.doctor.shutil.which", return_value="/bin/ollama"):
            self.assertIn("--start-ollama", "\n".join(doctor.missing_server(FakeServer(up=False))))
            lines = doctor.missing_server(FakeServer(models=["qwen2.5:14b", "llama3:latest"]))
            self.assertIn("ollama pull qwen2.5-coder:7b", lines[1])
            self.assertIn("--model qwen2.5:14b (also installed: llama3:latest)", lines[2])

    def test_optional_tools_are_reported_found_or_with_an_install_command(self):
        with patch("offline_agent.doctor.shutil.which", return_value=None), \
                patch("offline_agent.doctor.importlib.util.find_spec", return_value=None):
            text = "\n".join(doctor.extras())
        self.assertIn("brew install tesseract", text)
        self.assertIn("pip install pypdf", text)
        self.assertIn("Isolated tests: off", text)
        with patch("offline_agent.doctor.shutil.which", return_value="/bin/found"):
            self.assertIn("Tesseract found", "\n".join(doctor.extras("image:1")))

    def test_a_running_server_is_used_as_it_is(self):
        with patch("offline_agent.doctor.subprocess.Popen") as popen:
            self.assertIsNone(doctor.start_ollama(FakeServer(), output=lambda _line: None))
        popen.assert_not_called()

    def test_a_server_is_started_on_loopback_with_cloud_off_and_stopped_again(self):
        server, process = FakeServer(up=False), Process()

        def start(command, env, **_options):
            self.assertEqual(command, ["/bin/ollama", "serve"])
            self.assertEqual((env["OLLAMA_NO_CLOUD"], env["OLLAMA_HOST"]), ("1", "127.0.0.1:11434"))
            server.up = True
            return process
        with patch("offline_agent.doctor.shutil.which", return_value="/bin/ollama"), \
                patch("offline_agent.doctor.subprocess.Popen", side_effect=start):
            self.assertIs(doctor.start_ollama(server, output=lambda _line: None), process)
        doctor.stop_ollama(process)
        self.assertTrue(process.stopped)
        doctor.stop_ollama(None)  # Nothing was started: nothing to stop.

    def test_starting_fails_clearly_without_ollama_or_when_it_exits(self):
        with patch("offline_agent.doctor.shutil.which", return_value=None), self.assertRaisesRegex(ValueError, "not installed"):
            doctor.start_ollama(FakeServer(up=False))
        dead = Process()
        dead.returncode = 1
        with patch("offline_agent.doctor.shutil.which", return_value="/bin/ollama"), \
                patch("offline_agent.doctor.subprocess.Popen", return_value=dead), \
                self.assertRaisesRegex(ValueError, "stopped as soon as it started"):
            doctor.start_ollama(FakeServer(up=False))

    def test_the_command_explains_an_unreachable_server(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), patch("offline_agent.doctor.shutil.which", return_value="/bin/ollama"):
            self.assertEqual(main(["--doctor", "--endpoint", "http://127.0.0.1:9"]), 1)
        self.assertIn("no server answers at http://127.0.0.1:9", errors.getvalue())
        self.assertIn("--start-ollama", errors.getvalue())

    def test_the_page_can_save_a_chat_and_flashcards(self):
        page = page_text()
        for needed in ('id="save-chat"', 'id="cards-save"', "function chatMarkdown()", '"text/tab-separated-values"'):
            self.assertIn(needed, page)


if __name__ == "__main__":
    unittest.main()
