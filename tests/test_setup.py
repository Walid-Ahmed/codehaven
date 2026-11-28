"""--setup: a model suggested for the computer's memory, and a download only after a yes."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from offline_agent import doctor
from offline_agent.model import MODEL


class Server:
    def __init__(self, installed):
        self.model, self.installed = MODEL, list(installed)

    def request(self, path, payload=None):
        return {"models": [{"name": name} for name in self.installed]}

    def local_models(self):
        return [{"name": name} for name in self.installed]


class SetupTests(unittest.TestCase):
    def test_the_suggestion_follows_memory(self):
        self.assertEqual(doctor.suggested_model(8)[0], "qwen2.5-coder:3b")
        self.assertEqual(doctor.suggested_model(16)[0], MODEL)
        self.assertEqual(doctor.suggested_model(64)[0], "qwen2.5:14b")

    @patch("offline_agent.doctor.shutil.which", return_value="/usr/local/bin/ollama")
    @patch("offline_agent.doctor.suggested_model", return_value=(MODEL, "the tested default"))
    def test_nothing_is_downloaded_without_a_yes(self, _suggested, _which):
        pulls, lines = [], []
        code = doctor.setup(Server([]), ask=lambda _: "n", output=lines.append, run=lambda *a, **k: pulls.append(a))
        self.assertEqual((code, pulls), (1, []))
        self.assertIn(f"ollama pull {MODEL}", lines[-1])

    @patch("offline_agent.doctor.shutil.which", return_value="/usr/local/bin/ollama")
    @patch("offline_agent.doctor.suggested_model", return_value=(MODEL, "the tested default"))
    def test_a_yes_pulls_the_model_with_cloud_features_off(self, _suggested, _which):
        pulls = []

        def run(command, env):
            pulls.append((command, env["OLLAMA_NO_CLOUD"]))
            return SimpleNamespace(returncode=0)
        code = doctor.setup(Server(["nomic-embed-text:latest"]), ask=lambda _: "y", output=lambda _: None, run=run)
        self.assertEqual(code, 0)
        self.assertEqual(pulls, [(["/usr/local/bin/ollama", "pull", MODEL], "1")])

    @patch("offline_agent.doctor.shutil.which", return_value=None)
    def test_a_missing_ollama_is_explained(self, _which):
        lines = []
        self.assertEqual(doctor.setup(Server([]), output=lines.append), 1)
        self.assertIn("https://ollama.com/", lines[0])


if __name__ == "__main__":
    unittest.main()
