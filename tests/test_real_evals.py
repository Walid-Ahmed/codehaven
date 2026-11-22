"""Real-repository evals: the pinned list, the question sets, and the script that downloads the repositories."""
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent import evaluation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_eval_repos", ROOT / "scripts" / "fetch_eval_repos.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class SuiteTests(unittest.TestCase):
    def test_every_pinned_repository_has_questions_and_a_full_commit(self):
        entries = json.loads((evaluation.REAL / "repos.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(entry["name"] for entry in entries), evaluation.suites())
        for entry in entries:
            self.assertRegex(entry["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(entry["url"].startswith("https://github.com/"))
            self.assertGreaterEqual(len(evaluation.load_tasks(evaluation.REAL / f"{entry['name']}_tasks.json")), 10)

    def test_question_ids_are_unique_across_suites(self):
        ids = [task["id"] for name in evaluation.suites() for task in evaluation.load_tasks(evaluation.REAL / f"{name}_tasks.json")]
        self.assertEqual(len(ids), len(set(ids)))

    def test_an_unknown_suite_and_a_missing_repository_are_explained(self):
        with self.assertRaisesRegex(ValueError, "Unknown suite"):
            evaluation.suite_paths("nonesuch")
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluation, "REAL", Path(folder)):
            Path(folder, "demo_tasks.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, r"fetch_eval_repos\.py demo"):
                evaluation.suite_paths("demo")
            Path(folder, "repos", "demo").mkdir(parents=True)
            tasks, repo, report = evaluation.suite_paths("demo")
            self.assertEqual((tasks.name, repo.name, report.name), ("demo_tasks.json", "demo", "demo.json"))

    def test_the_downloaded_repositories_are_ignored_by_git(self):
        self.assertIn("evals/real/repos/", (ROOT / ".gitignore").read_text())

    def test_the_package_never_runs_the_download_script(self):
        # Only the eval harness names the script, in the message that tells a developer to run it by hand,
        # and the harness has no way to run a program or open a connection itself.
        named = [path.name for path in (ROOT / "offline_agent").glob("*.py") if "fetch_eval_repos" in path.read_text(encoding="utf-8")]
        self.assertEqual(named, ["evaluation.py"])
        code = (ROOT / "offline_agent" / "evaluation.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^(?:import|from) (?:subprocess|urllib|socket|http)\b", code, re.M))


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class FetchTests(unittest.TestCase):
    """A repository made on this computer stands in for GitHub, so nothing here uses the network."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name, "source")
        self.source.mkdir()
        (self.source / "app.py").write_text("print('hello')\n")

        def git(*arguments):
            return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments], cwd=self.source,
                                  check=True, capture_output=True, text=True).stdout.strip()
        git("init", "--quiet")
        git("add", ".")
        git("commit", "--quiet", "-m", "first")
        git("tag", "v1")
        self.entry = {"name": "demo", "url": self.source.as_uri(), "tag": "v1", "commit": git("rev-parse", "HEAD")}
        self.folder = Path(self.temp.name, "repos")

    def test_a_pinned_release_is_downloaded_without_its_history(self):
        self.assertIn("downloaded v1", fetcher.fetch(self.entry, self.folder))
        self.assertEqual((self.folder / "demo" / "app.py").read_text(), "print('hello')\n")
        self.assertFalse((self.folder / "demo" / ".git").exists())
        self.assertIn("already here", fetcher.fetch(self.entry, self.folder))

    def test_a_different_commit_is_refused_and_leaves_nothing_behind(self):
        with self.assertRaisesRegex(ValueError, "not the recorded"):
            fetcher.fetch({**self.entry, "commit": "0" * 40}, self.folder)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_an_unknown_name_is_an_error(self):
        self.assertEqual(fetcher.main(["nonesuch"]), 1)


if __name__ == "__main__":
    unittest.main()
