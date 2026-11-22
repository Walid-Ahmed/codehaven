"""Questions whose answer is one line in the project: quoted settings and dotted members are searched up front."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.tasks import phrases, search_named
from offline_agent.workspace import Workspace


class PhraseTests(unittest.TestCase):
    """phrases() picks out the literal text worth searching for in a question."""

    def test_quoted_settings_are_searched_as_written(self):
        self.assertEqual(phrases("What is the default value of the 'trust proxy' setting?"), ["trust proxy"])
        self.assertEqual(phrases('Which value does it set for "subdomain offset" and for `etag`?'), ["subdomain offset", "etag"])

    def test_dotted_members_are_searched_as_assignments_and_keys(self):
        self.assertEqual(phrases("Which package provides express.static?"), [".static =", "'static'"])
        self.assertEqual(phrases("How does req.secure decide its value?"), [".secure =", "'secure'"])

    def test_file_names_code_names_and_plain_questions_add_nothing(self):
        for question in ("Explain src/app.py", "Explain what api.py does", "Which function calls default_user_agent?",
                         "What does config.max_retries hold?", "How does this project work?", "Don't do it"):
            with self.subTest(question=question):
                self.assertEqual(phrases(question), [])


class SearchNamedTests(unittest.TestCase):
    """search_named shows the lines a question's names and phrases appear on, source files first."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, text in (("lib/application.js", "app.defaultConfiguration = function () {\n  this.set('trust proxy', false);\n};\n"),
                           ("lib/express.js", "exports.static = require('serve-static');\n"),
                           ("test/app.js", "app.set('trust proxy', true);\n"),
                           ("History.md", "  * Add 'trust proxy' setting\n")):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text)
        self.ws = Workspace(self.root)

    def test_a_quoted_setting_is_shown_from_the_source_line_that_sets_it(self):
        shown, evidence = search_named(self.ws, "What is the default value of the 'trust proxy' setting?")
        self.assertEqual(shown, ["- lib/application.js:2 (top level): this.set('trust proxy', false);"])
        self.assertEqual([(row["path"], row["line"]) for row in evidence], [("lib/application.js", 2)])

    def test_a_dotted_member_finds_its_export(self):
        shown, _ = search_named(self.ws, "Which package provides express.static?")
        self.assertIn("- lib/express.js:1 (top level): exports.static = require('serve-static');", shown)

    def test_code_name_matches_list_source_files_before_tests(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_rate.py").write_text("assert SERVICE_RATE\n")
        (self.root / "fees.py").write_text("SERVICE_RATE = 0.05\n")
        shown, _ = search_named(self.ws, "What is SERVICE_RATE?")
        self.assertTrue(shown[0].startswith("- fees.py:1"), shown)
