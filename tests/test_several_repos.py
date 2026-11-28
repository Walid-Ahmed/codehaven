"""Several repositories opened together: one index across them, and nothing else under their common parent."""
import json
import tempfile
import unittest
from pathlib import Path

from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import combined


class SeveralReposTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        for name, text in {"api/server.py": "def zqxserve():\n    return 'api'\n",
                           "web/src/client.js": "export function zqxfetch() { return 1; }\n",
                           "private/notes.py": "zqxsecret = 1\n", "README.md": "# not opened\n"}.items():
            path = self.parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.ws = combined([self.parent / "api", self.parent / "web"])

    def test_only_the_opened_repositories_are_seen(self):
        self.assertEqual(self.ws.scopes, ("api", "web"))
        self.assertEqual(self.ws.files(), ["api/server.py", "web/src/client.js"])
        for name in ("private/notes.py", "README.md"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ws.read(name)
        self.assertNotIn("zqxsecret", self.ws.search("zqx"))

    def test_one_index_answers_across_them_and_keeps_them_after_a_refresh(self):
        kb = KnowledgeBase(workspace=self.ws)
        sources = {row["source"] for row in json.loads(kb.retrieve("zqxserve zqxfetch"))["results"]}
        self.assertEqual(sources, {"repo/api/server.py", "repo/web/src/client.js"})
        (self.parent / "web" / "src" / "more.js").write_text("export const zqxmore = 2;\n")
        self.assertTrue(kb.refresh())
        self.assertIn("repo/web/src/more.js", kb.retrieve("zqxmore"))
        self.assertNotIn("zqxsecret", kb.retrieve("zqxsecret"))

    def test_one_folder_or_a_parent_of_the_others_is_opened_as_it_is(self):
        self.assertEqual(combined([self.parent / "api"]).scopes, ())
        self.assertEqual(combined([self.parent, self.parent / "api"]).scopes, ())


if __name__ == "__main__":
    unittest.main()
