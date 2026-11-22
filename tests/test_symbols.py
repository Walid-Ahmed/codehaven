"""The functions and classes a question names are found in the parsed code and shown to the model first."""
import tempfile
import unittest
from pathlib import Path

from offline_agent import symbols
from offline_agent.agent import Session
from offline_agent.answers import with_points
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace

FILES = {
    "shop/api.py": "from shop.session import Session\n\n\ndef get(url):\n    \"\"\"Fetch a page.\"\"\"\n    return Session().get(url)\n",
    "shop/session.py": "LIMIT = 30\n\n\nclass Session:\n    \"\"\"A session.\"\"\"\n\n    def __init__(self):\n        self.max_redirects = LIMIT\n\n"
                       "    def get(self, url):\n        return url\n\n    def app_dir(self):\n        return 'APPDATA'\n\n"
                       + "".join(f"    def filler_{n}(self):\n        return {n}\n\n" for n in range(120))
                       + "    def mount(self, prefix):\n        self.prefix = prefix\n",
    "web/request.js": "defineGetter(req, 'xhr', function xhr() {\n  return true;\n});\n\nres.send = function send(body) {\n  return body;\n};\n",
    "tests/test_api.py": "def get():\n    return 1\n",
}


class Finisher:
    """A model that answers at once, and records the prompt it was given."""
    prompt_budget = 12_000

    def __init__(self):
        self.prompts = []

    def chat(self, messages, _schema, on_text=None):
        self.prompts.append(messages)
        return {"action": "finish", "path": "", "content": "Done.", "message": "Done", "points": []}


class SymbolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for name, text in FILES.items():
            target = Path(self.temp.name, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.workspace = Workspace(self.temp.name)
        self.knowledge = KnowledgeBase(None, workspace=self.workspace, context_budget=4000)

    def names(self, task, **options):
        return [(path, name) for path, name, _, _ in symbols.find(task, self.knowledge, **options)]

    def test_only_words_that_look_like_code_are_looked_up(self):
        self.assertEqual(symbols.words("How does `get` use Session.get and app_dir? Explain Session."),
                         [("get", True), ("Session.get", True), ("app_dir", True), ("Explain", False), ("Session", False)])
        self.assertEqual(self.names("What does the get request return for a mount?"), [])  # Plain English words.
        self.assertEqual(self.names("Which does Explain mean?"), [])  # A capitalised word that is no class.

    def test_names_are_matched_to_the_right_definition(self):
        self.assertEqual(self.names("What does Session.get return?"), [("shop/session.py", "Session.get")])
        self.assertEqual(self.names("What does shop.get do?"), [("shop/api.py", "get")])  # The module's function, not the method.
        self.assertEqual(self.names("What does `get` do?"), [("shop/api.py", "get")])  # Code before its tests.
        self.assertEqual(self.names("How many redirects does a Session follow?"), [("shop/session.py", "Session")])
        self.assertEqual(self.names("What does req.xhr check, and what does res.send do?"),
                         [("web/request.js", "xhr"), ("web/request.js", "res.send")])
        # A method and then its own class: the class adds nothing, so only the method is kept.
        self.assertEqual(self.names("What does Session.app_dir read, and what is Session?"), [("shop/session.py", "Session.app_dir")])
        self.assertEqual(self.names("What does `get` do?", paths={"shop/session.py"}), [("shop/session.py", "Session.get")])

    def test_a_short_definition_is_shown_whole_with_its_real_line_numbers(self):
        found = symbols.find("What does Session.app_dir read?", self.knowledge)
        text, evidence, log = symbols.shown(self.workspace, found)
        self.assertIn("  13|     def app_dir(self):\n  14|         return 'APPDATA'", text)
        self.assertEqual([(row["path"], row["line"], row["end_line"]) for row in evidence], [("shop/session.py", 13, 14)])
        self.assertEqual(log, ["Preloaded the definition of Session.app_dir (shop/session.py:13-14) because the task names it."])

    def test_a_long_class_is_shown_around_the_words_of_the_question(self):
        task = "Which prefix does a new Session mount?"
        found = symbols.find(task, self.knowledge)
        text, evidence, log = symbols.shown(self.workspace, found, task)
        self.assertLessEqual(len(text.encode()), symbols.SYMBOL_BYTES + 400)
        self.assertIn("class Session:", text)
        self.assertIn("def mount(self, prefix):", text)  # Near the end of the class, kept because the task says "mount".
        self.assertIn("too long to show whole", text)
        self.assertNotIn("def filler_60", text)
        self.assertIn("in parts", log[0])
        self.assertGreater(len(evidence), 1)

    def test_the_definition_reaches_the_model_before_its_first_step(self):
        model, log = Finisher(), []
        session = Session(model, self.workspace, knowledge=self.knowledge, output=log.append)
        session.ask("What does Session.app_dir return?")
        shown = "\n".join(message["content"] for message in model.prompts[0])
        self.assertIn("the definition of Session.app_dir, lines 13-14", shown)
        self.assertIn("  14|         return 'APPDATA'", shown)
        self.assertIn("Preloaded the definition of Session.app_dir (shop/session.py:13-14) because the task names it.", log)
        self.assertIn(("shop/session.py", 13), {(row["path"], row["line"]) for row in session.evidence})

    def test_the_values_of_constants_a_definition_uses_are_shown_with_it(self):
        (Path(self.temp.name) / "shop" / "codes.py").write_text("REDIRECTS = (\n    301,\n    302,\n)\n")
        (Path(self.temp.name) / "shop" / "check.py").write_text(
            "from shop.codes import REDIRECTS\nfrom shop.session import LIMIT\n\n\ndef is_redirect(code, hops):\n"
            "    return code in REDIRECTS and hops < LIMIT\n")
        self.knowledge.refresh()
        task = "What does is_redirect check?"
        text, evidence, _ = symbols.shown(self.workspace, symbols.find(task, self.knowledge), task, knowledge=self.knowledge)
        self.assertIn("Constants these lines use, and where each is set:", text)
        self.assertIn("shop/codes.py:1| REDIRECTS = (\nshop/codes.py:2|     301,\nshop/codes.py:3|     302,\nshop/codes.py:4| )", text)
        self.assertIn("shop/session.py:1| LIMIT = 30", text)  # Set in another file: found through the index.
        self.assertIn(("shop/codes.py", 1, 4), {(row["path"], row["line"], row["end_line"]) for row in evidence})

    def test_a_point_that_names_the_right_thing_in_the_wrong_place_cites_a_line_that_was_shown(self):
        # The model was shown where LIMIT is set (session.py:1) and where it is used (session.py:7-8), and nothing of api.py.
        evidence = [{"path": "shop/session.py", "line": 1, "end_line": 1, "text": "LIMIT = 30"},
                    {"path": "shop/session.py", "line": 7, "end_line": 8, "text": "    def __init__(self):\n        self.max_redirects = LIMIT"}]

        def cited(path, symbol):
            reply = {"action": "finish", "path": "", "content": "Thirty.", "message": "",
                     "points": [{"path": path, "symbol": symbol, "text": "the limit"}]}
            return with_points(reply, self.workspace, evidence)[0]["content"].splitlines()[1]
        # Named in a file that does not contain it: the shown line that sets it is cited instead of "not found".
        self.assertEqual(cited("shop/api.py", "LIMIT"), "- shop/session.py:1 `LIMIT`: the limit")
        # Found in the named file, on lines that were shown: left as looked up.
        self.assertEqual(cited("shop/session.py", "max_redirects"), "- shop/session.py:8 `max_redirects`: the limit")
        # Found only on lines that were never shown, with no shown line naming it: left alone, to be flagged.
        self.assertEqual(cited("shop/session.py", "mount"), "- shop/session.py:376-377 `mount`: the limit")

    def test_a_task_with_no_code_names_gets_no_extra_message(self):
        model = Finisher()
        Session(model, self.workspace, knowledge=self.knowledge, output=lambda _line: None).ask("What is this project for?")
        self.assertNotIn("definitions the task names", "\n".join(message["content"] for message in model.prompts[0]))


if __name__ == "__main__":
    unittest.main()
