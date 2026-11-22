"""Correct answers should carry checkable citations: lines are found from retained evidence, and named code is cited."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run
from offline_agent.answers import MENTION_HEADING, VALUE_HEADING, check_values, cite_mentions, literal_values, observed_line, row_text
from offline_agent.evidence import check_answer, retain
from offline_agent.workspace import Workspace


def action(kind, path="", content="", message="Working"):
    """A model reply as the agent receives it."""
    return dict(action=kind, path=path, content=content, message=message)


class ScriptedModel:
    """Replays replies in order."""
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, _messages, _schema, on_text=None):
        return self.replies.pop(0)


class CitationTests(unittest.TestCase):
    """Citations come from lines the model saw, read back from the files when evidence keeps only a snippet."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "lib").mkdir()
        (self.root / "lib" / "request.js").write_text(
            "var req = exports = module.exports = {};\n\n"
            "defineGetter(req, 'secure', function secure(){\n  return this.protocol === 'https';\n});\n")
        (self.root / "pricing.py").write_text("def apply_discount(p):\n    return p\n\n\ndef checkout_total(p):\n"
                                              "    return apply_discount(p)\n")
        self.ws = Workspace(self.root)

    def test_retained_rows_keep_no_text_so_lines_are_read_back(self):
        rows, _ = retain([], [{"path": "lib/request.js", "line": 1, "end_line": 5, "text": "x" * 500}])
        self.assertNotIn("text", rows[0])
        self.assertIn("function secure", row_text(rows[0], self.ws))
        # The point that was always missing in real runs: a dotted name found on a shown line by its last part.
        self.assertEqual(observed_line("req.secure", "lib/request.js", rows, self.ws), ("lib/request.js", 3))
        self.assertIsNone(observed_line("req.secure", "lib/request.js", rows))  # Without the workspace, as before.

    def test_named_code_in_an_uncited_answer_is_cited_from_shown_lines(self):
        rows, _ = retain([], [{"path": "pricing.py", "line": 1, "end_line": 6}])
        answer = "checkout_total calls apply_discount, then returns the result."
        self.assertEqual(cite_mentions(answer, self.ws, rows),
                         ["- pricing.py:5 `checkout_total`", "- pricing.py:1 `apply_discount`"])
        self.assertEqual(cite_mentions("It is `missing_name`, in `pricing.py`.", self.ws, rows), [])

    def test_an_uncited_correct_answer_is_verified_with_added_citations(self):
        model = ScriptedModel([action("read", "pricing.py"), action("finish", content="checkout_total calls it.")])
        output = []
        self.assertEqual(run(model, self.ws, "Which function calls apply_discount?", output=output.append), 0)
        printed = "\n".join(output)
        self.assertIn(MENTION_HEADING + "\n- pricing.py:5 `checkout_total`", printed)
        self.assertIn("CodeHaven added 1 citation(s)", printed)
        self.assertNotIn("Answer is unverified", printed)

    def test_an_answer_naming_nothing_that_was_shown_stays_unverified(self):
        model = ScriptedModel([action("finish", content="It is handled somewhere else entirely.")])
        output = []
        self.assertEqual(run(model, self.ws, "Which function calls apply_discount?", output=output.append), 2)
        self.assertIn("Answer is unverified; check the cited evidence before relying on it.", output)

    def test_citations_the_model_made_up_are_still_flagged(self):
        rows, _ = retain([], [{"path": "pricing.py", "line": 1, "end_line": 2}])
        self.assertEqual(check_answer("It is here. [pricing.py:5-6]", rows),
                         ["Citation not supported by retained evidence: pricing.py:5-6.",
                          "Repository claims require path:line citations supported by observed evidence."])


class ValueCheckTests(unittest.TestCase):
    """Each literal value an answer reports is found in the code: on the cited lines, or on a line that gets cited."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "adapters.py").write_text("DEFAULT_POOLSIZE = 10\nDEFAULT_RETRIES = 0\n\n\n\n\n\n\n"
                                               "class HTTPAdapter:\n    def __init__(self, pool_connections=DEFAULT_POOLSIZE):\n"
                                               "        self.agent = 'python-requests'\n")
        self.ws = Workspace(self.root)
        self.shown, _ = retain([], [{"path": "adapters.py", "line": 1, "end_line": 11}])

    def test_values_are_quoted_strings_and_numbers_but_not_line_numbers_paths_or_the_question(self):
        answer = "It is 10, see lines 9-10 and at 42, in 'adapters.py', named 'python-requests', not 3 or 0.5.1."
        self.assertEqual(literal_values(answer, "Is the size 10 or 20?"),
                         [("text", "python-requests"), ("number", "0.5.1")])

    def test_a_value_on_or_near_a_cited_line_needs_nothing(self):
        self.assertEqual(check_values("The pool size is 10. [adapters.py:2]", "", self.ws, self.shown), ([], []))  # Line 1 is in reach.

    def test_a_value_shown_elsewhere_gets_a_citation_of_its_line(self):
        answer = "HTTPAdapter sends 'python-requests' with a pool of 10. [adapters.py:9]"
        self.assertEqual(check_values(answer, "", self.ws, self.shown), (["- adapters.py:1 `10`"], []))

    def test_a_value_on_no_shown_line_is_invented(self):
        self.assertEqual(check_values("The pool size is 100. [adapters.py:1]", "", self.ws, self.shown), ([], ["100"]))

    def test_the_agent_cites_a_shown_value_and_flags_an_invented_one(self):
        model = ScriptedModel([action("read", "adapters.py"),
                               action("finish", content="The default pool size is 10. [adapters.py:9]")])
        output = []
        self.assertEqual(run(model, self.ws, "What is HTTPAdapter's default pool size?", output=output.append), 0)
        self.assertIn(VALUE_HEADING + "\n- adapters.py:1 `10`", "\n".join(output))
        model = ScriptedModel([action("read", "adapters.py"),
                               action("finish", content="The default pool size is 100. [adapters.py:1]")])
        output = []
        self.assertEqual(run(model, self.ws, "What is HTTPAdapter's default pool size?", output=output.append), 2)
        self.assertIn("Answer check: The answer reports '100', which is on no line the model was shown.", output)

