"""Study aids: the structure map is exact static analysis, and flashcards cite lines that were looked up."""
import tempfile
import unittest
from pathlib import Path

from offline_agent import study
from offline_agent.knowledge import KnowledgeBase
from offline_agent.model import InvalidReply
from offline_agent.workspace import Workspace

PRICING = ("RATE = 0.1\n\n\ndef apply_discount(price):\n    return price * (1 - RATE)\n\n\n"
           "class Cart:\n    def __init__(self):\n        self.items = []\n\n"
           "    def total(self):\n        return apply_discount(sum(self.items))\n")
CHECKOUT = "from pkg.pricing import Cart\n\n\ndef checkout():\n    return Cart().total()\n"


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, schema, on_text=None):
        self.calls.append((messages, schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def card(question, answer, symbol):
    return {"question": question, "answer": answer, "symbol": symbol}


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "pkg").mkdir()
        (root / "pkg" / "pricing.py").write_text(PRICING)
        (root / "pkg" / "checkout.py").write_text(CHECKOUT)
        (root / "app.py").write_text("from pkg.checkout import checkout\n\nprint(checkout())\n")
        (root / "main.go").write_text("package main\n\nfunc Add(a int) int {\n\treturn Sub(a)\n}\n\nfunc Sub(a int) int {\n\treturn a\n}\n")
        (root / ".env").write_text("SECRET=1\n")
        self.workspace = Workspace(root)
        self.knowledge = KnowledgeBase(None, workspace=self.workspace, context_budget=4000)

    def test_one_file_maps_its_parts_and_the_calls_between_them(self):
        result = study.structure(self.workspace, self.knowledge, ["pkg/pricing.py"])
        parts = {part["name"]: part for part in result["files"][0]["parts"]}
        self.assertEqual(list(parts), ["apply_discount", "Cart", "Cart.__init__", "Cart.total"])
        self.assertEqual((parts["Cart"]["kind"], parts["Cart"]["start"], parts["Cart"]["end"]), ("class", 8, 13))
        self.assertEqual(parts["Cart.total"]["kind"], "method")
        self.assertEqual(parts["Cart.total"]["calls"], ["apply_discount"])
        self.assertEqual(parts["Cart"]["calls"], [])  # A class's calls belong to its methods.

    def test_a_group_of_files_maps_imports_and_names_neighbours(self):
        result = study.structure(self.workspace, self.knowledge, ["pkg/pricing.py", "repo/pkg/checkout.py"])
        self.assertEqual([file["path"] for file in result["files"]], ["pkg/pricing.py", "pkg/checkout.py"])
        self.assertIn({"from": "pkg/checkout.py", "to": "pkg/pricing.py", "names": ["Cart"]}, result["links"])
        self.assertIn({"from": "app.py", "to": "pkg/checkout.py", "names": ["checkout"]}, result["links"])
        self.assertEqual(result["neighbours"], ["app.py"])
        self.assertEqual(result["files"][1]["parts"][0]["calls"], [])  # Cart is defined in another file.

    def test_other_languages_fall_back_to_definition_lines(self):
        parts = study.structure(self.workspace, self.knowledge, ["main.go"])["files"][0]["parts"]
        self.assertEqual([(part["name"], part["start"], part["end"], part["calls"]) for part in parts],
                         [("Add", 3, 6, ["Sub"]), ("Sub", 7, 9, [])])

    def test_choices_follow_the_workspace_rules_and_limits(self):
        for paths, message in (([], "at least one"), ([".env"], "unavailable"), (["../x.py"], "relative path"),
                               (["missing.py"], "missing.py"), ("app.py", "list of paths"),
                               ([f"f{n}.py" for n in range(study.MAP_FILES + 1)], "at most")):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, message):
                study.structure(self.workspace, self.knowledge, paths)

    def test_flashcards_cite_looked_up_lines_and_drop_bad_cards(self):
        model = ScriptedModel([{"cards": [card("What does apply_discount return?", "The price less RATE.", "apply_discount"),
                                          card("what does apply_discount return?", "A duplicate.", "apply_discount"),
                                          card("How is a cart totalled?", "It discounts the sum of its items.", "Cart.total"),
                                          card("Why is it fast?", "A guess about nothing in the file.", "turbo_mode"),
                                          card("", "No question.", "Cart"), {"question": 1}]}])
        events = list(study.flashcards(model, self.workspace, ["pkg/pricing.py"], 5))
        self.assertEqual([event["type"] for event in events], ["file", "cards", "done"])
        cards = events[1]["cards"]
        self.assertEqual([(c["symbol"], c["start"], c["end"]) for c in cards],
                         [("apply_discount", 4, 5), ("Cart.total", 12, 13), ("turbo_mode", None, None)])
        self.assertEqual(events[-1]["cards"], 3)
        messages, schema = model.calls[0]
        self.assertIn("untrusted data", messages[0]["content"])
        self.assertIn(PRICING, messages[1]["content"])
        self.assertEqual(schema["properties"]["cards"]["maxItems"], 5)
        # Every string the model writes is bounded, so it cannot loop inside one.
        self.assertTrue(all("maxLength" in field for field in schema["properties"]["cards"]["items"]["properties"].values()))

    def test_one_failed_file_does_not_stop_the_others(self):
        model = ScriptedModel([InvalidReply("Model returned an invalid JSON response"),
                               {"cards": [card("What does checkout return?", "The cart's total.", "checkout")]},
                               {"cards": []}])
        events = list(study.flashcards(model, self.workspace, ["pkg/pricing.py", "pkg/checkout.py", "app.py"], 3))
        self.assertEqual([(event["type"], event.get("path")) for event in events],
                         [("file", "pkg/pricing.py"), ("error", "pkg/pricing.py"), ("file", "pkg/checkout.py"),
                          ("cards", "pkg/checkout.py"), ("file", "app.py"), ("error", "app.py"), ("done", None)])
        self.assertEqual(events[-1]["cards"], 1)

    def test_a_long_file_is_covered_from_its_start_and_says_so(self):
        long = "".join(f"def function_{n}():\n    return {n}\n\n\n" for n in range(400))
        (self.workspace.root / "long.py").write_text(long)
        model = ScriptedModel([{"cards": [card("What does function_0 return?", "0.", "function_0")]}])
        events = list(study.flashcards(model, self.workspace, ["long.py"], 3))
        self.assertLess(events[1]["covered"], events[1]["lines"])
        self.assertIn(f"Only lines 1-{events[1]['covered']} of {events[1]['lines']}", model.calls[0][0][1]["content"])
        self.assertLessEqual(len(model.calls[0][0][1]["content"].encode()), 6400)

    def test_card_counts_and_file_counts_are_bounded(self):
        for paths, count in ((["app.py"], 0), (["app.py"], study.CARDS + 1), (["app.py"], True), (["app.py"], "5"),
                             ([f"f{n}.py" for n in range(study.CARD_FILES + 1)], 3)):
            with self.subTest(count=count), self.assertRaises(ValueError):
                next(study.flashcards(ScriptedModel([]), self.workspace, paths, count))


if __name__ == "__main__":
    unittest.main()
