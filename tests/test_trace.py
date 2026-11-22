"""Trace a function: the calls below it and its callers come from parsing; the model only describes each step."""
import tempfile
import unittest
from pathlib import Path

from offline_agent import trace
from offline_agent.agent import Session
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace

FILES = {
    "shop/pricing.py": "RATE = 0.1\n\n\ndef with_tax(total):\n    return total * (1 + RATE)\n\n\n"
                       "def checkout_total(items):\n    # with_fee(total) was removed\n    total = sum(items)\n"
                       "    print(\"with_fee(\")\n    return with_tax(round_up(total))\n",
    "shop/rounding.py": "def round_up(value):\n    return int(value) + 1\n\n\ndef with_fee(value):\n    return value + 1\n",
    "shop/store.py": "from shop.pricing import checkout_total\n\n\nclass Store:\n    def __init__(self):\n        self.items = []\n\n"
                     "    def add(self, price):\n        self.items.append(price)\n        self.flush()\n\n"
                     "    def flush(self):\n        pass\n\n    def total(self):\n        return checkout_total(self.items)\n\n\n"
                     "class Log:\n    def flush(self):\n        pass\n\n\ndef close(file):\n    file.flush()\n",
    "web/cart.js": "exports.total = function (items) {\n  return sum(items);\n};\n\nfunction sum(items) {\n  return items.length;\n}\n",
    "tests/test_store.py": "from shop.store import Store\n\n\ndef test_total():\n    assert Store().total() == 1\n",
}


class Describer:
    """Stands in for the local model: one sentence per function, and a record of what it was shown."""
    prompt_budget = 12_000

    def __init__(self, fail=False):
        self.shown, self.fail = [], fail

    def chat(self, messages, _schema, on_text=None):
        self.shown.append(messages[-1]["content"])
        if self.fail:
            raise ValueError("no reply")
        return {"sentence": "Does one thing."}


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for name, text in FILES.items():
            target = Path(self.temp.name, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.workspace = Workspace(self.temp.name)
        self.knowledge = KnowledgeBase(None, workspace=self.workspace, context_budget=4000)

    def run_trace(self, name, model=None, **options):
        return trace.trace(model or Describer(), self.workspace, self.knowledge, name, output=lambda _line: None, **options)

    def test_requests_that_ask_for_a_trace(self):
        for task, name in [("Trace checkout_total", "checkout_total"), ("trace the call path of `Store.add`", "Store.add"),
                           ("What happens when checkout_total is called?", "checkout_total"),
                           ("What does checkout_total call?", "checkout_total"), ("Show the call chain of total()", "total"),
                           ("تتبع الدالة checkout_total", "checkout_total")]:
            self.assertEqual(trace.requested(task), name, task)
        for task in ["Explain this project", "What happens when Store.add is called with a negative price?",
                     "trace the bug in store.py", "Which function calls checkout_total?", "Trace the function"]:
            self.assertIsNone(trace.requested(task), task)

    def test_calls_are_followed_across_files_in_call_order(self):
        found = self.run_trace("Store.total")
        lines = found["text"].splitlines()
        self.assertEqual([line.split(": Does")[0] for line in lines[1:5]],
                         ["1. shop/store.py:15-16 Store.total",
                          "  2. shop/pricing.py:8-12 checkout_total (called by Store.total)",
                          "    3. shop/pricing.py:4-5 with_tax (called by checkout_total)",
                          "    4. shop/rounding.py:1-2 round_up (called by checkout_total)"])
        self.assertEqual(found["checked"], 4)
        self.assertNotIn("with_fee", found["text"])  # Named only in a comment and in a string.
        self.assertEqual([(row["path"], row["line"], row["end_line"]) for row in found["evidence"][:2]],
                         [("shop/store.py", 15, 16), ("shop/pricing.py", 8, 12)])
        self.assertIn("Called from: tests/test_store.py:5 in test_total", found["text"])

    def test_a_method_call_means_this_class_and_a_call_on_another_object_is_not_guessed(self):
        self.assertIn("2. shop/store.py:12-13 Store.flush (called by Store.add)", self.run_trace("Store.add")["text"])
        self.assertIn("calls no other function", self.run_trace("close")["text"])  # file.flush() could be anything.

    def test_commonjs_functions_are_traced_and_languages_are_kept_apart(self):
        found = self.run_trace("total")["text"]
        self.assertIn("1. web/cart.js:1-3 total", found)
        self.assertIn("  2. web/cart.js:5-7 sum (called by total)", found)
        self.assertNotIn("Called from", found)  # checkout_total( and Python's Store.total() are not calls to it.
        self.assertIn("Called from: web/cart.js:2 in total", self.run_trace("sum")["text"])

    def test_a_name_shared_by_several_functions_says_so(self):
        found = self.run_trace("flush")["text"]
        self.assertIn("1. shop/store.py:12-13 Store.flush", found)
        self.assertIn("Other functions share this name: shop/store.py:20", found)

    def test_an_unknown_name_and_an_unusable_reply_are_handled(self):
        self.assertIn("found no function or method named nope", self.run_trace("nope")["text"])
        self.assertIn("(no description", self.run_trace("with_tax", Describer(fail=True))["text"])

    def test_the_model_sees_one_function_at_a_time(self):
        model = Describer()
        self.run_trace("checkout_total", model)
        self.assertEqual(len(model.shown), 3)
        self.assertIn("def checkout_total", model.shown[0])
        self.assertNotIn("def with_tax", model.shown[0])

    def test_a_session_routes_the_request_and_cites_the_lines(self):
        log = []
        session = Session(Describer(), self.workspace, knowledge=self.knowledge, output=log.append)
        self.assertEqual(session.ask("Trace checkout_total"), 0)
        self.assertIn("shop/pricing.py:8-12 checkout_total", session.last_state["final_text"])
        self.assertEqual(session.last_state["steps"], 3)
        self.assertIn(("shop/pricing.py", 8), {(row["path"], row["line"]) for row in session.evidence})


if __name__ == "__main__":
    unittest.main()
