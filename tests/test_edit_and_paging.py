"""Line-range reads keep large files usable within small local budgets; parts of code resolve to real line ranges."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import action_key, run
from offline_agent.workspace import Workspace, line_range, symbol_lines


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class ScriptedModel:
    """Replays replies and records every message list sent to the model."""
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def chat(self, messages, schema):
        self.calls.append(messages)
        return next(self.replies)


class LineRangeTests(unittest.TestCase):
    def test_parses_ranges_and_treats_other_text_as_whole_file(self):
        self.assertEqual(line_range("120-180"), (120, 180))
        self.assertEqual(line_range(" lines 5 - 9 "), (5, 9))
        self.assertEqual(line_range("40"), (40, None))
        self.assertIsNone(line_range(""))
        self.assertIsNone(line_range("read the whole file"))
        for bad in ("0-4", "9-3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                line_range(bad)

    def test_repeated_lookup_key_distinguishes_ranges_only(self):
        self.assertEqual(action_key(action("read", "a.py", "narration")), action_key(action("read", "a.py")))
        self.assertNotEqual(action_key(action("read", "a.py", "1-50")), action_key(action("read", "a.py", "51-100")))


class PageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)
        (self.root / "big.py").write_text("".join(f"value_{n} = {n}\n" for n in range(1, 1001)))

    def test_small_whole_file_is_unchanged(self):
        (self.root / "small.py").write_text("a = 1\n")
        self.assertEqual(self.ws.page("small.py", "", 4000), ("a = 1\n", 1, 1, 1))

    def test_large_file_is_cut_at_line_boundary_within_limit(self):
        text, start, end, total = self.ws.page("big.py", "", 4000)
        self.assertEqual((start, total), (1, 1000))
        self.assertLessEqual(len(text.encode()), 4000)
        self.assertTrue(text.endswith(f"value_{end} = {end}\n"))

    def test_requested_range_and_out_of_range_errors(self):
        self.assertEqual(self.ws.page("big.py", "500-502", 4000)[0], "value_500 = 500\nvalue_501 = 501\nvalue_502 = 502\n")
        self.assertEqual(self.ws.page("big.py", "999-2000", 4000)[1:], (999, 1000, 1000))
        with self.assertRaisesRegex(ValueError, "1000 lines"):
            self.ws.page("big.py", "1001-1002", 4000)


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)

    def test_large_read_is_paged_with_next_range_and_cited_lines(self):
        (self.root / "big.py").write_text("".join(f"value_{n} = {n}\n" for n in range(1, 1001)))
        model = ScriptedModel([action("read", "big.py"), action("read", "big.py", "990-1000"),
                               action("finish", content="value_1000 is 1000. [big.py:1000]")])
        status = run(model, self.ws, "What is value_1000?", output=lambda _: None)
        self.assertEqual(status, 0)
        first_page = model.calls[1][-1]["content"]
        self.assertRegex(first_page, r"Lines 1-\d+ of 1000; \d+ more lines, request content \"\d+-\d+\"")
        self.assertLess(len(first_page.encode()), 6500)
        self.assertIn("Lines 990-1000 of 1000:\nvalue_990 = 990", model.calls[2][-1]["content"])


class SymbolLineTests(unittest.TestCase):
    SOURCE = ("import os\nfrom pathlib import Path\n\nROUTES = {\n    'a': 1,\n}\n\n\n"
              "class Store:\n    def add(self):\n        pass\n\n    @property\n    def low(self):\n        return 1\n\n\n"
              "@app.get('/health')\nasync def health():\n    return {}\n")

    def test_python_parts_resolve_to_their_full_ranges(self):
        cases = {"imports": (1, 2), "ROUTES": (4, 6), "Store": (9, 15), "Store.low": (13, 15), "low": (13, 15),
                 "def add": (10, 11), "`health()`": (18, 20), "/health": (18, 20)}
        for symbol, expected in cases.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(symbol_lines("store.py", self.SOURCE, symbol), expected)

    def test_unknown_names_resolve_to_nothing(self):
        for symbol in ("main", "Other.low", "/missing", "", "os.path"):
            with self.subTest(symbol=symbol):
                self.assertIsNone(symbol_lines("store.py", self.SOURCE, symbol))

    def test_other_files_match_the_first_line_naming_the_part(self):
        script = "const total = 1;\nfunction subtotal() {}\nfunction total_tax() { return total; }\n"
        self.assertEqual(symbol_lines("cart.js", script, "total"), (1, 1))
        self.assertEqual(symbol_lines("cart.js", script, "subtotal()"), (2, 2))
        self.assertEqual(symbol_lines("broken.py", "def go(:\n    pass\n", "go"), (1, 1))  # Unparsable: text match.
        rows = 'CATALOG = [\n    ("SKU-0001", 1.5),\n    ("SKU-0002", 2.5),\n]\n'
        self.assertEqual(symbol_lines("catalog.py", rows, "SKU-0002"), (3, 3))  # Not a definition: text match.
