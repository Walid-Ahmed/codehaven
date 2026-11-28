"""Explain an error: a pasted stack trace's frames are found in the project and shown to the model."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run_state
from offline_agent.tracebacks import frames, locate, requested
from offline_agent.workspace import Workspace

PYTHON = """Traceback (most recent call last):
  File "/home/ci/app/shop/cart.py", line 5, in pay
    return checkout_total(sum(items))
  File "/home/ci/app/shop/pricing.py", line 2, in checkout_total
    return p / count
  File "/usr/lib/python3.12/site-packages/thing.py", line 9, in other
ZeroDivisionError: division by zero"""
NODE = """TypeError: Cannot read properties of undefined (reading 'price')
    at total (/srv/app/src/cart.js:3:15)
    at Object.<anonymous> (/srv/app/src/index.js:2:1)
    at node:internal/modules/cjs/loader:1256:14"""


class TracebackTests(unittest.TestCase):
    def test_frames_are_read_from_several_languages(self):
        self.assertEqual([(path.rsplit("/", 1)[-1], line) for path, line, _ in frames(PYTHON)],
                         [("cart.py", 5), ("pricing.py", 2), ("thing.py", 9)])
        self.assertEqual([(path.rsplit("/", 1)[-1], line) for path, line, _ in frames(NODE)][:2], [("cart.js", 3), ("index.js", 2)])
        self.assertEqual(frames("at com.shop.Cart.total(Cart.java:42)")[0][1:], (42, "com.shop.Cart.total"))
        self.assertTrue(requested(PYTHON))
        self.assertFalse(requested("Explain checkout_total"))

    def test_a_frame_is_matched_to_the_file_its_path_ends_with(self):
        files = ["shop/cart.py", "shop/pricing.py", "tests/pricing.py"]
        self.assertEqual(locate("/home/ci/app/shop/pricing.py", files), "shop/pricing.py")
        self.assertIsNone(locate("/elsewhere/pricing.py", ["a/pricing.py", "b/pricing.py"]))

    def test_the_model_is_shown_the_frames_innermost_first(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "shop").mkdir()
            (Path(folder) / "shop" / "pricing.py").write_text("def checkout_total(p, count=0):\n    return p / count\n")
            (Path(folder) / "shop" / "cart.py").write_text("from shop.pricing import checkout_total\n\n\n"
                                                         "def pay(items):\n    return checkout_total(sum(items))\n")
            seen = []

            class Model:
                prompt_budget = 12_000

                def chat(self, messages, schema, on_text=None):
                    seen.append(messages)
                    return {"action": "finish", "path": "", "message": "Done", "points": [],
                            "content": "count defaults to 0, so shop/pricing.py:2 divides by zero."}

            lines = []
            state = run_state(Model(), Workspace(folder), "Why do I get this error?\n" + PYTHON, output=lines.append)
            prompt = "\n".join(message["content"] for message in seen[0])
            self.assertIn("shop/pricing.py:2 in checkout_total (where the error was raised", prompt)
            self.assertLess(prompt.index("shop/pricing.py:2 in"), prompt.index("shop/cart.py:5 in"))
            self.assertIn("   2>     return p / count", prompt)
            self.assertIn("2 frames of the trace are in this project; 1 is in libraries or files not here.", lines)
            self.assertEqual(state["issues"], [])


if __name__ == "__main__":
    unittest.main()
