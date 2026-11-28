"""Dependencies, security scan, and impact: whole-project answers made by plain code, each finding cited."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run_state
from offline_agent.knowledge import KnowledgeBase
from offline_agent.scans import requested
from offline_agent.workspace import Workspace

FILES = {
    "pyproject.toml": '[project]\nname = "shop"\ndependencies = [\n    "httpx>=0.27",\n    "PyYAML",\n]\n',
    "shop/__init__.py": "",
    "shop/pricing.py": "import httpx\nimport yaml\nimport stripe\n\n\ndef apply_discount(p):\n    return p * 0.9\n\n\n"
                       "def checkout_total(p):\n    return apply_discount(p)\n",
    "shop/db.py": "import os\nAPI_KEY = 'sk-live1234567890abcdefghij'\nPASSWORD = 'changeme'\nTOKEN = os.environ['TOKEN']\n\n\n"
                  "def find(cursor, name):\n    cursor.execute(f\"SELECT * FROM items WHERE name = '{name}'\")\n    return eval(name)\n",
    "shop/cart.py": "from shop.pricing import checkout_total\n\n\ndef pay(items):\n    return checkout_total(sum(items))\n",
    "tests/test_pricing.py": "from shop.pricing import apply_discount\n\n\ndef test_discount():\n    assert apply_discount(10) == 9\n",
}


class Model:
    prompt_budget = 12_000

    def chat(self, *_):
        raise AssertionError("these answers use no model")


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for name, text in FILES.items():
            path = Path(self.temp.name, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.ws = Workspace(self.temp.name)
        self.kb = KnowledgeBase(workspace=self.ws)

    def ask(self, task):
        state = run_state(Model(), self.ws, task, knowledge=self.kb, output=lambda _: None)
        self.assertEqual(state["exit_code"], 0)
        return state["final_text"]

    def test_requests_are_recognised(self):
        self.assertEqual(requested("Which libraries does this project use?"), ("dependencies", None))
        self.assertEqual(requested("Run a security scan"), ("security", None))
        self.assertEqual(requested("What breaks if I change `apply_discount`?"), ("impact", "apply_discount"))
        self.assertIsNone(requested("Explain checkout_total"))

    def test_dependencies_are_listed_with_what_no_manifest_declares(self):
        text = self.ask("Which libraries does this project use?")
        self.assertIn("`httpx>=0.27`".replace(">=0.27", ""), text)
        self.assertIn("pyproject.toml:4", text)
        self.assertIn("`yaml`: 1 file", text)
        self.assertNotIn("`yaml`: 1 file, first at shop/pricing.py:2; not declared", text)  # PyYAML provides yaml.
        self.assertIn("`stripe`: 1 file, first at shop/pricing.py:3; not declared in a manifest", text)
        self.assertNotIn("`os`", text)

    def test_the_security_scan_cites_risks_and_masks_secrets(self):
        text = self.ask("Scan the project for secrets")
        self.assertIn("shop/db.py:2", text)
        self.assertNotIn("sk-live1234567890abcdefghij", text)
        self.assertNotIn("shop/db.py:3", text)  # A placeholder password is not a secret.
        self.assertNotIn("shop/db.py:4", text)  # Read from the environment.
        self.assertIn("SQL built from a string", text)
        self.assertIn("shop/db.py:9", text)

    def test_impact_lists_callers_importers_and_tests(self):
        text = self.ask("What breaks if I change apply_discount?")
        self.assertIn("defined at shop/pricing.py:6-7", text)
        self.assertIn("shop/pricing.py:11, in `checkout_total`", text)
        self.assertIn("tests/test_pricing.py:5", text)
        text = self.ask("Who depends on checkout_total?")
        self.assertIn("shop/cart.py:5, in `pay`", text)
        self.assertIn("- shop/cart.py", text)


if __name__ == "__main__":
    unittest.main()
