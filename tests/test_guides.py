"""Onboarding guide and Mermaid diagrams, drawn from the parsed code."""
import tempfile
import unittest
from pathlib import Path

from offline_agent.agent import run_state
from offline_agent.guides import requested
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace

FILES = {
    "README.md": "# Shop\n",
    "pyproject.toml": '[project]\nname = "shop"\ndependencies = ["httpx"]\n[tool.pytest.ini_options]\n',
    "shop/__init__.py": "",
    "shop/__main__.py": "from shop.cart import pay\npay([1])\n",
    "shop/cart.py": "from shop.pricing import checkout_total\n\n\ndef pay(items):\n    return checkout_total(sum(items))\n",
    "shop/pricing.py": "def apply_discount(p):\n    return p * 0.9\n\n\ndef checkout_total(p):\n    return apply_discount(p)\n",
    "tests/test_cart.py": "from shop.cart import pay\n\n\ndef test_pay():\n    assert pay([10]) == 9\n",
}


class Model:
    prompt_budget = 12_000

    def chat(self, messages, schema, on_text=None):
        return {"sentence": "Sets prices."}


class GuideTests(unittest.TestCase):
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
        return run_state(Model(), self.ws, task, knowledge=self.kb, output=lambda _: None)["final_text"]

    def test_requests_are_recognised(self):
        self.assertEqual(requested("I'm new to this project, where do I begin?"), ("onboarding", None))
        self.assertEqual(requested("Draw an architecture diagram"), ("diagram", None))
        self.assertEqual(requested("Mermaid diagram of the calls of checkout_total"), ("diagram", "checkout_total"))

    def test_the_import_diagram(self):
        text = self.ask("Draw an architecture diagram")
        self.assertIn("```mermaid\nflowchart LR", text)
        self.assertIn("n_shop_cart_py --> n_shop_pricing_py", text)
        self.assertNotIn("test_cart", text)

    def test_the_call_diagram(self):
        text = self.ask("Mermaid diagram of the calls of pay")
        self.assertIn("n_shop_cart_py_pay --> n_shop_pricing_py_checkout_total", text)
        self.assertIn("n_shop_pricing_py_checkout_total --> n_shop_pricing_py_apply_discount", text)

    def test_the_onboarding_guide(self):
        text = self.ask("Write an onboarding guide for a new developer")
        self.assertIn("`python -m pip install -e .` (pyproject.toml:1)", text)
        self.assertIn("`python -m pytest`", text)
        self.assertIn("tests/ (1 files)", text)
        self.assertIn("`httpx`", text)
        self.assertIn("shop/__main__.py", text)


if __name__ == "__main__":
    unittest.main()


class ReportTests(unittest.TestCase):
    setUp, ask = GuideTests.setUp, GuideTests.ask

    def test_one_report_holds_every_part(self):
        text = self.ask("Write a project report")
        for heading in ("## Reading order", "## How the parts connect", "## Libraries", "## Functions no test mentions",
                        "## Security and secrets"):
            self.assertIn(heading, text)
        self.assertIn("```mermaid", text)
