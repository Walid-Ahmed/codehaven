"""The browser page in a real browser: open a repository, ask, stream the answer, and open a cited line.

Runs with Playwright and the Google Chrome already installed on this computer (nothing is downloaded), and is
skipped when either is missing. The model is scripted, so no Ollama is needed.
"""
import tempfile
import threading
import unittest
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError, sync_playwright
except ImportError:  # A dev-only dependency: python -m pip install playwright
    sync_playwright = None

from offline_agent.gui import App, make_server


class Model:
    prompt_budget = 12_000

    def chat(self, messages, schema, on_text=None):
        answer = "`checkout_total` applies the discount and returns the total (pkg/pricing.py:5-6)."
        if on_text:
            on_text(answer)
        return {"action": "finish", "path": "", "message": "Done", "content": answer,
                **({"points": []} if "points" in schema.get("properties", {}) else {})}


@unittest.skipUnless(sync_playwright, "Playwright is not installed (python -m pip install playwright)")
class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name, "repo")
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "pkg" / "pricing.py").write_text("def apply_discount(p):\n    return p * 0.9\n\n\n"
                                                      "def checkout_total(p):\n    return apply_discount(p)\n")
        self.app = App(Model())
        self.server, _, self.url = make_server(self.app)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), thread.join()))
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        try:
            self.browser = self.playwright.chromium.launch(channel="chrome", headless=True)
        except PlaywrightError as exc:
            self.skipTest(f"Google Chrome is not available to Playwright: {str(exc).splitlines()[0]}")
        self.addCleanup(self.browser.close)
        self.page = self.browser.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("console", lambda message: message.type == "error" and self.errors.append(message.text))

    def test_open_ask_and_follow_a_citation(self):
        page = self.page
        page.goto(self.url)
        page.fill("#path", str(self.repo))
        page.click("#index")
        # The page's content security policy forbids string evaluation, so waits use selectors, not JavaScript.
        page.wait_for_selector("#question:enabled", timeout=15_000)
        page.fill("#question", "Explain checkout_total")
        page.click("#send")
        link = page.locator("#messages a.cite").first
        link.wait_for(timeout=15_000)
        self.assertEqual(link.text_content(), "pkg/pricing.py:5-6")
        link.click()
        page.wait_for_selector("#drawer-code:has-text('def checkout_total')", timeout=10_000)
        self.assertIn("pkg/pricing.py:5-6", page.text_content("#drawer-title"))
        self.assertEqual(self.errors, [])

    def test_a_stopped_server_gets_a_plain_message(self):
        page = self.page
        page.goto(self.url)
        self.server.shutdown()
        self.server.server_close()
        message = page.evaluate("api('/api/status').then(() => 'no error', (error) => error.message)")
        self.assertIn("local server is not answering", message)


if __name__ == "__main__":
    unittest.main()
