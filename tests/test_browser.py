"""The browser page in a real browser: open a repository, ask, stream the answer, and open a cited line.

Runs with Playwright and the Google Chrome already installed on this computer (nothing is downloaded), and is
skipped when either is missing. The model is scripted, so no Ollama is needed.
"""
import tempfile
import threading
import unittest
import unittest.mock
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


class EditModel:
    prompt_budget = 12_000

    def chat(self, messages, schema, on_text=None):
        return {"summary": "Include items at the threshold.",
                "edits": [{"path": "pkg/pricing.py", "old": "return p * 0.9", "new": "return p * 0.8"}]}


@unittest.skipUnless(sync_playwright, "Playwright is not installed (python -m pip install playwright)")
class BrowserEditTests(BrowserTests):
    """With --allow-edits, a change is shown as a diff on the page and applied only on Apply; Undo puts it back."""

    def setUp(self):
        super().setUp()
        self.app.model = EditModel()
        self.app.allow_edits = True
        self.journal = tempfile.TemporaryDirectory()
        self.addCleanup(self.journal.cleanup)
        patcher = unittest.mock.patch("offline_agent.edits.UNDO_FOLDER", Path(self.journal.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_open_ask_and_follow_a_citation(self):
        pass  # Covered by BrowserTests; this class tests changing code.

    def test_a_stopped_server_gets_a_plain_message(self):
        pass

    def test_apply_then_undo_from_the_page(self):
        page, source = self.page, self.repo / "pkg" / "pricing.py"
        before = source.read_text()
        page.goto(self.url)
        page.fill("#path", str(self.repo))
        page.click("#index")
        page.wait_for_selector("#question:enabled", timeout=15_000)
        page.wait_for_selector("#undo-change:visible", timeout=5_000)
        page.fill("#question", "Fix apply_discount in pkg/pricing.py")
        page.click("#send")
        page.wait_for_selector(".diff-box .del:has-text('return p * 0.9')", timeout=15_000)
        self.assertEqual(source.read_text(), before)  # Nothing is written before Apply.
        page.click(".diff-actions button:has-text('Apply')")
        page.wait_for_selector("#messages :text('Changed pkg/pricing.py')", timeout=15_000)
        self.assertIn("return p * 0.8", source.read_text())
        page.wait_for_selector("#question:enabled", timeout=15_000)
        page.click("#undo-change")
        page.wait_for_selector("#toast:has-text('Undid')", timeout=10_000)
        self.assertEqual(source.read_text(), before)
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
