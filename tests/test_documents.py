"""Notebooks and PDFs: read into one text file on this computer, cut at a cell or page, original untouched."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent import documents
from offline_agent.gui import App
from offline_agent.workspace import MAX_BYTES


def notebook(cells, language="python"):
    return json.dumps({"cells": cells, "metadata": {"kernelspec": {"language": language, "name": language}},
                       "nbformat": 4, "nbformat_minor": 5}).encode()


def code(text, outputs=()):
    return {"cell_type": "code", "source": text.splitlines(keepends=True), "outputs": list(outputs), "metadata": {}}


def markdown(text):
    return {"cell_type": "markdown", "source": text, "metadata": {}}


def pdf(pages):
    """A small real PDF with one block of Courier text per page, written by hand so no PDF library is needed."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
    kids = []
    for lines in pages:
        escaped = (line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines)
        stream = "BT /F1 11 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in escaped) + " ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


CAN_READ_PDF = bool(importlib.util.find_spec("pypdf") or shutil.which("pdftotext"))


class NotebookTests(unittest.TestCase):
    def test_cells_become_one_python_file_without_outputs(self):
        data = notebook([markdown("# Fees\nHow late fees work."),
                         code("%matplotlib inline\nimport math\n\ndef late_fee(days):\n    return min(days * 2, 20)\n",
                              [{"output_type": "stream", "text": ["SECRET OUTPUT"]}]),
                         code("   "), code("!pip install x\nlate_fee(3)")])
        found = documents.read_notebook("Fees Notes.ipynb", data)
        self.assertEqual(found["name"], "Fees Notes.py")
        self.assertEqual(found["text"].splitlines()[:4], ["# %% [markdown] Cell 1", "# # Fees", "# How late fees work.", ""])
        self.assertIn("# %% Cell 2\n# %matplotlib inline\nimport math\n\ndef late_fee(days):\n    return min(days * 2, 20)\n",
                      found["text"])
        self.assertIn("# %% Cell 3\n# !pip install x\nlate_fee(3)\n", found["text"])  # The blank cell is skipped.
        self.assertNotIn("SECRET OUTPUT", found["text"])
        compile(found["text"], found["name"], "exec")  # Magics are commented out, so the file is valid Python.
        self.assertIn("Read 3 cells (2 with code)", found["note"])

    def test_other_languages_and_long_notebooks(self):
        found = documents.read_notebook("stats.ipynb", notebook([code("x <- 1")], "R"))
        self.assertEqual(found["name"], "stats.r")
        cells = [code(f"value_{n} = '{'x' * 900}'") for n in range(120)]
        found = documents.read_notebook("long.ipynb", notebook(cells))
        self.assertLessEqual(len(found["text"].encode()), MAX_BYTES)
        self.assertRegex(found["note"], r"Only cells 1-\d+ of 120 fit")
        self.assertTrue(found["text"].rstrip().endswith("'"))  # Cut between cells, never inside one.

    def test_files_that_are_not_notebooks_are_refused(self):
        for data in (b"not json", b"[]", b'{"cells": 3}', notebook([])):
            with self.assertRaises(ValueError):
                documents.read_notebook("x.ipynb", data)

    def test_upload_opens_the_notebook_as_a_file_to_chat_about(self):
        app = App(object())
        self.addCleanup(app.close)
        summary = app.open_file("demo.ipynb", notebook([code("def area(r):\n    return 3.14 * r * r\n")]))
        self.assertEqual(summary["upload"], "demo.py")
        self.assertEqual(summary["document"]["kind"], "notebook")
        self.assertIn("def area(r):", app.text()["text"])
        self.assertIsNone(app.summary()["picture"])


class PdfTests(unittest.TestCase):
    PAGES = [["Late fees", "", "def late_fee(days):", "    if days <= 0:", "        return 0", "    return min(days * 2, 20)"],
             ["Second page", "x = late_fee(3)"]]

    @unittest.skipUnless(CAN_READ_PDF, "neither pypdf nor pdftotext is installed")
    def test_pages_become_text_with_page_marks_and_indentation(self):
        found = documents.read_pdf("paper.pdf", pdf(self.PAGES))
        self.assertEqual(found["name"], "paper.txt")
        lines = found["text"].splitlines()
        self.assertEqual(lines[0], "===== Page 1 =====")
        self.assertIn("def late_fee(days):", lines)
        # Readers space indentation differently; what matters is that the nested line stays the most indented.
        self.assertTrue(any(line.strip() == "return 0" and line.startswith("        ") for line in lines), lines)
        self.assertIn("===== Page 2 =====", lines)
        self.assertIn("Read 2 pages", found["note"])

    @unittest.skipUnless(CAN_READ_PDF, "neither pypdf nor pdftotext is installed")
    def test_upload_opens_the_pdf_and_leaves_the_original_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder, "paper.pdf")
            original.write_bytes(pdf(self.PAGES))
            before = original.read_bytes()
            app = App(object())
            self.addCleanup(app.close)
            summary = app.open_file("paper.pdf", before)
            self.assertEqual((summary["upload"], summary["document"]["kind"]), ("paper.txt", "pdf"))
            self.assertEqual(original.read_bytes(), before)

    def test_long_pdfs_are_cut_at_a_page_and_scans_say_what_to_do(self):
        with patch.object(documents, "pdf_pages", return_value=(["word " * 4000] * 10, 10)):
            found = documents.read_pdf("long.pdf", b"%PDF-1.4")
        self.assertLessEqual(len(found["text"].encode()), MAX_BYTES)
        self.assertRegex(found["note"], r"Only pages 1-3 of 10 fit")
        with patch.object(documents, "pdf_pages", return_value=(["", "  \n"], 2)):
            with self.assertRaisesRegex(ValueError, "scanned pages"):
                documents.read_pdf("scan.pdf", b"%PDF-1.4")

    def test_without_a_pdf_reader_the_message_says_how_to_get_one(self):
        with patch.dict("sys.modules", {"pypdf": None}), patch.object(documents.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "pip install pypdf"):
                documents.read_pdf("paper.pdf", pdf(self.PAGES))


if __name__ == "__main__":
    unittest.main()
