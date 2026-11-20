"""Reading code from a picture. Only the last test class needs Tesseract; the rest use a table like the one it prints."""
import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent import ocr
from offline_agent.cli import main
from offline_agent.gui import App

FIXTURE = Path(__file__).with_name("fixtures") / "area.png"
PNG = b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + (800).to_bytes(4, "big") + (600).to_bytes(4, "big") + b"\x08\x02\0\0\0"
HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def table(*lines):
    """Tesseract's table for lines of (column, row, text, confidence) words in a 10-pixel-wide, 20-pixel-high font."""
    rows = []
    for number, words in enumerate(lines, 1):
        for column, row, text, *confidence in words:
            rows.append(f"5\t1\t1\t1\t{number}\t1\t{20 + column * 10}\t{row * 20}\t{len(text) * 10 - 2}\t14\t"
                        f"{confidence[0] if confidence else 95}\t{text}")
    return HEADER + "4\t1\t1\t1\t1\t0\t0\t0\t0\t0\t-1\t\n" + "\n".join(rows) + "\n"


FUNCTION = table([(0, 0, "def"), (4, 0, "area(w,"), (12, 0, "h):")],
                 [(4, 1, "if"), (7, 1, "w"), (9, 1, "<"), (11, 1, "@:", 70)],
                 [(8, 2, "return"), (15, 2, "“none”", 40)],
                 [(4, 3, "return"), (11, 3, "w"), (13, 3, "*"), (15, 3, "h")],
                 [(0, 6, "print(area(3,"), (14, 6, "4))")])


class LayoutTests(unittest.TestCase):
    def read(self, text, name="shot.png", **options):
        with patch.object(ocr, "recognise", return_value=text):
            return ocr.read_code(name, PNG, **options)

    def test_indentation_and_blank_lines_come_from_word_positions(self):
        found = self.read(FUNCTION)
        self.assertEqual(found["text"], 'def area(w, h):\n    if w < 0:\n        return "none"\n    return w * h\n\n\n'
                                        'print(area(3, 4))\n')
        self.assertEqual((found["name"], found["image"], found["lines"], found["syntax"]), ("shot.py", "shot.png", 7, None))

    def test_unsure_words_and_repaired_characters_are_listed_for_checking(self):
        self.assertEqual(self.read(FUNCTION)["check"], [2, 3])  # Line 2 had @ for 0; line 3 was read with low confidence.

    def test_python_that_does_not_parse_is_reported_with_its_line(self):
        found = self.read(table([(0, 0, "def"), (4, 0, "area(w)")], [(4, 1, "return"), (11, 1, "w")]))
        self.assertIn("line 1", found["syntax"])

    def test_a_decorator_or_matrix_product_keeps_its_at_sign(self):
        self.assertEqual(ocr.tidy("@property\ndef x(self): pass\n", True), ("@property\ndef x(self): pass\n", []))
        self.assertEqual(ocr.tidy("c = a @ b\n", True), ("c = a @ b\n", []))
        self.assertEqual(ocr.tidy("x = 1@ — 2\ndef f() :\n    pass\n", True), ("x = 10 - 2\ndef f():\n    pass\n", [1]))

    def test_an_editors_line_number_column_is_dropped(self):
        found = self.read(table([(0, 0, "8"), (3, 0, "x"), (5, 0, "="), (7, 0, "1")],
                                [(0, 1, "9"), (3, 1, "if"), (6, 1, "x:")],
                                [(0, 2, "10"), (7, 2, "y"), (9, 2, "="), (11, 2, "2")]))
        self.assertEqual(found["text"], "x = 1\nif x:\n    y = 2\n")

    def test_the_file_is_python_unless_the_pictures_name_says_otherwise(self):
        self.assertEqual([ocr.output_name(name) for name in ("Screenshot_at_10.30.15.png", "cart.js.png", "a.jpeg")],
                         ["Screenshot_at_10.30.15.py", "cart.js", "a.py"])
        found = self.read(table([(0, 0, "let"), (4, 0, "x"), (6, 0, "="), (8, 0, "1;")]), "cart.js.png")
        self.assertEqual((found["name"], found["syntax"]), ("cart.js", None))  # Only Python is syntax-checked.

    def test_only_pictures_within_the_limits_are_read(self):
        huge = PNG[:16] + (9000).to_bytes(4, "big") * 2 + PNG[24:]
        for data, message in [(b"GIF89a", "PNG and JPEG"), (PNG + b"\0" * ocr.IMAGE_BYTES, "8 MB"), (huge, "megapixels"),
                              (b"\xff\xd8\xff\xe0", "megapixels")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                with patch.object(ocr, "recognise", side_effect=AssertionError("must not run")):
                    ocr.read_code("shot.png", data)
        jpeg = b"\xff\xd8\xff\xe0\x00\x04\0\0\xff\xc0\x00\x11\x08" + (600).to_bytes(2, "big") + (800).to_bytes(2, "big")
        self.assertEqual((ocr.pixels(jpeg), ocr.pixels(PNG)), (480_000, 480_000))
        with self.assertRaisesRegex(ValueError, "No text was found"):
            self.read(HEADER)

    def test_a_missing_tesseract_is_explained(self):
        with patch.object(ocr.shutil, "which", return_value=None), self.assertRaisesRegex(ValueError, "brew install tesseract"):
            ocr.read_code("shot.png", PNG)


class PictureUploadTests(unittest.TestCase):
    def test_a_picture_becomes_the_uploaded_file_and_the_picture_itself_is_not_kept(self):
        app = App(object())
        self.addCleanup(app.close)
        with patch.object(ocr, "recognise", return_value=FUNCTION):
            summary = app.open_file("My Shot.png", PNG)
        self.assertEqual((summary["upload"], summary["repository"]), ("My_Shot.py", None))
        self.assertEqual(summary["picture"], {"image": "My_Shot.png", "lines": 7, "check": [2, 3], "syntax": None})
        self.assertEqual(app.summary()["picture"], summary["picture"])
        folder = app.upload_folder
        assert folder is not None
        self.assertEqual([path.name for path in folder.iterdir()], ["My_Shot.py"])
        self.assertEqual(app.text(), {"name": "My_Shot.py", "text": (folder / "My_Shot.py").read_text()})
        self.assertIn("    if w < 0:", app.file("My_Shot.py", 2)["lines"])
        app.open_file("notes.txt", b"plain\n")  # Opening something else forgets the picture.
        self.assertIsNone(app.summary()["picture"])
        self.assertFalse(folder.exists())

    def test_a_picture_that_cannot_be_read_opens_nothing(self):
        app = App(object())
        with patch.object(ocr.shutil, "which", return_value=None), self.assertRaisesRegex(ValueError, "Tesseract"):
            app.open_file("shot.png", PNG)
        self.assertEqual((app.upload_folder, app.busy), (None, False))
        with self.assertRaisesRegex(ValueError, "Upload a file first"):
            app.text()


class CommandTests(unittest.TestCase):
    def run_command(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch.object(ocr, "recognise", return_value=FUNCTION):
            return main(list(arguments)), out.getvalue(), err.getvalue()

    def test_the_command_writes_a_new_file_and_never_replaces_one(self):
        with tempfile.TemporaryDirectory() as folder:
            picture = Path(folder, "shot.png")
            picture.write_bytes(PNG)
            code, out, _ = self.run_command("--ocr", str(picture))
            self.assertEqual(code, 0)
            self.assertIn("Compare with the picture: line 2, 3.", out)
            self.assertIn("It is valid Python.", out)
            written = Path(folder, "shot.py").read_text()
            self.assertTrue(written.startswith("def area(w, h):\n    if w < 0:\n"))
            Path(folder, "shot.py").write_text("mine\n")
            code, _, err = self.run_command("--ocr", str(picture))
            self.assertEqual((code, Path(folder, "shot.py").read_text()), (1, "mine\n"))
            self.assertIn("already exists", err)
            self.assertEqual(self.run_command("--ocr", str(picture), "--out", str(Path(folder, "area.py")))[0], 0)
            self.assertEqual(Path(folder, "area.py").read_text(), written)
            self.assertEqual(picture.read_bytes(), PNG)


@unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
class TesseractTests(unittest.TestCase):
    def test_a_real_screenshot_is_read_with_its_indentation(self):
        found = ocr.read_code("area.png", FIXTURE.read_bytes())
        self.assertEqual(found["text"], "def area(width, height):\n    if width < 0:\n        return 0\n"
                                        "    return width * height\n\n\nprint(area(3, 4))\n")
        self.assertEqual((found["name"], found["syntax"]), ("area.py", None))


if __name__ == "__main__":
    unittest.main()
