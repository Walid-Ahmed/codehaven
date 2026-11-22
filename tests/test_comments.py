"""Every module, class, and function of CodeHaven says what it is for, and so does every function of the page."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommentTests(unittest.TestCase):
    def test_python_code_has_docstrings(self):
        missing = []
        for path in sorted([*(ROOT / "offline_agent").glob("*.py"), *(ROOT / "scripts").glob("*.py")]):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not ast.get_docstring(tree):
                missing.append(f"{path.name}: the module")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(missing, [])

    def test_test_files_say_what_they_cover(self):
        missing = [path.name for path in sorted((ROOT / "tests").glob("test_*.py"))
                   if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))]
        self.assertEqual(missing, [])

    def test_page_functions_have_a_comment_above_them(self):
        lines = (ROOT / "offline_agent" / "gui.js").read_text(encoding="utf-8").splitlines()
        missing = [f"gui.js:{number} {line[:60]}" for number, line in enumerate(lines[1:], 2)
                   if re.match(r"(?:async )?function\*? \w+", line) and not lines[number - 2].lstrip().startswith("//")]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
