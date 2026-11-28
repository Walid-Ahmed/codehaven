"""Outlines and imports for C, C++, C#, Ruby, PHP, Swift, and Kotlin, parsed and never run."""
import json
import tempfile
import unittest
from pathlib import Path

from offline_agent.knowledge import KnowledgeBase
from offline_agent.more_languages import MoreDependencies, parser
from offline_agent.workspace import Workspace

FILES = {
    "src/util.h": "int add(int a, int b);\n",
    "src/util.c": '#include "util.h"\n#include <stdio.h>\nint add(int a, int b) {\n  return a + b;\n}\n',
    "geo/shape.hpp": "namespace geo {\nclass Shape {\n public:\n  int area() const { return 1; }\n  int perimeter();\n};\n}\n",
    "geo/shape.cpp": '#include "shape.hpp"\n#include "gone.hpp"\nnamespace geo {\nint Shape::perimeter() { return 2; }\n}\n',
    "Models/Item.cs": "namespace Shop.Models {\n  public class Item { public int Price() { return 1; } }\n}\n",
    "Services/Cart.cs": "using Shop.Models;\nusing Newtonsoft.Json;\nnamespace Shop.Services {\n  public class Cart {\n"
                        "    public int Total(Item i) { return i.Price(); }\n  }\n}\n",
    "lib/models/item.rb": "class Item\n  def price\n    1\n  end\nend\n",
    "lib/cart.rb": "require_relative 'models/item'\nrequire 'json'\nmodule Shop\n  class Cart\n    def total(n)\n      n\n    end\n"
                   "  end\nend\n",
    "app/Models/Item.php": "<?php\nnamespace App\\Models;\nclass Item { public function price() { return 1; } }\n",
    "app/Services/Cart.php": "<?php\nnamespace App\\Services;\nuse App\\Models\\Item;\nrequire_once 'helpers.php';\n"
                             "class Cart {\n  public function total($n) { return $n; }\n}\n",
    "app/Services/helpers.php": "<?php\nfunction h() {}\n",
    "Sources/Cart.swift": "import Foundation\nimport Alamofire\nstruct Cart {\n  func total(n: Int) -> Int { return n }\n}\n",
    "shop/models/Item.kt": "package shop.models\nclass Item { fun price(): Int { return 1 } }\n",
    "shop/services/Cart.kt": "package shop.services\nimport shop.models.Item\nclass Cart {\n  fun total(n: Int): Int { return n }\n}\n",
}


@unittest.skipUnless(all(parser(suffix) for suffix in (".c", ".cpp", ".cs", ".rb", ".php", ".swift", ".kt")),
                     "the Tree-sitter grammars for these languages are not installed")
class MoreLanguagesTests(unittest.TestCase):
    def setUp(self):
        self.found = MoreDependencies(FILES)

    def spans(self, name):
        return {span["name"]: (span["start"], span["end"]) for span in self.found.spans[name]}

    def test_functions_types_and_methods_are_outlined_with_their_lines(self):
        self.assertEqual(self.spans("src/util.c")["add"], (3, 5))
        self.assertIn("Shape.area", self.spans("geo/shape.hpp"))
        self.assertEqual(self.spans("geo/shape.cpp")["Shape.perimeter"], (4, 4))
        self.assertIn("Cart.Total", self.spans("Services/Cart.cs"))
        self.assertEqual(self.spans("lib/cart.rb")["Shop.Cart.total"], (5, 7))
        self.assertIn("Cart.total", self.spans("app/Services/Cart.php"))
        self.assertIn("Cart.total", self.spans("Sources/Cart.swift"))
        self.assertIn("Cart.total", self.spans("shop/services/Cart.kt"))
        # A nested type is found by its own name too, as a question would name it.
        self.assertEqual(self.found.definitions["lib/cart.rb"]["Cart"], 4)

    def test_imports_between_project_files_become_edges(self):
        edges = self.found.edges
        self.assertEqual(set(edges["src/util.c"]), {"src/util.h"})
        self.assertEqual(set(edges["geo/shape.cpp"]), {"geo/shape.hpp"})
        self.assertEqual(edges["Services/Cart.cs"], {"Models/Item.cs": ["Item"]})
        self.assertEqual(set(edges["lib/cart.rb"]), {"lib/models/item.rb"})
        self.assertEqual(set(edges["app/Services/Cart.php"]), {"app/Models/Item.php", "app/Services/helpers.php"})
        self.assertEqual(edges["shop/services/Cart.kt"], {"shop/models/Item.kt": ["Item"]})

    def test_imports_of_code_that_is_not_here_are_noted_and_standard_ones_are_not(self):
        missing = {name: [row["module"] for row in rows] for name, rows in self.found.missing.items() if rows}
        self.assertEqual(missing, {"geo/shape.cpp": ["gone.hpp"], "Services/Cart.cs": ["Newtonsoft.Json"],
                                   "lib/cart.rb": ["json"], "Sources/Cart.swift": ["Alamofire"]})

    def test_the_index_uses_them(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in FILES.items():
                path = Path(folder, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            kb = KnowledgeBase(workspace=Workspace(folder))
            self.assertIn(("lib/cart.rb", 4), kb.symbols["Cart"])
            result = json.loads(kb.retrieve("How does Shape perimeter work?"))
            self.assertTrue(any(row["source"].startswith("repo/geo/") for row in result["results"]))


if __name__ == "__main__":
    unittest.main()


class NotebookTests(unittest.TestCase):
    def test_notebooks_in_the_repository_are_indexed_without_their_outputs(self):
        notebook = {"cells": [{"cell_type": "code", "execution_count": 1, "metadata": {},
                               "outputs": [{"name": "stdout", "output_type": "stream", "text": ["zqxoutputonly\n"]}],
                               "source": ["def zqxcellcode(n):\n", "    return n + 1\n"]},
                              {"cell_type": "markdown", "metadata": {}, "source": ["# zqxheading\n"]}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "analysis.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
            kb = KnowledgeBase(workspace=Workspace(folder))
            found = json.loads(kb.retrieve("zqxcellcode zqxheading"))["results"]
            self.assertEqual({row["source"] for row in found}, {"repo/analysis.ipynb"})
            self.assertNotIn("zqxoutputonly", kb.retrieve("zqxoutputonly"))
            # Line numbers are those of the real file, so a citation of the code cell can be checked.
            lines = Path(folder, "analysis.ipynb").read_text().splitlines()
            row = found[0]
            self.assertIn("zqxcellcode", "\n".join(lines[row["start_line"] - 1:row["end_line"]]))
