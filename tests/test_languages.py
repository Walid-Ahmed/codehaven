"""Go, Java, and Rust: imports between files and where each part is, found by reading text, never by running code."""
import tempfile
import unittest
from pathlib import Path

from offline_agent import study
from offline_agent.knowledge import KnowledgeBase
from offline_agent.languages import OtherDependencies, expand
from offline_agent.workspace import Workspace

GO = {
    "go.mod": "module example.com/shop\n\ngo 1.22\n",
    "main.go": 'package main\n\nimport (\n\t"fmt"\n\n\t"example.com/shop/store"\n\tp "example.com/shop/pricing"\n)\n\n'
               'func main() {\n\ts := store.New()\n\tfmt.Println(p.Total(s.Count()), "}")\n}\n',
    "store/store.go": "package store\n\ntype Store struct {\n\titems map[string]int\n}\n\n// New makes a store.\n"
                      "func New() *Store {\n\treturn &Store{items: map[string]int{}}\n}\n\n"
                      "func (s *Store) Count() int {\n\treturn len(s.items)\n}\n",
    "store/audit.go": "package store\n\nfunc Audit() {}\n",
    "store/store_test.go": "package store\n\nfunc TestNew() {}\n",
    "pricing/pricing.go": "package pricing\n\nconst Rate = 2\n\nfunc Total(n int) int {\n\treturn n * Rate\n}\n",
}
JAVA = {
    "src/main/java/shop/App.java": "package shop;\n\nimport shop.pricing.Pricing;\nimport static shop.util.Text.slug;\n\n"
                                   "public class App {\n    public static void main(String[] args) {\n"
                                   "        Store store = new Store();\n        if (store.count() > 0) {\n"
                                   "            System.out.println(Pricing.total(store.count()));\n        }\n    }\n}\n",
    "src/main/java/shop/Store.java": "package shop;\n\npublic class Store {\n    private int items;\n\n    public int count() {\n"
                                     "        return items;\n    }\n\n    public void add(int n) { items += n; }\n}\n",
    "src/main/java/shop/Unused.java": "package shop;\n\nclass Unused {}\n",
    "src/main/java/shop/pricing/Pricing.java": "package shop.pricing;\n\npublic final class Pricing {\n"
                                               "    public static int total(int n) {\n        return n * 2;\n    }\n}\n",
    "src/main/java/shop/util/Text.java": "package shop.util;\n\npublic class Text {\n    public static String slug(String s) {\n"
                                         "        return s.toLowerCase();\n    }\n}\n",
}
RUST = {
    "Cargo.toml": "[package]\nname = \"shop\"\n",
    "src/main.rs": "mod store;\nmod pricing;\n\nuse crate::store::{Store, audit::log};\n\nfn main() {\n    let s = Store::new();\n"
                   "    log(pricing::total(s.count()));\n}\n",
    "src/store.rs": "pub mod audit;\n\nuse super::pricing::RATE;\n\npub struct Store {\n    items: usize,\n}\n\nimpl Store {\n"
                    "    pub fn new() -> Self {\n        Store { items: RATE }\n    }\n\n    pub fn count(&self) -> usize {\n"
                    "        self.items\n    }\n}\n\npub trait Named {\n    fn name(&self) -> String;\n}\n",
    "src/store/audit.rs": "pub fn log(n: usize) {\n    println!(\"{}\", n);\n}\n",
    "src/pricing.rs": "pub const RATE: usize = 2;\n\npub fn total(n: usize) -> usize {\n    n * RATE\n}\n",
}


def spans(found, name):
    return {span["name"]: (span["start"], span["end"]) for span in found.spans[name]}


class LanguageTests(unittest.TestCase):
    def test_go_imports_link_to_the_files_that_define_the_names_used(self):
        found = OtherDependencies(GO, {"go.mod": GO["go.mod"]})
        self.assertEqual(found.edges["main.go"], {"pricing/pricing.go": ["Total"], "store/store.go": ["New"]})
        self.assertEqual(found.edges["store/store.go"], {})
        self.assertEqual(spans(found, "store/store.go"), {"Store": (3, 5), "New": (8, 10), "Store.Count": (12, 14)})
        self.assertEqual(found.definitions["pricing/pricing.go"], {"Rate": 3, "Total": 5})
        self.assertEqual(spans(found, "main.go"), {"main": (10, 13)})  # The "}" inside a string does not end it early.
        without_module = OtherDependencies(GO)
        self.assertEqual(without_module.edges["main.go"], found.edges["main.go"])

    def test_java_imports_static_imports_and_same_package_use(self):
        found = OtherDependencies(JAVA)
        base = "src/main/java/shop/"
        self.assertEqual(found.edges[base + "App.java"],
                         {base + "Store.java": ["Store"], base + "pricing/Pricing.java": ["Pricing"], base + "util/Text.java": ["slug"]})
        self.assertEqual(spans(found, base + "Store.java"), {"Store": (3, 11), "Store.count": (6, 8), "Store.add": (10, 10)})
        self.assertEqual(spans(found, base + "App.java"), {"App": (6, 13), "App.main": (7, 12)})  # "if (...)" is not a method.

    def test_rust_modules_and_use_paths(self):
        found = OtherDependencies(RUST)
        self.assertEqual(found.edges["src/main.rs"],
                         {"src/pricing.rs": [""], "src/store.rs": ["", "Store"], "src/store/audit.rs": ["log"]})
        self.assertEqual(found.edges["src/store.rs"], {"src/pricing.rs": ["RATE"], "src/store/audit.rs": [""]})
        self.assertEqual(spans(found, "src/store.rs"),
                         {"Store": (5, 7), "Store.new": (10, 12), "Store.count": (14, 16), "Named": (19, 21), "Named.name": (20, 20)})
        self.assertEqual(found.definitions["src/pricing.rs"], {"RATE": 1, "total": 3})
        self.assertEqual(expand("crate::a::{b, c::{D, E as F}}"), ["crate::a::b", "crate::a::c::D", "crate::a::c::E"])

    def test_syntax_trees_handle_what_line_patterns_cannot(self):
        go = ("package shop\n\nconst (\n\tRate = 2\n\tLimit = 9\n)\n\nvar banner = `\n}\n`\n\n"
              "func Total(\n\tn int,\n) int {\n\t/* } */\n\treturn n * Rate\n}\n")
        java = ("package shop;\n\npublic class Cart {\n    private int items;\n\n    /* a stray } in a comment */\n"
                "    public Cart(int items) {\n        this.items = items;\n    }\n\n    @Override\n    public String toString(\n"
                "            ) {\n        return \"}\";\n    }\n\n    interface Listener {\n        void changed();\n    }\n}\n")
        rust = ("pub struct Cart<T> {\n    items: Vec<T>,\n}\n\nimpl<T: Clone> Cart<T> {\n    /* } */\n    pub fn total<F>(\n"
                "        &self,\n        price: F,\n    ) -> usize\n    where\n        F: Fn(&T) -> usize,\n    {\n"
                "        self.items.iter().map(price).sum()\n    }\n}\n")
        found = OtherDependencies({"shop.go": go, "Cart.java": java, "cart.rs": rust})
        self.assertEqual(found.parse_failures, 0)
        self.assertEqual(found.definitions["shop.go"], {"Rate": 4, "Limit": 5, "banner": 8, "Total": 12})
        self.assertEqual(spans(found, "shop.go")["Total"], (12, 17))
        self.assertEqual(spans(found, "Cart.java"),
                         {"Cart": (3, 20), "Cart.Cart": (7, 9), "Cart.toString": (11, 15), "Cart.Listener": (17, 19)})
        self.assertEqual(spans(found, "cart.rs"), {"Cart": (1, 3), "Cart.total": (7, 15)})

    def test_a_file_that_does_not_parse_falls_back_to_line_patterns(self):
        broken = "package shop\n\nfunc Total(n int) int {\n\treturn n *\n}\n\nfunc Count() int {\n\treturn 1\n}\n@@@\n"
        found = OtherDependencies({"shop.go": broken})
        self.assertEqual(found.parse_failures, 1)
        self.assertEqual(spans(found, "shop.go"), {"Total": (3, 5), "Count": (7, 9)})

    def test_the_map_and_retrieval_use_them(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in {**GO, **RUST}.items():
                target = Path(folder, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            workspace = Workspace(folder)
            knowledge = KnowledgeBase(workspace=workspace, context_budget=4000)
            drawn = study.structure(workspace, knowledge, ["main.go", "store/store.go", "src/main.rs"])
            links = {(link["from"], link["to"]) for link in drawn["links"]}
            self.assertIn(("main.go", "store/store.go"), links)
            self.assertIn(("src/main.rs", "src/store.rs"), links)
            parts = {part["name"]: part for part in drawn["files"][1]["parts"]}
            self.assertEqual((parts["Store.Count"]["start"], parts["Store.Count"]["end"]), (12, 14))
            self.assertEqual(knowledge.symbols["Total"], [("pricing/pricing.go", 5)])


if __name__ == "__main__":
    unittest.main()
