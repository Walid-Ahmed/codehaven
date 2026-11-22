"""Static import and definition analysis for Go, Java, and Rust. Files are read as text, never compiled or run.

Where each function, method, and type starts and ends is found with the Tree-sitter grammar of each language, the
same way JavaScript and TypeScript are parsed, so multi-line strings, block comments, and signatures spread over
several lines are handled. A file the grammar cannot parse cleanly, or a missing grammar package, falls back to
line patterns and brace matching, which is an approximation: there a brace inside a multi-line string or block
comment can end a span early. Imports are found with line patterns in both cases.

What counts as an import between files:
- Go: an import of a package in this module (from go.mod, or a folder matching the end of the import path)
  links to the files of that package that define the names used, such as store.New.
- Java: an import of a class in this project, a wildcard import of one of its packages, or the use of a class
  from the same package.
- Rust: `mod name;` and `use crate::...`, `use super::...`, or `use self::...` paths that lead to a file.
"""
import importlib
import posixpath
import re
from pathlib import PurePosixPath

from tree_sitter import Language, Parser

SUFFIXES = {".go", ".java", ".rs"}
SPAN_LINES = 3000  # The farthest a closing brace is looked for.
STRINGS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|`[^`]*`')

GO_FUNC = re.compile(r"^func\s+(?:\(\s*\w*\s*\*?\s*(\w+)(?:\[[^\]]*\])?\s*\)\s*)?(\w+)\s*[(\[]")
GO_TYPE = re.compile(r"^type\s+(\w+)\b")
GO_VALUE = re.compile(r"^(?:var|const)\s+(\w+)\b")
GO_IMPORT = re.compile(r'^import\s*\(([^)]*)\)|^import\s+(?:([\w.]+)\s+)?"([^"]+)"', re.M)
GO_IMPORT_LINE = re.compile(r'^\s*(?:([\w.]+)\s+)?"([^"]+)"', re.M)

JAVA_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(\.\*)?\s*;", re.M)
JAVA_MODIFIERS = r"(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed|strictfp|synchronized|native|default)\s+)*"
JAVA_TYPE = re.compile(rf"^\s*{JAVA_MODIFIERS}(?:class|interface|enum|record|@interface)\s+(\w+)")
JAVA_METHOD = re.compile(rf"^\s*{JAVA_MODIFIERS}(?:<[^>]+>\s+)?[\w.]+(?:<[^()]*>)?(?:\[\])*\s+(\w+)\s*\(")
JAVA_WORDS = {"if", "for", "while", "switch", "catch", "return", "new", "else", "throw", "try", "do", "synchronized", "assert"}

RUST_ITEM = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|const|unsafe|default)\s+|extern\s+\"[^\"]*\"\s+)*"
                       r"(fn|struct|enum|trait|union|type|static|const)\s+(\w+)")
RUST_IMPL = re.compile(r"^\s*(?:unsafe\s+)?impl\b(?:\s*<[^{]*?>)?\s+(?:[^{]*?\s+for\s+)?(?:&\s*)?(?:dyn\s+)?([\w:]+)")
RUST_MOD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M)
RUST_USE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+((?:crate|super|self)::[^;]+);", re.M)


def clean(line):
    """A line without its string literals and trailing comment, so braces and names in them are not counted."""
    return re.sub(r"//.*$", "", STRINGS.sub('""', line))


def block_end(cleaned, start):
    """Index of the line that closes the block opened at or after index start; start itself for a one-line item."""
    depth, opened = 0, False
    for index in range(start, min(len(cleaned), start + SPAN_LINES)):
        for char in cleaned[index]:
            if char == "{":
                depth, opened = depth + 1, True
            elif char == "}":
                depth -= 1
                if opened and depth <= 0:
                    return index
        if not opened and cleaned[index].rstrip().endswith(";"):
            return index
    return start


def split_top(text):
    """Split at commas that are not inside braces."""
    parts, depth, current = [], 0, ""
    for char in text:
        depth += (char == "{") - (char == "}")
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [part.strip() for part in parts + [current] if part.strip()]


def expand(path):
    """Every plain path a Rust use tree names: a::{b, c::D} gives a::b and a::c::D."""
    path = re.sub(r"\s+as\s+\w+", "", " ".join(path.split()))
    brace = path.find("{")
    if brace < 0:
        return [path.replace(" ", "")]
    close = path.rfind("}")
    prefix = path[:brace].replace(" ", "")
    return [found for part in split_top(path[brace + 1:close if close > brace else len(path)]) for found in expand(prefix + part)]


GRAMMARS = {".go": "tree_sitter_go", ".java": "tree_sitter_java", ".rs": "tree_sitter_rust"}
JAVA_TYPES = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"}
RUST_ITEMS = {"function_item", "struct_item", "enum_item", "union_item", "trait_item", "type_item", "static_item", "const_item"}
RUST_VALUES = {"type_item", "static_item", "const_item"}  # Definitions, but not blocks worth showing as a span.
_parsers = {}


def parser(suffix):
    """The Tree-sitter parser for a file suffix, made once; None when its grammar package cannot be loaded."""
    if suffix not in _parsers:
        try:
            _parsers[suffix] = Parser(Language(importlib.import_module(GRAMMARS[suffix]).language()))
        except Exception:  # Not installed, or built for another Tree-sitter version: the line patterns take over.
            _parsers[suffix] = None
    return _parsers[suffix]


def node_name(node, field="name"):
    """The text of a node's name field, or "" when it has none."""
    found = node.child_by_field_name(field) if node is not None else None
    return found.text.decode("utf-8", "replace") if found is not None else ""


def parsed_outline(suffix, text):
    """(definitions, spans) of one file from its syntax tree, or None when it cannot be parsed cleanly.

    Names follow the line patterns' conventions: Type.method for methods, Outer.Inner for nested Java types.
    """
    found = parser(suffix)
    if found is None:
        return None
    try:
        root = found.parse(text.encode("utf-8")).root_node
    except (ValueError, UnicodeError):
        return None
    if root.has_error:
        return None
    definitions, spans = {}, []

    def add(symbol, node, *, span=True, shared=None):
        """Record one symbol. shared is the name other files use for it, when that differs from needing no dot."""
        if not symbol or symbol.endswith("."):
            return
        start, end = node.start_point[0] + 1, node.end_point[0] + 1
        public = shared if shared is not None else (symbol if "." not in symbol else None)
        if public:
            definitions.setdefault(public, start)
        if span:
            spans.append({"name": symbol, "start": start, "end": end})

    if suffix == ".go":
        for node in root.named_children:
            if node.type == "function_declaration":
                add(node_name(node), node)
            elif node.type == "method_declaration":
                receiver = node.child_by_field_name("receiver")
                owner = next((child.text.decode() for child in walk(receiver) if child.type == "type_identifier"), "") if receiver else ""
                name = node_name(node)
                # Another file calls a method by its short name, so that is what it is looked up by.
                add((owner + "." if owner else "") + name, node, shared=name)
            elif node.type in ("type_declaration", "var_declaration", "const_declaration"):
                specs = [child for child in walk(node) if child.type in ("type_spec", "type_alias", "var_spec", "const_spec")]
                for spec in specs:
                    names = [child for child in spec.children_by_field_name("name")]
                    for name in names:
                        # A declaration of one thing is shown whole, with its keyword; a grouped one item by item.
                        add(name.text.decode(), node if len(specs) == 1 and len(names) == 1 else spec)
    elif suffix == ".java":
        def visit(body, owner):
            """Record the types and methods directly inside one type body, or at the top of the file."""
            for node in body.named_children:
                if node.type == "enum_body_declarations":
                    visit(node, owner)
                elif node.type in JAVA_TYPES:
                    symbol = (owner + "." if owner else "") + node_name(node)
                    add(symbol, node)
                    inner = node.child_by_field_name("body")
                    if inner is not None:
                        visit(inner, symbol)
                elif owner and node.type in ("method_declaration", "constructor_declaration") \
                        and node.child_by_field_name("body") is not None:
                    add(owner + "." + node_name(node), node)
        visit(root, "")
    else:
        for node in root.named_children:
            if node.type in RUST_ITEMS:
                symbol = node_name(node)
                add(symbol, node, span=node.type not in RUST_VALUES)
                if node.type == "trait_item":
                    members(node, symbol, add)
            elif node.type == "impl_item":
                # An impl block defines nothing itself; it names the type that the functions inside belong to.
                kind = node.child_by_field_name("type")
                owner = re.findall(r"[A-Za-z_]\w*", re.sub(r"<.*", "", kind.text.decode(), flags=re.S))[-1:] if kind else []
                if owner:
                    members(node, owner[0], add)
    return definitions, spans


def members(node, owner, add):
    """Record the functions inside a Rust trait or impl block as Owner.function."""
    body = node.child_by_field_name("body")
    for child in body.named_children if body is not None else ():
        if child.type in ("function_item", "function_signature_item"):
            add(owner + "." + node_name(child), child)


def walk(root):
    """Every named node under root, parents before children, in source order."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


class OtherDependencies:
    """Edges, definitions, and spans for the Go, Java, and Rust files among files, shaped like the Python analyser's."""

    def __init__(self, files, manifests=None):
        """files maps paths to text. manifests maps go.mod paths to their text, which names each Go module's import prefix."""
        self.edges, self.definitions, self.spans = {}, {}, {}
        self.parse_failures = 0  # Files outlined with line patterns because their syntax tree had errors.
        self.files = {name: text for name, text in files.items() if PurePosixPath(name).suffix in SUFFIXES}
        self.names = set(self.files)
        for name, text in self.files.items():
            parsed = parsed_outline(PurePosixPath(name).suffix, text)
            if parsed is None:
                self.parse_failures += parser(PurePosixPath(name).suffix) is not None
                self.outline(name, text)
            else:
                self.definitions[name], self.spans[name] = parsed
        # (folder, module path) for each go.mod, longest folder first.
        go_modules = sorted(((posixpath.dirname(name), match[1]) for name, text in (manifests or {}).items()
                             if (match := re.search(r"^module\s+(\S+)", text, re.M))), key=lambda row: -len(row[0]))
        # The package of every Java file; "" is the default package.
        java = {}
        for name, text in self.files.items():
            if name.endswith(".java"):
                package = JAVA_PACKAGE.search(text)
                java[name] = package[1] if package else ""
        for name, text in self.files.items():
            suffix = PurePosixPath(name).suffix
            found = (self.go_imports(name, text, go_modules) if suffix == ".go" else
                     self.java_imports(name, text, java) if suffix == ".java" else self.rust_imports(name, text))
            self.edges[name] = {target: sorted(symbols) for target, symbols in sorted(found.items()) if target != name}

    def outline(self, name, text):
        """The fallback outline, from line patterns: definitions (top level) and spans (all) of one file."""
        lines = text.splitlines()
        cleaned = [clean(line) for line in lines]
        suffix = PurePosixPath(name).suffix
        # owners is a stack of (name, brace depth of its body) for the enclosing class, trait, or impl block;
        # depth is the brace depth at the start of the current line.
        definitions, spans, owners, depth = {}, [], [], 0
        for index, line in enumerate(cleaned):
            while owners and depth < owners[-1][1]:
                owners.pop()  # The block that owned later items has closed.
            owner = owners[-1][0] if owners else ""
            symbol, opens_owner = None, False
            if suffix == ".go" and depth == 0:
                function = GO_FUNC.match(line)
                kind = GO_TYPE.match(line) or GO_VALUE.match(line)
                if function:
                    symbol = (function[1] + "." if function[1] else "") + function[2]
                elif kind:
                    symbol = kind[1]
            elif suffix == ".java":
                kind = JAVA_TYPE.match(line)
                # Methods are looked for only directly inside a type's body, not inside nested blocks.
                method = JAVA_METHOD.match(line) if owners and depth == owners[-1][1] else None
                if kind:
                    symbol, opens_owner = (owner + "." if owner else "") + kind[1], True
                # Not statements that look like a method, such as "return find(...)", nor declarations without a body.
                elif method and method[1] not in JAVA_WORDS and line.split("(")[0].split()[0] not in JAVA_WORDS \
                        and not line.rstrip().endswith(";"):
                    symbol = owner + "." + method[1]
            elif suffix == ".rs":
                implementation = RUST_IMPL.match(line)
                item = RUST_ITEM.match(line)
                # An impl block defines nothing itself; it names the type that the functions inside belong to.
                if implementation and not item:
                    owners.append((implementation[1].rsplit("::", 1)[-1], depth + 1))
                elif item and (depth == 0 or (owners and depth == owners[-1][1])):
                    inside = owner and depth > 0
                    symbol = (owner + "." if inside else "") + item[2]
                    opens_owner = item[1] == "trait"
            if symbol:
                end = block_end(cleaned, index)
                short = symbol.rsplit(".", 1)[-1]
                # definitions holds what another file can use by name: top-level items, and Go methods under their short name.
                if "." not in symbol or suffix == ".go":
                    definitions.setdefault(symbol if "." not in symbol else short, index + 1)
                # Rust type aliases, statics, and constants are definitions, but not spans to show as a block.
                if suffix != ".rs" or not re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:type|static|const)\s", line):
                    spans.append({"name": symbol, "start": index + 1, "end": end + 1})
                if opens_owner:
                    owners.append((symbol, depth + 1))
            depth = max(0, depth + line.count("{") - line.count("}"))
        self.definitions[name], self.spans[name] = definitions, spans

    def package_files(self, folder, suffix):
        """The indexed files directly in folder with this suffix, Go test files left out: for Go, the files of one package."""
        return [name for name in self.names if name.endswith(suffix) and posixpath.dirname(name) == folder
                and not name.endswith("_test.go")]

    def go_imports(self, name, text, modules):
        """{file: names used} for the files of this project's packages that one Go file imports."""
        found = {}
        imports = []
        for match in GO_IMPORT.finditer(text):
            imports += GO_IMPORT_LINE.findall(match[1]) if match[1] is not None else [(match[2] or "", match[3])]
        folders = {posixpath.dirname(other) for other in self.names if other.endswith(".go")}
        for alias, path in imports:
            # Under a known module, the rest of the import path is the folder inside that module.
            folder = next((posixpath.normpath(posixpath.join(base, path[len(module):].lstrip("/")))
                           for base, module in modules if path == module or path.startswith(module + "/")), None)
            if folder is None:  # No go.mod was indexed: accept a folder that the import path ends with.
                parts = path.split("/")
                folder = next((candidate for size in range(len(parts), 0, -1)
                               if (candidate := "/".join(parts[-size:])) in folders and size > 1 or
                               (size == 1 and (candidate := parts[-1]) in folders and len(parts) > 1)), None)
            folder = "" if folder == "." else folder
            targets = self.package_files(folder, ".go") if folder is not None else []
            if not targets or posixpath.dirname(name) == folder:
                continue
            # Without a usable alias, the package is referred to by the last part of its path. Uses such as store.New then
            # tell which files of the package are needed.
            alias = alias if alias not in ("", "_", ".") else path.rsplit("/", 1)[-1]
            used = set(re.findall(rf"\b{re.escape(alias)}\.([A-Za-z_]\w*)", "\n".join(clean(line) for line in text.splitlines())))
            linked = False
            for target in targets:
                names = used & set(self.definitions.get(target, {}))
                if names:
                    found.setdefault(target, set()).update(names)
                    linked = True
            if not linked:  # Imported for its side effects, or its names could not be told apart.
                for target in targets:
                    found.setdefault(target, set()).add("")
        return found

    def java_imports(self, name, text, packages):
        """{file: names used} for the project classes one Java file imports or, within its own package, mentions."""
        found = {}
        by_class = {(package + "." if package else "") + PurePosixPath(other).stem: other for other, package in packages.items()}
        for path, wildcard in JAVA_IMPORT.findall(text):
            if wildcard:
                for other, package in packages.items():
                    if package == path:
                        found.setdefault(other, set()).add("")
                continue
            parts = path.split(".")
            # A static import names a member, and a nested class its outer class: try shorter paths too.
            for size in range(len(parts), max(len(parts) - 3, 0), -1):
                target = by_class.get(".".join(parts[:size]))
                if target:
                    found.setdefault(target, set()).add(parts[size] if size < len(parts) else parts[size - 1])
                    break
        # Classes of the same package need no import, so their names are looked for in the code itself.
        body = "\n".join(clean(line) for line in text.splitlines())
        for other, package in packages.items():
            stem = PurePosixPath(other).stem
            if other != name and package == packages[name] and other not in found and re.search(rf"\b{re.escape(stem)}\b", body):
                found.setdefault(other, set()).add(stem)
        return found

    def module_file(self, base, segments):
        """The indexed file of a Rust module path under base: a/b.rs or a/b/mod.rs. None when not indexed or outside the repo."""
        stem = posixpath.normpath(posixpath.join(base, *segments)) if segments else None
        if not stem or stem.startswith(".."):
            return None
        return next((candidate for candidate in (stem + ".rs", stem + "/mod.rs") if candidate in self.names), None)

    def rust_imports(self, name, text):
        """{file: names used} for the files one Rust file reaches through mod declarations and crate/super/self use paths."""
        found = {}
        folder, stem = posixpath.dirname(name), PurePosixPath(name).stem
        # Where this file's child modules live: beside main.rs, lib.rs, or mod.rs; otherwise in a folder named after the file.
        children = folder if stem in ("main", "lib", "mod") else posixpath.join(folder, stem)
        for module in RUST_MOD.findall(text):
            target = self.module_file(children, [module])
            if target:
                found.setdefault(target, set()).add("")
        parts = folder.split("/") if folder else []
        # crate:: paths start at the nearest src folder above this file, or at the file's own folder when there is none.
        root = "/".join(parts[:len(parts) - parts[::-1].index("src")]) if "src" in parts else folder
        for tree in RUST_USE.findall(text):
            for path in expand(tree):
                segments = [segment for segment in path.split("::") if segment]
                head, base = segments[0], root
                rest = segments[1:]
                if head == "self":
                    base = children
                elif head == "super":
                    base = posixpath.dirname(children) if stem in ("main", "lib", "mod") else folder
                    while rest and rest[0] == "super":
                        base, rest = posixpath.dirname(base), rest[1:]
                # The longest leading part of the path that is a file wins; the segment after it is the name taken from that file.
                for size in range(len(rest), 0, -1):
                    target = self.module_file(base, rest[:size])
                    if target:
                        symbol = rest[size] if size < len(rest) and rest[size] not in ("*", "self") else ""
                        found.setdefault(target, set()).add(symbol)
                        break
        return found
