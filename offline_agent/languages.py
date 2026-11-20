"""Static import and definition analysis for Go, Java, and Rust. Files are read as text, never compiled or run.

Python is parsed with its own ast module and JavaScript/TypeScript with tree-sitter. These three languages are
handled here with line patterns and brace matching instead, so they need no extra packages. That is enough for
what CodeHaven draws and retrieves: which file imports which, and where each function, method, and type starts
and ends. It is an approximation: macros, generated code, and unusual layouts can be missed, and a brace inside
a multi-line string or block comment can end a span early.

What counts as an import between files:
- Go: an import of a package in this module (from go.mod, or a folder matching the end of the import path)
  links to the files of that package that define the names used, such as store.New.
- Java: an import of a class in this project, a wildcard import of one of its packages, or the use of a class
  from the same package.
- Rust: `mod name;` and `use crate::...`, `use super::...`, or `use self::...` paths that lead to a file.
"""
import posixpath
import re
from pathlib import PurePosixPath

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


class OtherDependencies:
    """Edges, definitions, and spans for the Go, Java, and Rust files among files, shaped like the Python analyser's."""

    def __init__(self, files, manifests=None):
        self.edges, self.definitions, self.spans = {}, {}, {}
        self.parse_failures = 0
        self.files = {name: text for name, text in files.items() if PurePosixPath(name).suffix in SUFFIXES}
        self.names = set(self.files)
        for name, text in self.files.items():
            self.outline(name, text)
        go_modules = sorted(((posixpath.dirname(name), match[1]) for name, text in (manifests or {}).items()
                             if (match := re.search(r"^module\s+(\S+)", text, re.M))), key=lambda row: -len(row[0]))
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
        """Where each function, method, and type of one file is: definitions (top level) and spans (all)."""
        lines = text.splitlines()
        cleaned = [clean(line) for line in lines]
        suffix = PurePosixPath(name).suffix
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
                method = JAVA_METHOD.match(line) if owners and depth == owners[-1][1] else None
                if kind:
                    symbol, opens_owner = (owner + "." if owner else "") + kind[1], True
                elif method and method[1] not in JAVA_WORDS and line.split("(")[0].split()[0] not in JAVA_WORDS \
                        and not line.rstrip().endswith(";"):
                    symbol = owner + "." + method[1]
            elif suffix == ".rs":
                implementation = RUST_IMPL.match(line)
                item = RUST_ITEM.match(line)
                if implementation and not item:
                    owners.append((implementation[1].rsplit("::", 1)[-1], depth + 1))
                elif item and (depth == 0 or (owners and depth == owners[-1][1])):
                    inside = owner and depth > 0
                    symbol = (owner + "." if inside else "") + item[2]
                    opens_owner = item[1] == "trait"
            if symbol:
                end = block_end(cleaned, index)
                short = symbol.rsplit(".", 1)[-1]
                if "." not in symbol or suffix == ".go":
                    definitions.setdefault(symbol if "." not in symbol else short, index + 1)
                if suffix != ".rs" or not re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:type|static|const)\s", line):
                    spans.append({"name": symbol, "start": index + 1, "end": end + 1})
                if opens_owner:
                    owners.append((symbol, depth + 1))
            depth = max(0, depth + line.count("{") - line.count("}"))
        self.definitions[name], self.spans[name] = definitions, spans

    def package_files(self, folder, suffix):
        return [name for name in self.names if name.endswith(suffix) and posixpath.dirname(name) == folder
                and not name.endswith("_test.go")]

    def go_imports(self, name, text, modules):
        found = {}
        imports = []
        for match in GO_IMPORT.finditer(text):
            imports += GO_IMPORT_LINE.findall(match[1]) if match[1] is not None else [(match[2] or "", match[3])]
        folders = {posixpath.dirname(other) for other in self.names if other.endswith(".go")}
        for alias, path in imports:
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
        body = "\n".join(clean(line) for line in text.splitlines())
        for other, package in packages.items():
            stem = PurePosixPath(other).stem
            if other != name and package == packages[name] and other not in found and re.search(rf"\b{re.escape(stem)}\b", body):
                found.setdefault(other, set()).add(stem)
        return found

    def module_file(self, base, segments):
        stem = posixpath.normpath(posixpath.join(base, *segments)) if segments else None
        if not stem or stem.startswith(".."):
            return None
        return next((candidate for candidate in (stem + ".rs", stem + "/mod.rs") if candidate in self.names), None)

    def rust_imports(self, name, text):
        found = {}
        folder, stem = posixpath.dirname(name), PurePosixPath(name).stem
        children = folder if stem in ("main", "lib", "mod") else posixpath.join(folder, stem)
        for module in RUST_MOD.findall(text):
            target = self.module_file(children, [module])
            if target:
                found.setdefault(target, set()).add("")
        parts = folder.split("/") if folder else []
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
                for size in range(len(rest), 0, -1):
                    target = self.module_file(base, rest[:size])
                    if target:
                        symbol = rest[size] if size < len(rest) and rest[size] not in ("*", "self") else ""
                        found.setdefault(target, set()).add(symbol)
                        break
        return found
