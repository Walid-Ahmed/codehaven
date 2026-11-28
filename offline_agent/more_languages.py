"""Outlines and imports for C, C++, C#, Ruby, PHP, Swift, and Kotlin. Files are parsed, never compiled or run.

Each file is parsed with its Tree-sitter grammar to find where its functions, methods, and types start and end,
named the way the other analysers name them (Type.method for a method), so a question that names one is shown it
whole, a trace can follow it, and the structure map can draw it. A grammar that is not installed leaves those
files to keyword search, as before. A file with a syntax error is still outlined from what the grammar could
recover, and is counted in parse_failures.

What counts as an import between files:
- C and C++: #include "file", found beside the including file or anywhere in the project by the end of its path.
  #include <file> names a system or library header, and is noted as code that is not in the repository.
- C#: using Name.Space links to the files of that namespace that define a type the importing file mentions.
- Ruby: require_relative "path", and require "path" when the project has lib/path.rb or path.rb.
- PHP: require/include of a literal path, and use Name\\Space\\Class to the file that declares that class there.
- Kotlin: import a.b.Name to the file of package a.b that defines Name.
- Swift: files of one module see each other without imports, so no links are drawn; import X is noted as
  code that is not in the repository.
"""
import importlib
import posixpath
import re
from collections import defaultdict
from pathlib import PurePosixPath

from tree_sitter import Language, Parser

# Suffix -> (grammar module, the function that gives its language).
GRAMMARS = {
    ".c": ("tree_sitter_c", "language"), ".h": ("tree_sitter_c", "language"),
    ".cc": ("tree_sitter_cpp", "language"), ".cpp": ("tree_sitter_cpp", "language"),
    ".cxx": ("tree_sitter_cpp", "language"), ".hpp": ("tree_sitter_cpp", "language"),
    ".hh": ("tree_sitter_cpp", "language"), ".hxx": ("tree_sitter_cpp", "language"),
    ".cs": ("tree_sitter_c_sharp", "language"),
    ".rb": ("tree_sitter_ruby", "language"),
    ".php": ("tree_sitter_php", "language_php"),
    ".swift": ("tree_sitter_swift", "language"),
    ".kt": ("tree_sitter_kotlin", "language"), ".kts": ("tree_sitter_kotlin", "language"),
}
SUFFIXES = set(GRAMMARS)
# Nodes whose body holds methods: their name becomes the owner of what is defined inside.
TYPES = {"struct_specifier", "class_specifier", "union_specifier", "enum_specifier",          # C, C++
         "class_declaration", "struct_declaration", "interface_declaration", "enum_declaration",  # C#, PHP, Swift, Kotlin
         "record_declaration", "trait_declaration", "protocol_declaration", "object_declaration",
         "class", "module"}                                                                       # Ruby
# Nodes that are a function or method with a body.
FUNCTIONS = {"function_definition", "method_declaration", "constructor_declaration", "function_declaration",
             "init_declaration", "method", "singleton_method", "protocol_function_declaration"}
# Containers that only group definitions: they are walked through, and add nothing to a name.
GROUPS = {"namespace_definition", "declaration_list", "field_declaration_list", "body_statement", "class_body",
          "protocol_body", "namespace_declaration", "file_scoped_namespace_declaration", "compilation_unit",
          "translation_unit", "program", "source_file", "linkage_specification", "template_declaration",
          "enum_class_body", "statements"}
NAMES = ("identifier", "type_identifier", "simple_identifier", "constant", "name", "field_identifier",
         "qualified_identifier", "scope_resolution", "destructor_name", "operator_name")
_parsers = {}


def parser(suffix):
    """The Tree-sitter parser for a suffix, made once; None when its grammar package is not installed."""
    if suffix not in _parsers:
        try:
            module, function = GRAMMARS[suffix]
            _parsers[suffix] = Parser(Language(getattr(importlib.import_module(module), function)()))
        except Exception:  # Not installed, or built for another Tree-sitter version: keyword search still works.
            _parsers[suffix] = None
    return _parsers[suffix]


def text(node):
    """A node's source text."""
    return node.text.decode("utf-8", "replace") if node is not None else ""


def name_of(node):
    """The name a definition node declares: its name field, else its declarator's name, else its first name child."""
    found = node.child_by_field_name("name")
    if found is None:
        declarator = node.child_by_field_name("declarator")
        # C and C++ wrap a function's name in declarators: pointer, reference, function, then the identifier.
        while declarator is not None and declarator.type not in NAMES:
            declarator = declarator.child_by_field_name("declarator") or next(
                (child for child in declarator.named_children if child.type in NAMES), None)
        found = declarator
    if found is None:
        # Swift names an extension by its user_type, and Kotlin and Swift by an unlabelled identifier child.
        found = next((child for child in node.named_children if child.type in NAMES or child.type == "user_type"), None)
    if found is None:
        return ""
    if node.type == "init_declaration":
        return "init"
    # Cart::total and Shop::Cart become Cart.total and Shop.Cart, like the other languages' method names.
    return re.sub(r"\s+", "", text(found)).replace("::", ".").replace("\\", ".").strip(".")


def outline(suffix, source):
    """(definitions, spans, clean) of one file: top-level names and their lines, every function and type, and
    whether the grammar parsed it without errors. None when the grammar is not installed."""
    found = parser(suffix)
    if found is None:
        return None
    try:
        root = found.parse(source.encode("utf-8")).root_node
    except (ValueError, UnicodeError):
        return None
    definitions, spans = {}, []

    def visit(node, owner):
        """Record the definitions directly inside node, with owner as the name of the type they belong to."""
        for child in node.named_children:
            if child.type in TYPES or child.type in FUNCTIONS:
                name = name_of(child)
                body = child.child_by_field_name("body") or next(
                    (part for part in child.named_children if part.type in GROUPS), None)
                # A declaration without a body, such as struct point; or a prototype, is not a definition here.
                if not name or (body is None and child.type not in FUNCTIONS - {"function_definition"}):
                    if child.type in GROUPS:
                        visit(child, owner)
                    continue
                symbol = (owner + "." if owner and "." not in name else "") + name
                start, end = child.start_point[0] + 1, child.end_point[0] + 1
                # Top-level names, and every type by its own name: Shop::Cart in Ruby is asked about as Cart.
                if not owner or child.type in TYPES:
                    definitions.setdefault(name.rsplit(".", 1)[-1], start)
                spans.append({"name": symbol, "start": start, "end": end})
                if child.type in TYPES and body is not None:
                    visit(body, symbol)
            elif child.type in GROUPS or child.type == "ERROR":
                visit(child, owner)
    visit(root, "")
    return definitions, spans, not root.has_error


INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]+)"|<([^>]+)>)', re.M)
CSHARP_USING = re.compile(r"^\s*(?:global\s+)?using\s+(?:static\s+)?([\w.]+)\s*;", re.M)
CSHARP_NAMESPACE = re.compile(r"^\s*namespace\s+([\w.]+)", re.M)
RUBY_REQUIRE = re.compile(r"""^\s*(require_relative|require|load)\s*\(?\s*['"]([^'"]+)['"]""", re.M)
PHP_INCLUDE = re.compile(r"""\b(?:require|include)(?:_once)?\s*\(?\s*(?:__DIR__\s*\.\s*)?['"]([^'"]+\.php)['"]""")
PHP_NAMESPACE = re.compile(r"^\s*namespace\s+([\w\\]+)\s*[;{]", re.M)
PHP_USE = re.compile(r"^\s*use\s+(?!function\b|const\b)([\w\\]+)(?:\s+as\s+(\w+))?\s*;", re.M)
KOTLIN_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)", re.M)
KOTLIN_IMPORT = re.compile(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", re.M)
SWIFT_IMPORT = re.compile(r"^\s*(?:@\w+\s+)*import\s+(?:(?:class|struct|enum|protocol|func|var|let|typealias)\s+)?(\w+)", re.M)
# Standard headers and modules, which need no source to be understood; left out of the notes on missing code.
STANDARD = {"Foundation", "UIKit", "SwiftUI", "Combine", "XCTest", "Darwin", "Dispatch", "os", "AppKit", "Swift"}


class MoreDependencies:
    """Edges, definitions, spans, and imports of missing code for C, C++, C#, Ruby, PHP, Swift, and Kotlin files,
    shaped like the other analysers' results."""

    def __init__(self, files):
        """files maps repository paths to text; only the suffixes above are analysed."""
        self.edges, self.definitions, self.spans, self.missing = {}, {}, {}, {}
        self.parse_failures = 0
        self.files = {name: source for name, source in files.items() if PurePosixPath(name).suffix.lower() in SUFFIXES}
        self.names = set(files)
        outlined = {}
        for name, source in self.files.items():
            found = outline(PurePosixPath(name).suffix.lower(), source)
            if found is None:
                continue
            definitions, spans, clean = found
            self.parse_failures += not clean
            self.definitions[name], self.spans[name] = definitions, spans
            outlined[name] = definitions
        # Which files declare each namespace or package, for C#, PHP, and Kotlin imports.
        spaces = defaultdict(list)
        for name, source in self.files.items():
            pattern = {".cs": CSHARP_NAMESPACE, ".php": PHP_NAMESPACE, ".kt": KOTLIN_PACKAGE, ".kts": KOTLIN_PACKAGE}.get(
                PurePosixPath(name).suffix.lower())
            if pattern is not None:
                for space in pattern.findall(source):
                    spaces[space.replace("\\", ".")].append(name)
        for name, source in self.files.items():
            if name not in outlined:
                continue
            found, missing = defaultdict(set), []
            suffix = PurePosixPath(name).suffix.lower()
            if suffix in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"):
                self.includes(name, source, found, missing)
            elif suffix == ".cs":
                for space in CSHARP_USING.findall(source):
                    targets = [other for other in spaces.get(space, []) if other != name
                               and any(re.search(rf"\b{re.escape(symbol)}\b", source) for symbol in outlined.get(other, {}))]
                    for other in targets:
                        found[other].update(symbol for symbol in outlined[other] if re.search(rf"\b{re.escape(symbol)}\b", source))
                    if not spaces.get(space) and not space.startswith("System"):
                        missing.append(self.note(space, space.rsplit(".", 1)[-1], source, space))
            elif suffix == ".rb":
                for kind, path in RUBY_REQUIRE.findall(source):
                    target = self.ruby(name, kind, path)
                    if target:
                        found[target].add("")
                    elif kind == "require_relative":
                        missing.append({**self.note(path, PurePosixPath(path).name, source, path), "local": True})
                    else:
                        missing.append(self.note(path, path.split("/")[0], source, path))
            elif suffix == ".php":
                for path in PHP_INCLUDE.findall(source):
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), path))
                    if target in self.names:
                        found[target].add("")
                for qualified, alias in PHP_USE.findall(source):
                    space, _, symbol = qualified.replace("\\", ".").rpartition(".")
                    targets = [other for other in spaces.get(space, []) if symbol in outlined.get(other, {})]
                    for other in targets:
                        found[other].add(symbol)
                    if not targets:
                        missing.append(self.note(qualified, alias or symbol, source, qualified))
            elif suffix in (".kt", ".kts"):
                for qualified, alias in KOTLIN_IMPORT.findall(source):
                    space, _, symbol = qualified.rpartition(".")
                    targets = [other for other in spaces.get(space, []) if symbol in outlined.get(other, {})]
                    for other in targets:
                        found[other].add(symbol)
                    if not targets and not qualified.startswith(("kotlin.", "java.", "javax.")):
                        missing.append(self.note(qualified, alias or symbol, source, qualified))
            elif suffix == ".swift":
                for module in SWIFT_IMPORT.findall(source):
                    if module not in STANDARD:
                        missing.append(self.note(module, module, source, module))
            self.edges[name] = {target: sorted(symbols) for target, symbols in sorted(found.items()) if target != name}
            self.missing[name] = missing

    @staticmethod
    def note(module, bound, source, needle):
        """One import of code that is not here, with the line it is on."""
        index = source.find(needle)
        return {"module": module, "name": bound, "line": source.count("\n", 0, max(0, index)) + 1, "local": False}

    def includes(self, name, source, found, missing):
        """Follow #include "file" to a project file: beside the includer first, else the one file ending in that path."""
        for local, system in INCLUDE.findall(source):
            path = local or system
            beside = posixpath.normpath(posixpath.join(posixpath.dirname(name), path))
            target = beside if beside in self.names else None
            if target is None:
                ends = [other for other in self.names if other == path or other.endswith("/" + path)]
                target = ends[0] if len(ends) == 1 else None
            if target:
                found[target].add("")
            elif local:
                missing.append({**self.note(path, PurePosixPath(path).stem, source, path), "local": True})
            # A <header> that is not in the project is the standard library or a system package, known code.

    def ruby(self, name, kind, path):
        """The project file a Ruby require names, or None."""
        path = path if path.endswith(".rb") else path + ".rb"
        if kind == "require_relative":
            candidates = [posixpath.normpath(posixpath.join(posixpath.dirname(name), path))]
        else:
            candidates = [path, "lib/" + path]
        return next((candidate for candidate in candidates if candidate in self.names), None)
