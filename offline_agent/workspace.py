"""Bounded, read-only UTF-8 workspace access: listing, reading, paging, and literal search. Nothing here writes files."""
import ast
import json
import os
import re
from pathlib import Path
from .exclusions import Exclusions, IgnoreConfigurationError

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000


def line_range(spec):
    """Parse an optional read range such as 120-180 or 120; anything else means the whole file."""
    match = re.fullmatch(r"\s*(?:lines?\s*)?(\d+)(?:\s*[-:]\s*(\d+))?\s*", spec or "", re.I)
    if not match:
        return None
    start, end = int(match[1]), int(match[2]) if match[2] else None
    if start < 1 or (end is not None and end < start):
        raise ValueError("Read range must be start-end with 1 <= start <= end")
    return start, end


def symbol_lines(name, text, symbol):
    """Line range (start, end) of a named part of a file, or None when nothing by that name is defined.

    Python files are parsed, never executed: a function or class (decorators included, Class.method for
    methods), a module-level variable, a route path such as /health, or "imports". Anything else, in any file,
    matches the first line that mentions the name.
    """
    symbol = re.sub(r"^(?:async\s+def|def|class)\s+", "", symbol.strip().strip("`'\" ")).removesuffix("()").strip()
    symbol = re.sub(r"\[[^\]]*\]$", "", symbol)  # CATALOG[123] names a part of CATALOG.
    if not symbol:
        return None
    if Path(name).suffix in (".py", ".pyi"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        found = python_symbol(tree, text, symbol) if tree is not None else None
        if found is not None:
            return found
        # Not a definition, such as a row in a data table: fall through to the first line that mentions it.
    pattern = re.compile(rf"(?<![\w.]){re.escape(symbol)}(?![\w])")
    for number, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            return number, number
    return None


def python_symbol(tree, text, symbol):
    if symbol.lower() == "imports":
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        return (imports[0].lineno, imports[-1].end_lineno) if imports else None
    owner, _, member = symbol.rpartition(".")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scopes = ([c.body for c in classes if c.name == owner] if owner else [tree.body] + [c.body for c in classes])
    route = re.compile(rf"[\"']{re.escape(symbol)}[\"']") if symbol.startswith("/") else None
    for node in (node for body in scopes for node in body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = [ast.get_source_segment(text, d) or "" for d in node.decorator_list]
            if node.name == member or (route and any(route.search(d) for d in decorators)):
                return min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == member for t in targets):
                return node.lineno, node.end_lineno
    return None


def sections(text, limit, focus=(), overhead=0):
    """Line ranges (1-based, inclusive) to show of a file too long for limit bytes.

    Shows the start and the end of the file, plus a few lines around each focus line, because what a task
    asks about is often not in the first page. overhead is extra bytes per line, such as a line number.
    """
    lines = text.splitlines(keepends=True)
    cost = [len(line.encode()) + overhead for line in lines]
    if sum(cost) <= limit:
        return [(1, max(1, len(lines)))]
    chosen, used = set(), 0

    def take(indexes, budget):
        nonlocal used
        spent = 0
        for index in indexes:
            if index in chosen:
                continue
            if spent + cost[index] > budget:
                return
            chosen.add(index)
            spent += cost[index]
            used += cost[index]

    if focus:
        window = [index for line in focus for index in range(max(0, line - 4), min(len(lines), line + 3))]
        take(window, limit * 3 // 10)
    remaining = limit - used
    take(range(len(lines)), remaining * 3 // 4)
    take(reversed(range(len(lines))), limit - used)
    if not chosen:
        chosen.add(0)  # A single line longer than the limit still gets shown; callers mark it incomplete.
    ranges = []
    for index in sorted(chosen):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index + 1, index + 1])
    return [tuple(pair) for pair in ranges]


DEFINITION = re.compile(r"^(\s*)(?:[\w@.<>\[\],\s]*\s)?(?:function|class|def|func|fn|interface|struct|impl)\s+([\w$]+)")


def enclosing_parts(name, text):
    """Map a function: line number -> (enclosing name, whether that line is the definition itself)."""
    spans = []
    if Path(name).suffix in (".py", ".pyi"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
        if tree is not None:
            def visit(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        qualified = prefix + child.name
                        spans.append((child.lineno, child.end_lineno, qualified))
                        visit(child, qualified + ".")
                    else:
                        visit(child, prefix)
            visit(tree, "")

            def lookup(number):
                inside = [span for span in spans if span[0] <= number <= span[1]]
                if not inside:
                    return None, False
                innermost = max(inside, key=lambda span: span[0])
                return innermost[2], innermost[0] == number
            return lookup
    lines = text.splitlines()

    def lookup(number):
        for index in range(min(number, len(lines)) - 1, -1, -1):
            found = DEFINITION.match(lines[index])
            if found:
                return found[2], index == number - 1
        return None, False
    return lookup


class Workspace:
    def __init__(self, root, *, excludes=()):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")
        self.excludes = tuple(excludes)
        self.exclusions = Exclusions(self.root, self.excludes)

    def path(self, name):
        rel = Path(name)
        if not name or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Use a relative path inside the workspace")
        if any(p in EXCLUDED or p.startswith(".") for p in rel.parts):
            raise ValueError("Hidden and excluded paths are unavailable")
        target = self.root / rel
        current = self.root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlink paths are unavailable")
        target.resolve().relative_to(self.root)
        if self.exclusions.ignored(rel, directory=target.is_dir()):
            raise ValueError(f"Path is excluded by ignore rules: {name}")
        return target

    def files(self, scope=""):
        start = self.path(scope) if scope else self.root
        if not start.is_dir():
            raise ValueError("File listing scope must be a directory")
        found = []
        def visible_directory(path):
            try:
                self.path(path.relative_to(self.root).as_posix())
                return True
            except IgnoreConfigurationError:
                raise
            except (ValueError, OSError):
                return False

        for base, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = sorted(d for d in dirs if visible_directory(Path(base) / d))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                rel = (Path(base) / name).relative_to(self.root).as_posix()
                try:
                    path = self.path(rel)
                    if path.is_file() and path.stat().st_size <= MAX_BYTES:
                        found.append(rel)
                except IgnoreConfigurationError:
                    raise
                except (ValueError, OSError):
                    continue
                if len(found) >= 1000:
                    return found
        return found

    def read(self, name):
        target = self.path(name)
        if not target.is_file():
            raise ValueError("Only regular files can be read")
        with target.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or b"\0" in raw:
            raise ValueError("Only UTF-8 text files up to 64 KB can be read")
        return raw.decode("utf-8")

    def page(self, name, spec="", limit=MAX_BYTES):
        """Return (text, start, end, total) for a line range, cut at a line boundary to fit limit bytes."""
        text = self.read(name)
        lines = text.splitlines(keepends=True)
        total = len(lines)
        requested = line_range(spec)
        if requested is None and len(text.encode()) <= limit:
            return text, 1, max(1, total), total
        start, end = requested or (1, None)
        if start > total:
            raise ValueError(f"{name} has {total} lines; request a range within 1-{total}")
        shown, size = [], 0
        for line in lines[start - 1:min(end or total, total)]:
            size += len(line.encode())
            if shown and size > limit:
                break
            shown.append(line)
        return "".join(shown), start, start + len(shown) - 1, total

    def search(self, query, scope=""):
        """Case-sensitive literal search; return bounded, numbered UTF-8 snippets."""
        if (not isinstance(query, str) or not query.strip() or len(query) > 256
                or any(c in query for c in "\n\r\0")):
            raise ValueError("Search requires a nonblank single-line literal of at most 256 characters")
        if scope:
            target = self.path(scope)
            if not target.exists():
                raise ValueError("Search scope does not exist")
            candidates = [scope] if target.is_file() else self.files(scope)
        else:
            candidates = self.files()
        result = {"query": query, "scope": scope or ".", "literal": True,
                  "matches": [], "files_searched": 0, "files_skipped": 0,
                  "limit_reached": False, "candidate_limit_reached": len(candidates) >= 1000}
        for name in candidates:
            try:
                text = self.read(name)
            except (ValueError, OSError, UnicodeError):
                result["files_skipped"] += 1
                continue
            result["files_searched"] += 1
            lookup = None
            for number, line in enumerate(text.splitlines(), 1):
                index = line.find(query)
                if index < 0:
                    continue
                start = max(0, index - 40)
                snippet = line[start:start + 320]
                match = {"path": name, "line": number, "text": snippet,
                         "snippet_truncated": start > 0 or start + len(snippet) < len(line)}
                lookup = lookup or enclosing_parts(name, text)
                # Name the surrounding function, so "who calls X?" is answered by the result itself.
                part, defines = lookup(number)
                if part:
                    match["in"] = part
                    if defines:
                        match["defines"] = True
                if len(result["matches"]) >= 50:
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
                result["matches"].append(match)
                # Reserve room for counters and the truncation flag on later iterations.
                if len(json.dumps(result, ensure_ascii=False).encode()) > 15_800:
                    result["matches"].pop()
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
        if not result["matches"]:
            result["note"] = ("No matches for this exact literal and scope among searched files. "
                              "This does not establish that a framework is absent. "
                              "Do not append file extensions or globs to a source-text query.")
        return json.dumps(result, ensure_ascii=False)
