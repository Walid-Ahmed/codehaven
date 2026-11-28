"""Bounded, read-only UTF-8 workspace access: listing, reading, paging, and literal search. Nothing here writes files."""
import ast
import errno
import json
import os
import re
import stat
from pathlib import Path, PurePath
from .exclusions import Exclusions, IgnoreConfigurationError

# Directory names that are never listed or read: version control, environments, build output, and credential folders.
EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000  # The largest file uploaded, read from a picture or document, and one page of a read.
# The largest project file listed, searched, indexed, or read in pages. Only bounded parts of a file ever reach a
# prompt, so this limits memory, not prompt size. It was MAX_BYTES until large projects showed that their central
# modules (FastAPI's routing.py and applications.py, about 176 KB each) were invisible to search and reads.
READ_BYTES = 1_000_000
MAX_FILES = 5000  # The most files one listing returns; callers report the listing as limited at this size.
MAX_MATCHES, SEARCH_BYTES = 50, 15_800
# Opening a file one directory at a time, refusing symlinks at each step, needs these; Windows lacks them.
SAFE_OPEN = all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")) and os.open in os.supports_dir_fd  # Bounds on one search result, so it always fits in a prompt.


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
    lines = text.splitlines()
    pattern = re.compile(rf"(?<![\w.]){re.escape(symbol)}(?![\w])")
    for number, line in enumerate(lines, 1):
        if pattern.search(line):
            return number, number
    # A member defined by assignment, as JavaScript modules do: exports.compileETag = function ..., app.listen = ...
    # The dot before the name kept these from matching above, so every exported function looked "not found".
    member = symbol.rsplit(".", 1)[-1]
    assigned = re.compile(rf"\.{re.escape(member)}\s*=(?!=)")
    for number, line in enumerate(lines, 1):
        if assigned.search(line):
            return number, number
    return None


def python_symbol(tree, text, symbol):
    """Line range of a named part of a parsed Python module, or None: see symbol_lines for what can be named."""
    if symbol.lower() == "imports":
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        return (imports[0].lineno, imports[-1].end_lineno) if imports else None
    owner, _, member = symbol.rpartition(".")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    # Class.method looks only inside that class; a bare name looks at module level, then inside every class.
    scopes = ([c.body for c in classes if c.name == owner] if owner else [tree.body] + [c.body for c in classes])
    # A name starting with / is a route path, found as a quoted string in a decorator such as @app.get("/health").
    route = re.compile(rf"[\"']{re.escape(symbol)}[\"']") if symbol.startswith("/") else None
    for node in (node for body in scopes for node in body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = [ast.get_source_segment(text, d) or "" for d in node.decorator_list]
            if node.name == member or (route and any(route.search(d) for d in decorators)):
                # Start at the first decorator, so the range covers the whole definition.
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
        """Choose lines in the given order until budget bytes are spent, skipping lines already chosen."""
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
        # Three lines either side of each focus line, using at most 30% of the limit.
        window = [index for line in focus for index in range(max(0, line - 4), min(len(lines), line + 3))]
        take(window, limit * 3 // 10)
    remaining = limit - used
    # Three quarters of what is left goes to the start of the file, and the rest to its end.
    take(range(len(lines)), remaining * 3 // 4)
    take(reversed(range(len(lines))), limit - used)
    if not chosen:
        chosen.add(0)  # A single line longer than the limit still gets shown; callers mark it incomplete.
    # Merge neighbouring line indexes into 1-based inclusive ranges.
    ranges = []
    for index in sorted(chosen):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index + 1, index + 1])
    return [tuple(pair) for pair in ranges]


# A definition line in common languages: optional modifiers, a keyword, then the name (group 2).
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
                """Record the line span and dotted name of every function and class, however deeply nested."""
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        qualified = prefix + child.name
                        spans.append((child.lineno, child.end_lineno, qualified))
                        visit(child, qualified + ".")
                    else:
                        visit(child, prefix)
            visit(tree, "")

            def lookup(number):
                """The innermost function or class that contains the line."""
                inside = [span for span in spans if span[0] <= number <= span[1]]
                if not inside:
                    return None, False
                innermost = max(inside, key=lambda span: span[0])
                return innermost[2], innermost[0] == number
            return lookup
    lines = text.splitlines()

    def lookup(number):
        """The nearest definition line at or above the line: a heuristic for other languages and unparsable Python."""
        for index in range(min(number, len(lines)) - 1, -1, -1):
            found = DEFINITION.match(lines[index])
            if found:
                return found[2], index == number - 1
        return None, False
    return lookup


def combined(roots, *, excludes=()):
    """One workspace for several repository folders: their common parent, limited to them.

    A single folder, or a folder that holds all the others, is opened as it is."""
    paths = [Path(root).expanduser().resolve(strict=True) for root in roots]
    common = Path(os.path.commonpath([str(path) for path in paths]))
    if len(paths) == 1 or common in paths:
        return Workspace(common, excludes=excludes)
    return Workspace(common, excludes=excludes, scopes=[path.relative_to(common).as_posix() for path in paths])


class Workspace:
    """Read-only view of one directory tree. Every name is checked by path() before it is listed, read, or searched."""

    def __init__(self, root, *, excludes=(), scopes=()):
        """Resolve the root once, so later checks compare against the real directory, and load the ignore rules.

        scopes, folders under root such as ("api", "web"), limits the workspace to them: several repositories
        are opened together as the folders of their common parent, and nothing else under that parent is seen.
        """
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")
        # The root as a string, and its case-folded form with a trailing separator: path() works on strings, because
        # pathlib's own symlink, resolve, and relative_to checks cost seconds per listing on a large project.
        self.root_text = str(self.root)
        self.root_key = os.path.normcase(self.root_text)
        self.excludes = tuple(excludes)
        self.scopes = tuple(sorted({PurePath(scope).as_posix().strip("/") for scope in scopes if str(scope).strip("/.")}))
        self.exclusions = Exclusions(self.root, self.excludes)
        for scope in self.scopes:
            if not self.path(scope).is_dir():
                raise ValueError(f"{scope} is not a folder under {self.root}")

    def path(self, name):
        """Return the real path for a workspace-relative name.

        Raises ValueError when the name would leave the root, or is hidden, excluded, reached through a symlink,
        or ignored by the ignore rules. This is the one gate every file access goes through.
        """
        rel = PurePath(name) if name else None
        # Path traversal: no absolute paths and no .. components.
        if not name or rel.is_absolute() or rel.anchor or ".." in rel.parts:
            raise ValueError("Use a relative path inside the workspace")
        if any(p in EXCLUDED or p.startswith(".") for p in rel.parts):
            raise ValueError("Hidden and excluded paths are unavailable")
        if self.scopes:
            posix = "/".join(rel.parts)
            # Inside an opened repository, or a folder on the way to one; nothing else under the common parent.
            if not any(posix == scope or posix.startswith(scope + "/") or scope.startswith(posix + "/") for scope in self.scopes):
                raise ValueError(f"{name} is not in one of the opened repositories")
        # A symlink at any level could point outside the workspace, so each component is checked, not only the last.
        current = self.root_text
        for part in rel.parts:
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise ValueError("Symlink paths are unavailable")
        # Final guard: the fully resolved path must still be inside the root.
        real = os.path.normcase(os.path.realpath(current))
        if real != self.root_key and not real.startswith(self.root_key.rstrip(os.sep) + os.sep):
            raise ValueError("Use a relative path inside the workspace")
        if self.exclusions.ignored(rel, directory=os.path.isdir(current)):
            raise ValueError(f"Path is excluded by ignore rules: {name}")
        return self.root / rel

    def files(self, scope=""):
        """Relative paths of the readable files under scope, in sorted walk order, at most MAX_FILES of them.

        Hidden, excluded, ignored, symlinked, and oversized files are left out.
        """
        if self.scopes and (not scope or any(other.startswith(scope.strip("/") + "/") for other in self.scopes)):
            # Several repositories: each one's files, in the order the repositories were given, within the one bound.
            found = []
            for other in self.scopes:
                if not scope or other.startswith(scope.strip("/") + "/"):
                    found += self.files(other)[:MAX_FILES - len(found)]
            return found
        start = self.path(scope) if scope else self.root
        if not start.is_dir():
            raise ValueError("File listing scope must be a directory")
        found = []

        def relative(base, name):
            """The workspace-relative POSIX name of name inside the walked directory base."""
            inside = base[len(self.root_text):].strip(os.sep)
            return (inside.replace(os.sep, "/") + "/" + name) if inside else name

        def visible_directory(base, name):
            """Whether a directory passes path(). A broken ignore file is an error to report, not a reason to hide it."""
            try:
                self.path(relative(base, name))
                return True
            except IgnoreConfigurationError:
                raise
            except (ValueError, OSError):
                return False

        for base, dirs, files in os.walk(start, followlinks=False):
            # Assigning in place prunes the walk, so an excluded directory is never entered.
            dirs[:] = sorted(d for d in dirs if visible_directory(base, d))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                rel = relative(base, name)
                try:
                    status = os.stat(self.path(rel))
                    if stat.S_ISREG(status.st_mode) and status.st_size <= READ_BYTES:
                        found.append(rel)
                except IgnoreConfigurationError:
                    raise
                except (ValueError, OSError):
                    continue
                if len(found) >= MAX_FILES:
                    return found
        return found

    def read_bytes(self, name, limit=READ_BYTES):
        """Return (the first limit + 1 bytes of a file, its mode), opened so that no symlink is ever followed.

        path() checks the name first. The checks and the open are separate steps, so a directory swapped for a
        symlink in between could otherwise lead the open out of the workspace. On macOS and Linux the file is
        therefore opened one component at a time from the root with O_NOFOLLOW, which refuses a symlink at any
        level at the moment of opening. Windows has no such open, and uses the checks alone.
        """
        self.path(name)
        parts = PurePath(name).parts
        if SAFE_OPEN:
            parent = os.open(self.root_text, os.O_RDONLY | os.O_DIRECTORY)
            try:
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    os.close(parent)
                    parent = child
                # O_NONBLOCK stops a named pipe from hanging the open; fstat below rejects anything but a regular file.
                descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValueError("Symlink paths are unavailable") from None
                raise
            finally:
                os.close(parent)
        else:
            target = self.root / name
            if not target.is_file():
                raise ValueError("Only regular files can be read")
            descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Only regular files can be read")
            # One byte past the limit is enough to know the file is too large, without reading all of it.
            return handle.read(limit + 1), info.st_mode

    def read(self, name):
        """Return a file's text. Raises ValueError for anything but a UTF-8 text file of at most READ_BYTES."""
        raw, _ = self.read_bytes(name)
        if len(raw) > READ_BYTES or b"\0" in raw:
            raise ValueError(f"Only UTF-8 text files up to {READ_BYTES // 1_000_000} MB can be read")
        # Windows line endings become plain ones, so a stray carriage return never reaches the model or a citation.
        # Nothing is written back: the file on disk keeps its own line endings.
        return raw.decode("utf-8").replace("\r\n", "\n")

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
            # The first line is always shown, even when it alone is over the limit.
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
                  "limit_reached": False, "candidate_limit_reached": len(candidates) >= MAX_FILES}
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
                # Show up to 40 characters before the match, and 320 characters in all.
                start = max(0, index - 40)
                snippet = line[start:start + 320]
                match = {"path": name, "line": number, "text": snippet,
                         "snippet_truncated": start > 0 or start + len(snippet) < len(line)}
                # Built once per file, and only for a file that has a match.
                lookup = lookup or enclosing_parts(name, text)
                # Name the surrounding function, so "who calls X?" is answered by the result itself.
                part, defines = lookup(number)
                if part:
                    match["in"] = part
                    if defines:
                        match["defines"] = True
                if len(result["matches"]) >= MAX_MATCHES:
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
                result["matches"].append(match)
                # Reserve room for counters and the truncation flag on later iterations.
                if len(json.dumps(result, ensure_ascii=False).encode()) > SEARCH_BYTES:
                    result["matches"].pop()
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
        if not result["matches"]:
            result["note"] = ("No matches for this exact literal and scope among searched files. "
                              "This does not establish that a framework is absent. "
                              "Do not append file extensions or globs to a source-text query.")
        return json.dumps(result, ensure_ascii=False)
