"""Bounded UTF-8 workspace access and reviewable file writes."""
import ast
import difflib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from .exclusions import Exclusions, IgnoreConfigurationError

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000
SEARCH, DIVIDER, REPLACE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"


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
    methods), a module-level variable, a route path such as /health, or "imports". Other files match the
    first line that defines or mentions the name.
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
        if tree is not None:
            return python_symbol(tree, text, symbol)
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


def python_syntax_error(name, content, before=None):
    """Describe a syntax error a change introduces into a Python file. Parses only; never executes."""
    if Path(name).suffix not in (".py", ".pyi"):
        return None
    try:
        ast.parse(content, filename=name)
        return None
    except SyntaxError as exc:
        error = exc
    except (ValueError, RecursionError, MemoryError):
        return None
    if before is not None:
        try:
            ast.parse(before, filename=name)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None  # Already broken before this change; do not block a repair attempt.
    line = (error.text or "").strip()
    return (f"Proposed {name} has a Python syntax error at line {error.lineno}: {error.msg}"
            + (f" ({line[:120]})" if line else "") + ". Nothing was previewed or written; fix the change and try again.")


# A comment or line standing in for code the model left out, as small models do when asked for a whole file.
PLACEHOLDER = re.compile(r"^\s*(?:#|//|/\*|<!--|--)?\s*(?:\.\.\.|…)\s*(?:\w+\s+){0,3}"
                         r"(?:rest|remaining|other|existing|unchanged|previous|same|more)\b"
                         r"|^\s*(?:#|//|/\*|<!--|--)\s*(?:the\s+)?(?:rest|remainder)\s+of\s+(?:the\s+)?(?:file|code|class|module)\b",
                         re.I | re.M)
SHRINK_MIN_LINES = 20


def write_problem(name, before, content, task=""):
    """Why a whole-file write would lose code the task did not ask to remove, or None."""
    if before is None:
        return None
    if PLACEHOLDER.search(content) and not PLACEHOLDER.search(before):
        return (f"Proposed {name} contains a placeholder for omitted code, which would delete that code. "
                "Nothing was previewed or written; use edit to change only the lines that need it.")
    old, new = len(before.splitlines()), len(content.splitlines())
    if old >= SHRINK_MIN_LINES and new < old / 2 and not re.search(
            r"\b(?:rewrite|replace|overwrite|truncate|shrink|shorten|simplify|trim|clear|empty|remove|delete)\b", task, re.I):
        return (f"Writing {name} would remove {old - new} of its {old} lines, and the task did not ask for that. "
                "Nothing was previewed or written; use edit to change only the lines that need it.")
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


def closest_text(text, wanted):
    """Point at the file lines most like a SEARCH text that did not match, so the next attempt can copy them."""
    lines = text.splitlines()
    want = [line.strip() for line in wanted.splitlines() if line.strip()]
    if not want or not lines:
        return ""
    close = difflib.get_close_matches(want[0], [line.strip() for line in lines], n=1, cutoff=0.5)
    if not close:
        return ""
    number = next(i for i, line in enumerate(lines) if line.strip() == close[0])
    shown = "\n".join(lines[number:number + max(1, min(len(want), 6))])
    return f" Closest text in the file, at line {number + 1}, to copy exactly:\n{shown[:600]}"


SEARCH_LINE = re.compile(r"\s*<{3,}\s*SEARCH\s*")
DIVIDER_LINE = re.compile(r"\s*={3,}\s*")
REPLACE_LINE = re.compile(r"\s*>{3,}\s*(?:REPLACE)?\s*")
SIMILAR_LINE, SIMILAR_REGION = 0.6, 0.75


def indent_of(line):
    return line[:len(line) - len(line.lstrip())]


def reindented(lines, have, wanted):
    """Move lines from indentation wanted to indentation have, leaving blank lines alone."""
    if have == wanted:
        return list(lines)
    return [have + line.removeprefix(wanted) if line.strip() else line for line in lines]


def loose_replace(text, old, new, newline):
    """Replace old ignoring indentation and trailing spaces, when exactly one place in the file matches."""
    want = [line.strip() for line in old.split(newline)]
    while want and not want[-1]:
        want.pop()
    while want and not want[0]:
        want.pop(0)
    lines = text.splitlines(keepends=True)
    stripped = [line.strip() for line in lines]
    hits = [i for i in range(len(lines) - len(want) + 1) if want and stripped[i:i + len(want)] == want]
    if len(hits) != 1:
        return None
    first = next(line for line in old.split(newline) if line.strip())
    start, end = hits[0], hits[0] + len(want)
    replacement = newline.join(reindented(new.split(newline), indent_of(lines[start]), indent_of(first)))
    ending = lines[end - 1][len(lines[end - 1].rstrip("\r\n")):]
    return "".join(lines[:start]) + replacement + ending + "".join(lines[end:])


def apply_edits(text, blocks):
    """Apply SEARCH/REPLACE blocks in order; each SEARCH text must match exactly once.

    Prose and code fences around the blocks are ignored. A SEARCH text that matches nowhere exactly may still
    match one place once indentation is ignored, because small models often lose indentation when copying.
    """
    lines = blocks.split("\n")
    edits, index = [], 0

    def marker(pattern, start):
        found = next((i for i in range(start, len(lines)) if pattern.fullmatch(lines[i])), None)
        if found is None:
            raise ValueError(f"Each edit block needs {DIVIDER} and {REPLACE} marker lines")
        return found

    if blocks.strip() and not any(SEARCH_LINE.fullmatch(line) for line in lines):
        raise ValueError(f"Edit content must contain only {SEARCH} ... {REPLACE} blocks")
    while index < len(lines):
        if not SEARCH_LINE.fullmatch(lines[index]):
            index += 1
            continue
        divider = marker(DIVIDER_LINE, index + 1)
        end = marker(REPLACE_LINE, divider + 1)
        edits.append(("\n".join(lines[index + 1:divider]), "\n".join(lines[divider + 1:end])))
        index = end + 1
    if not edits:
        raise ValueError("Edit needs at least one SEARCH/REPLACE block")
    newline = "\r\n" if "\r\n" in text else "\n"
    for number, (old, new) in enumerate(edits, 1):
        if not old.strip():
            raise ValueError(f"Edit block {number} has empty SEARCH text; use write to create a file")
        old, new = old.replace("\n", newline), new.replace("\n", newline)
        count = text.count(old)
        if count == 0:
            loose = loose_replace(text, old, new, newline)
            if loose is not None:
                text = loose
                continue
        if count != 1:
            hint = "read the file and copy the current lines exactly" if not count else "include more surrounding lines so it matches once"
            raise ValueError(f"Edit block {number} SEARCH text matches {count} times; {hint}. No change was made."
                             + (closest_text(text, old) if not count else ""))
        text = text.replace(old, new, 1)
    return text


def names_in(text):
    return set(re.findall(r"[\w.-]+", text))


def name_weights(stripped):
    """Weight each name by how rare it is in the file: Founders says far more about a line than Day does."""
    counts = Counter(name for line in stripped for name in names_in(line))
    return {name: 1 / count for name, count in counts.items()}


def shared_weight(names, line, weights):
    return sum(weights.get(name, 0) for name in names & names_in(line))


def check_placement(stripped, pairs):
    """Refuse when a changed line is put over one line but clearly belongs at another.

    Similar neighbours must not decide where a change lands: "Founders Day" written beside the last row of a
    table would otherwise overwrite that row instead of updating the Founders Day row.
    """
    weights = name_weights(stripped)
    for here, line in pairs:
        names = names_in(line)
        scores = [shared_weight(names, other, weights) if index != here else -1 for index, other in enumerate(stripped)]
        top = max(scores, default=0)
        # Equally rare names, such as a row's label and its new value: the more similar line wins.
        tied = [index for index, score in enumerate(scores) if score == top]
        best = max(tied, key=lambda index: difflib.SequenceMatcher(None, stripped[index], line).ratio(), default=None)
        if best is not None and top > shared_weight(names, stripped[here], weights):
            raise ValueError(f"Edit content puts {line[:80]!r} at line {here + 1}, but it looks like it belongs at line {best + 1}. "
                             f"No change was made; to change that line, send only its new version. Line {best + 1} is now:\n{stripped[best][:300]}")


def merge_fragment(text, fragment):
    """Put a new version of part of a file in place, and return the whole updated file.

    Small models tend to answer an edit by writing out the changed code rather than SEARCH/REPLACE blocks, and
    the code is usually right. Lines shared with the file locate the region; the rest of the file is untouched.
    Raises ValueError when the place is ambiguous or the change would remove much more than it shows.
    """
    if PLACEHOLDER.search(fragment) and not PLACEHOLDER.search(text):
        raise ValueError("Edit content contains a placeholder for omitted code. No change was made; "
                         "write out only the changed lines with a few unchanged lines around them.")
    frag = [line.rstrip("\r") for line in fragment.split("\n")]
    frag = [line for line in frag if not re.fullmatch(r"\s*```\w*\s*", line)]
    while frag and not frag[-1].strip():
        frag.pop()
    while frag and not frag[0].strip():
        frag.pop(0)
    if not frag:
        raise ValueError("Edit needs SEARCH/REPLACE blocks or the new version of the lines to change")
    lines = text.splitlines()
    newline = "\r\n" if "\r\n" in text else "\n"
    ending = newline if text.endswith(("\n", "\r")) else ""
    stripped, wanted = [line.strip() for line in lines], [line.strip() for line in frag]
    matcher = difflib.SequenceMatcher(None, stripped, wanted, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks()
              if block.size and any(wanted[block.b:block.b + block.size])]
    unplaced = ("Edit content does not match any one place in the file. No change was made; use SEARCH/REPLACE "
                "blocks copied exactly from the file, or read the lines to change first.")
    if not blocks:
        start, end = closest_region(stripped, wanted, unplaced)
        return newline.join(lines[:start] + reindented(frag, indent_of(lines[start]), indent_of(frag[0])) + lines[end:]) + ending
    largest = max(blocks, key=lambda block: block.size)
    run = stripped[largest.a:largest.a + largest.size]
    if sum(stripped[i:i + len(run)] == run for i in range(len(stripped) - len(run) + 1)) > 1:
        raise ValueError("Edit content matches several places in the file. No change was made; "
                         "include more surrounding lines so it fits only one place.")
    shared = sum(1 for block in blocks for line in wanted[block.b:block.b + block.size] if line)
    if shared * 2 < sum(1 for line in wanted if line):
        raise ValueError(unplaced)
    start, end = blocks[0].a, blocks[-1].a + blocks[-1].size
    # Changed lines before the first or after the last shared line replace the lines they resemble.
    for line in reversed(wanted[:blocks[0].b]):
        if start and difflib.SequenceMatcher(None, stripped[start - 1], line).ratio() >= SIMILAR_LINE:
            start -= 1
        else:
            break
    for line in wanted[blocks[-1].b + blocks[-1].size:]:
        if end < len(lines) and difflib.SequenceMatcher(None, stripped[end], line).ratio() >= SIMILAR_LINE:
            end += 1
        else:
            break
    pairs = list(zip(range(blocks[0].a - blocks[0].b, blocks[0].a), wanted[:blocks[0].b]))
    pairs = [(here, line) for here, line in pairs if here >= start]
    for left, right in zip(blocks, blocks[1:]):
        pairs += zip(range(left.a + left.size, right.a), wanted[left.b + left.size:right.b])
    last = blocks[-1]
    pairs += zip(range(last.a + last.size, end), wanted[last.b + last.size:])
    check_placement(stripped, [(here, line) for here, line in pairs if line])
    matched = sum(block.size for block in blocks)
    removed = end - start - matched
    if removed > max(3, len(frag)):
        raise ValueError(f"Edit content would remove {removed} lines it does not show. No change was made; "
                         "use SEARCH/REPLACE blocks for this change.")
    merged = reindented(frag, indent_of(lines[blocks[0].a]), indent_of(frag[blocks[0].b]))
    for block in blocks:  # Shared lines stay byte-for-byte as they are in the file.
        merged[block.b:block.b + block.size] = lines[block.a:block.a + block.size]
    return newline.join(lines[:start] + merged + lines[end:]) + ending


def closest_region(stripped, wanted, unplaced):
    """Locate a short fragment that shares no whole line with the file: the one region clearly most like it."""
    if len(wanted) > 5 or len(wanted) > len(stripped):
        raise ValueError(unplaced)
    text = "\n".join(wanted)
    names = names_in(text)
    weights = name_weights(stripped)
    scores = []
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(text)
    for start in range(len(stripped) - len(wanted) + 1):
        region = "\n".join(stripped[start:start + len(wanted)])
        matcher.set_seq1(region)
        if matcher.real_quick_ratio() >= SIMILAR_REGION and matcher.quick_ratio() >= SIMILAR_REGION:
            ratio = matcher.ratio()
            if ratio >= SIMILAR_REGION:
                # Whole shared names come first: SKU-0042 and SKU-0142 are nearly the same text but different rows.
                scores.append((round(shared_weight(names, region, weights), 6), ratio, start))
    scores.sort(reverse=True)
    if not scores:
        raise ValueError(unplaced)
    if len(scores) > 1 and scores[0][0] == scores[1][0] and scores[0][1] - scores[1][1] < 0.02:
        raise ValueError(unplaced)
    return scores[0][2], scores[0][2] + len(wanted)


def apply_change(text, content):
    """Apply an edit action's content: SEARCH/REPLACE blocks when marked, otherwise a new version of some lines."""
    if any(SEARCH_LINE.fullmatch(line) for line in content.split("\n")):
        return apply_edits(text, content)
    return merge_fragment(text, content)


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

    def propose(self, name, content):
        if not isinstance(content, str) or len(content.encode()) > MAX_BYTES or "\0" in content:
            raise ValueError("Proposed content must be UTF-8 text up to 64 KB")
        target = self.path(name)
        if target.exists() and target.stat().st_nlink > 1:
            raise ValueError("Hard-linked files cannot be edited")
        before = self.read(name) if target.exists() else None
        diff = "".join(difflib.unified_diff((before or "").splitlines(True),
                       content.splitlines(True), fromfile=name if before is not None else "/dev/null",
                       tofile=name))
        if not diff and before != content:
            diff = f"Create empty file: {name}\n"
        return before, diff

    def write(self, name, content, expected):
        self.propose(name, content)
        target = self.path(name)
        current = self.read(name) if target.exists() else None
        if current != expected:
            raise ValueError("File changed after preview; request a fresh proposal")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")  # Bytes, so Windows newline translation cannot change the file's line endings.
        if current is None:
            handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            return
        # Replace atomically, so a crash or full disk cannot leave a half-written file behind.
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".codehaven-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
