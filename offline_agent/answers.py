"""Checking and shaping what the model returns: a cited answer with looked-up line ranges, and readable tool output."""
import json
import re

from .evidence import CITATION
from .prompts import ACTIONS, ANSWER_CHARS, FINAL_ANSWER_CHARS, OVERVIEW_CHARS, POINT_CHARS, POINTS, SCHEMA
from .workspace import line_range, symbol_lines


# Notes added to an answer that was cut short, so the reader knows why it ends where it does.
REPEAT_NOTE = "[Answer stopped when it began repeating itself; only the text before the repetition is kept.]"
LIMIT_NOTE = "[Answer reached its length limit; it ends at its last complete sentence.]"


def trim_at_limit(text, limit):
    """End an answer cut by its schema's maxLength at its last complete sentence. Returns (text, trimmed).

    The cut lands mid-word, and Qwen often emits a stray Chinese token right at it.
    """
    # Shorter than the limit means the model stopped by itself, so there is nothing to trim.
    if not isinstance(limit, int) or len(text) < limit:
        return text, False
    # Cut after the last sentence end; with none, fall back to the last whole word and mark the cut with an ellipsis.
    end = max(text.rfind(mark) for mark in (". ", ".\n", "? ", "?\n", "! ", "!\n", ".)", ".`", "؟ ", "؟\n"))
    if end > 0:
        return text[:end + 1].rstrip(), True
    return text[:text.rstrip().rfind(" ")].rstrip() + " …" if " " in text.strip() else text, True


# Sent back to the model after an unusable reply; {} is the problem that reply_problem found.
CORRECTION = ("Your previous reply was invalid: {}. No tool ran. Reply again with one JSON object containing exactly "
              "action, path, content, and message as strings, where action is one of " + ", ".join(ACTIONS) + ".")


def reply_problem(reply, actions=ACTIONS):
    """Describe why a model reply is not a usable action, or return None. actions are the ones on offer."""
    if not isinstance(reply, dict):
        return "the reply is not a JSON object"
    missing = [key for key in SCHEMA["required"] if key not in reply]
    if missing:
        return "missing fields: " + ", ".join(missing)
    extra = sorted(str(key) for key in reply if key not in SCHEMA["required"])
    if extra:
        return "unexpected fields: " + ", ".join(extra)[:120]
    wrong = [key for key in SCHEMA["required"] if not isinstance(reply[key], str)]
    if wrong:
        return "fields must be strings: " + ", ".join(wrong)
    if reply["action"] not in actions:
        return f"unknown action {reply['action'][:40]!r}"
    return None


def with_actions(schema, actions):
    """A schema whose action field allows exactly actions; schema itself is left as it is."""
    return {**schema, "properties": {**schema["properties"], "action": {"type": "string", "enum": list(actions)}}}


def cited_schema(paths=(), brief=False):
    """Schema for questions and change requests: an answer, and points that each name a part of the code.

    Explaining named files gets a short overview and points limited to those files. brief is for a plain question,
    whose answer text is capped at ANSWER_CHARS; a change request keeps the room it needs to show new code.
    """
    # With named files, a point's path is limited to them, so the model cannot cite a file it was not shown.
    point = {"type": "object", "additionalProperties": False,
             "properties": {"path": {"type": "string", "enum": list(paths)} if paths else {"type": "string", "maxLength": 200},
                            "symbol": {"type": "string", "maxLength": 80},
                            "text": {"type": "string", "maxLength": POINT_CHARS}},
             "required": ["path", "symbol", "text"]}
    properties = {**SCHEMA["properties"],
                  "action": {"type": "string", "enum": list(ACTIONS)},
                  "content": {"type": "string", "maxLength": OVERVIEW_CHARS if paths else ANSWER_CHARS if brief else FINAL_ANSWER_CHARS},
                  "points": {"type": "array", "items": point, "maxItems": POINTS}}
    return {**SCHEMA, "properties": properties, "required": SCHEMA["required"] + ["points"]}


def retrieved_rows(data):
    """Evidence rows for a retrieval result: its passages, and the constant definitions shown with them."""
    rows = []
    for row in data["results"]:
        rows.append({"path": row["source"], "line": row["start_line"], "end_line": row["end_line"], "text": row["text"]})
        rows += [{"path": row["source"], "line": item["line"], "end_line": item["line"], "text": item["text"]}
                 for item in row.get("constants", ())]
    return rows


def seen_span(path, span, evidence, symbol=""):
    """Narrow a looked-up range to the lines the model saw, so a citation claims no more than was read."""
    # Keep the largest overlap between the looked-up range and any range of that file the model was shown.
    overlaps = []
    for row in evidence:
        if row["path"].removeprefix("repo/") == path:
            start, end = max(span[0], row["line"]), min(span[1], row.get("end_line", row["line"]))
            if start <= end:
                overlaps.append((end - start, start, end))
    if overlaps:
        _, start, end = max(overlaps)
        return start, end
    # A search match names its enclosing function, so citing that function can point at the match the model saw.
    named = next((row for row in evidence if row["path"].removeprefix("repo/") == path and symbol and row.get("in")
                  and row["in"].rsplit(".", 1)[-1] == symbol.rsplit(".", 1)[-1].removesuffix("()")), None)
    return (named["line"], named["line"]) if named else span


def row_text(row, workspace=None, files=None):
    """The text of the lines an evidence row covers.

    Retained rows keep only a short snippet, to stay small in the prompt, so the lines are read again from the
    workspace when it is given. files caches each file's lines for one answer. Reference documents outside the
    workspace, which cannot be read this way, give their snippet or nothing.
    """
    if row.get("text"):
        return str(row["text"])
    if workspace is None:
        return str(row.get("snippet") or "")
    path = row["path"].removeprefix("repo/")
    if files is None:
        files = {}
    if path not in files:
        try:
            files[path] = workspace.read(path).splitlines()
        except (ValueError, OSError, UnicodeError):
            files[path] = None
    lines = files[path]
    if lines is None:
        return str(row.get("snippet") or "")
    return "\n".join(lines[row["line"] - 1:row.get("end_line", row["line"])])


def observed_line(symbol, path, evidence, workspace=None, files=None):
    """(path, line) of a line the model was shown that contains the symbol's name, or None. The point's own file comes first.

    The model names the right thing but often the wrong place: the file that imports a constant instead of the one
    that sets it, or the line of an import. When the lines it was actually shown contain the name, citing one of
    those is both true and checkable, where the looked-up line would be flagged as never read. A dotted name such
    as req.secure is matched by its last part, which is how it appears where it is defined.
    """
    short = symbol.strip("`'\" ").removesuffix("()").rsplit(".", 1)[-1]
    if len(short) < 3:
        return None
    named = re.compile(rf"(?<![\w$]){re.escape(short)}(?![\w$])")
    rows = sorted(evidence, key=lambda row: row["path"].removeprefix("repo/") != path)  # Stable: the point's own file first.
    for row in rows:
        for offset, text in enumerate(row_text(row, workspace, files).splitlines()):
            if named.search(text) and row["line"] + offset <= row.get("end_line", row["line"]):
                return row["path"].removeprefix("repo/"), row["line"] + offset
    return None


# Code-like names in an answer: anything in backticks, or a word with an underscore, a dot between words, or camelCase.
MENTION = re.compile(r"`([^`\n]{2,60})`|(?<![\w.$/])([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+|[A-Za-z$]\w*_\w+|[a-z]+[A-Z]\w*)(?![\w$/])")
# The most citations added for one answer, and the heading they are listed under.
MENTIONS = 4
MENTION_HEADING = "Lines this answer refers to (cited by CodeHaven from the code it was shown):"


def cite_mentions(answer, workspace, evidence):
    """Citation lines for the code names an answer mentions, taken from the lines the model was shown.

    For an answer that cites nothing at all. The model often writes a correct sentence about checkout_total
    without a citation; pointing at the shown line that names it lets the reader check the answer, which is
    the point of the check. Names that appear on no shown line add nothing, so nothing is made up.
    """
    found, files = [], {}
    for quoted, bare in MENTION.findall(answer):
        name = (quoted or bare).strip()
        # File names and paths are not code parts; a call such as f(x) is looked up as f.
        name = re.sub(r"\(.*\)$", "", name)
        if not name or "/" in name or re.search(r"\.(py|js|ts|go|java|rs|md|json|txt)$", name) or " " in name:
            continue
        seen = observed_line(name, "", evidence, workspace, files)
        if seen and all(seen != (path, line) for path, line, _ in found):
            found.append((seen[0], seen[1], name))
        if len(found) >= MENTIONS:
            break
    return [f"- {path}:{line} `{name}`" for path, line, name in found]


def file_named(workspace, path, symbol, files):
    """The workspace file a point names, when the point is about a whole file rather than a part of one.

    Overviews often name files as their parts ("gpt.py"), sometimes beside the wrong folder. files() is a
    one-item cache of the workspace listing, filled only when a name has to be looked up across the project.
    """
    name = symbol.strip("`'\" ").removeprefix("repo/").removeprefix("./")
    # No symbol names no file; a symbol equal to the point's own path or file name means that file.
    if not name or name in (path, path.rsplit("/", 1)[-1]):
        return path if name else None
    # Otherwise it must look like a file: it has an extension, and a bare name has a lowercase one (Store.add is not a file).
    if not re.search(r"\.\w{1,8}$", name) or "/" not in name and not re.search(r"\.[a-z]{1,5}$", name):
        return None
    if not files:
        files.append(workspace.files())
    # Accepted only when exactly one workspace file matches the name.
    matches = [f for f in files[0] if f == name or f.endswith("/" + name)]
    return matches[0] if len(matches) == 1 else None


# Literal values an answer states as facts: a quoted string, or a number with two or more digits or a dot in it
# (0.08, 3.1.0, 302). Single digits and booleans are left out: they are often worked out rather than written in
# the code (cobra's ShellCompDirectiveError is 1 << iota), so a missing one would be a false alarm.
QUOTED_VALUE = re.compile(r"""(?<![\w])['"]([^'"\n]{2,40})['"](?![\w])""")
NUMBER_VALUE = re.compile(r"(?<![\w.:/-])(\d+\.\d+(?:\.\d+)*|\d{2,})(?![\w.]*\w)")
# A number right after one of these words is a line number the answer mentions, not a value from the code.
LINE_WORD = re.compile(r"\b(?:lines?(?:\s+(?:are|is|from|between))?|at|to|through|and)\s*$|-\s*$", re.I)
# Neither is the first number of a range such as "348-351", nor a percentage, which the code writes as a fraction.
NOT_A_VALUE_AFTER = re.compile(r"\s*(?:-\s*\d|%|percent\b)", re.I)
# Paths and file names are not values; citations already check those.
PATH_LIKE = re.compile(r"[/\\]|\.[A-Za-z]{1,5}$")
# How far around a cited range a value may sit: answers cite a definition and quote a value a line or two away.
VALUE_REACH = 3
VALUE_HEADING = "Where the values this answer reports are in the code (cited by CodeHaven):"


def literal_values(answer, task=""):
    """(kind, value) for each literal an answer reports, once each, without those the question itself states."""
    prose = CITATION.sub(" ", answer)
    found = []
    for value in QUOTED_VALUE.findall(prose):
        if not PATH_LIKE.search(value) and value.strip() == value:
            found.append(("text", value))
    for match in NUMBER_VALUE.finditer(prose):
        if not (LINE_WORD.search(prose[max(0, match.start() - 16):match.start()])
                or NOT_A_VALUE_AFTER.match(prose, match.end())):
            found.append(("number", match.group(1)))
    unique = []
    for kind, value in found:
        if (kind, value) not in unique and not contains(kind, value, task):
            unique.append((kind, value))
    return unique


def contains(kind, value, text):
    """True when value appears in text: a number not inside a longer number, a string exactly.

    Only digits bound a number, so 422 is found in HTTP_422_UNPROCESSABLE_ENTITY, a constant that holds it.
    """
    if kind == "number":
        return re.search(rf"(?<![\d.]){re.escape(value)}(?!\d|\.\d)", text) is not None
    return value in text


def check_values(answer, task, workspace, evidence):
    """Check each literal value an answer reports against the code. Returns (citation lines to add, invented values).

    A value on or near a line the answer cites is fine. A value the model was shown, but on a line the answer does
    not cite, gets a citation of the line it is on, so the answer can be checked where the value really is; nothing
    is flagged, because the answer is often right and only cited the wrong line. A value on no line the model was
    shown was not read from the code: it is returned as invented, and the answer is then marked unverified.
    The check cannot tell a value read from the wrong place, such as a default from a neighbouring class, from the
    right one; it confirms where a value comes from, not that it answers the question.
    """
    files = {}
    near = []
    for path, start, end in CITATION.findall(answer):
        row = {"path": path.removeprefix("repo/"), "line": max(1, int(start) - VALUE_REACH),
               "end_line": int(end or start) + VALUE_REACH}
        near.append(row_text(row, workspace, files))
    near_text = "\n".join(near)
    shown = []
    for row in evidence:
        text = row_text(row, workspace, files)
        shown += [(row["path"].removeprefix("repo/"), row["line"] + offset, line)
                  for offset, line in enumerate(text.splitlines()) if row["line"] + offset <= row.get("end_line", row["line"])]
    added, invented = [], []
    for kind, value in literal_values(answer, task):
        if contains(kind, value, near_text):
            continue
        where = next(((path, number) for path, number, line in shown if contains(kind, value, line)), None)
        if where is None:
            invented.append(value)
        elif all(where != (path, number) for path, number, _ in added):
            added.append((where[0], where[1], value))
    return [f"- {path}:{number} `{value}`" for path, number, value in added], invented


def with_points(reply, workspace, evidence=()):
    """Fold points into a finish answer as "- path:start-end `symbol`: text" lines.

    The line range comes from looking the symbol up in the file, so a citation is never a number the model
    made up. Returns the reply and the (path, symbol) pairs that name nothing in their file.
    """
    # points is always removed, so what is left has exactly the fields of the plain SCHEMA that reply_problem expects.
    points = reply.pop("points", None) if isinstance(reply, dict) else None
    if not isinstance(points, list) or reply.get("action") != "finish" or not isinstance(reply.get("content"), str):
        return reply, []
    lines, missing, files = [], [], []
    for point in points:
        # A malformed or empty point is skipped; the rest of the answer is still used.
        if not isinstance(point, dict) or not all(isinstance(point.get(key), str) for key in ("path", "symbol", "text")):
            continue
        path, symbol, text = point["path"].removeprefix("repo/"), point["symbol"].strip(), point["text"].strip()
        if not text:
            continue
        whole = file_named(workspace, path, symbol, files)
        try:
            if whole:  # A point about a whole file cites the file, narrowed to the lines the model saw.
                path = whole
                span = seen_span(path, (1, max(1, len(workspace.read(path).splitlines()))), evidence)
            else:
                span = symbol_lines(path, workspace.read(path), symbol)
                span = span and seen_span(path, span, evidence, symbol)
                # Looked up, but on lines the model never saw (or not found at all): cite a shown line that names it instead.
                shown = any(row["path"].removeprefix("repo/") == path and span and row["line"] <= span[1]
                            and span[0] <= row.get("end_line", row["line"]) for row in evidence)
                seen = None if shown else observed_line(symbol, path, evidence, workspace)
                if seen:
                    path, span = seen[0], (seen[1], seen[1])
        except (ValueError, OSError, UnicodeError):
            # Not a workspace file; it may be a retrieved reference document such as user/team.md.
            reference = next((row for row in evidence if row["path"] == point["path"]), None)
            span = reference and (reference["line"], reference.get("end_line", reference["line"]))
        if span is None:
            line = f"- {path} `{symbol}` (not found in the file): {text}"
        else:
            line = f"- {path}:{span[0]}" + (f"-{span[1]}" if span[1] != span[0] else "") + f" `{symbol}`: {text}"
        if line not in lines:
            lines.append(line)
            if span is None:
                missing.append((path, symbol))
    reply["content"] = "\n".join([reply["content"].strip(), *lines]).strip()
    return reply, missing


def action_key(reply):
    """Ignore narration when detecting repeated lookups; read content only matters as a line range."""
    content = reply.get("content", "")
    if reply.get("action") == "read":
        try:
            content = line_range(content)
        except ValueError:
            pass
    return (reply.get("action"), reply.get("path", ""), content)


# Search matches printed to the terminal; the model still receives all of them.
SHOWN_ROWS = 20


def clip(text, width=100):
    """Put text on one line and shorten it to at most width characters."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def format_search(data):
    """Readable terminal view of a search result; the model still receives the full JSON."""
    matches = data["matches"]
    limited = data.get("limit_reached") or data.get("candidate_limit_reached")
    skipped = f", {data['files_skipped']} skipped" if data.get("files_skipped") else ""
    lines = [f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} for {json.dumps(data.get('query', ''))} "
             f"in {data.get('scope', '.')} ({data.get('files_searched', 0)} files searched{skipped})"
             + ("; results limited" if limited else "")]
    width = max((len(f"{m['path']}:{m['line']}") for m in matches[:SHOWN_ROWS]), default=0)
    lines += [f"  {m['path'] + ':' + str(m['line']):<{width}}  {clip(m['text'])}"
              + (f"  (in {m['in']})" if m.get("in") and not m.get("defines") else "") for m in matches[:SHOWN_ROWS]]
    if len(matches) > SHOWN_ROWS:
        lines.append(f"  … {len(matches) - SHOWN_ROWS} more matches passed to the model")
    return "\n".join(lines)


def format_retrieval(data):
    """Readable terminal view of retrieved passages; the model still receives the full JSON."""
    rows = data["results"]
    limited = data.get("budget_limited") or data.get("results_limited")
    lines = [f"Retrieved {len(rows)} passage{'s' if len(rows) != 1 else ''} "
             f"({data.get('context_bytes', 0):,} of {data.get('context_budget_bytes', 0):,} bytes)"
             + ("; results limited" if limited else "")]
    labels = [f"{row['source']}:{row['start_line']}-{row['end_line']}" for row in rows]
    width = max(map(len, labels), default=0)
    for label, row in zip(labels, rows):
        detail = (f"score {row.get('score', 0):.2f}" if row.get("kind", "match") == "match"
                  else f"{row.get('kind', 'passage')} via {row.get('via', '?')}")
        if row.get("symbol"):
            detail += f", {row['symbol']}" + ("" if row.get("symbol_complete") else " (partial)")
        lines.append(f"  {label:<{width}}  {detail}")
    return "\n".join(lines)


def format_metrics(metrics):
    """One-line model timing and token summary from Ollama's reported metrics."""
    parts = []
    phases = [f"{name} {metrics[key]:.1f}s" for key, name in
              (("load_seconds", "load"), ("prompt_seconds", "prompt"), ("generation_seconds", "generation"))
              if key in metrics]
    if "total_seconds" in metrics:
        parts.append(f"{metrics['total_seconds']:.1f}s total" + (f" ({', '.join(phases)})" if phases else ""))
    elif phases:
        parts.append(", ".join(phases))
    if "prompt_tokens" in metrics:
        cached = metrics.get("cached_tokens")
        parts.append(f"{metrics['prompt_tokens']:,} prompt tokens"
                     + (f" ({cached:,} cached)" if isinstance(cached, int) else ""))
    if "generated_tokens" in metrics:
        rate = metrics.get("tokens_per_second")
        parts.append(f"{metrics['generated_tokens']:,} generated" + (f" at {rate:.1f} tok/s" if rate else ""))
    return "Ollama metrics: " + ("; ".join(parts) or "no timings or token counts reported")


# The most imports of missing code one answer lists; more than this says the answer leans on a whole library.
MISSING_SHOWN = 6
MISSING_HEADING = ("Code these lines use that is not in this repository, so what it does is not shown here "
                   "(found by CodeHaven from the imports):")


def missing_code(answer, workspace, knowledge):
    """Notes on the code an answer's cited lines use but the repository does not hold: a library that is not
    installed in it, or a relative import of a file that is not there.

    Only imports whose name appears on a cited line count, so a file's unrelated imports are left out. The notes
    name the import's line in words, not as a path:line citation, so they are never mistaken for evidence the
    answer relies on.
    """
    missing = getattr(getattr(knowledge, "dependencies", None), "missing", None)
    if not missing or not answer:
        return []
    cited = {}
    for path, start, end in CITATION.findall(answer):
        if path in missing:
            cited.setdefault(path, []).append((int(start), int(end or start)))
    notes, seen = [], set()
    for path, ranges in cited.items():
        try:
            lines = workspace.read(path).splitlines()
        except (ValueError, OSError, UnicodeError):
            continue
        shown = "\n".join(line for start, end in ranges for line in lines[max(0, start - 1):end])
        for row in missing[path]:
            used = re.search(rf"(?<![\w.$]){re.escape(row['name'])}(?![\w$])", shown)
            on_import = any(start <= row["line"] <= end for start, end in ranges)
            if not (used or on_import) or (path, row["module"]) in seen:
                continue
            seen.add((path, row["module"]))
            what = "a file this repository does not have" if row["local"] else "not part of this repository"
            notes.append(f"- `{row['module']}` ({what}), imported as `{row['name']}` on line {row['line']} of {path}")
            if len(notes) >= MISSING_SHOWN:
                return notes
    return notes
