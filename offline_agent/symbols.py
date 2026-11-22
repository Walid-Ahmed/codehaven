"""Find the functions and classes a question names, so their code can be shown to the model before it answers.

Measured on real repositories, the local model answers at its first step from whatever it was shown. When the
question was "which environment variables does get_app_dir read?" and the body of get_app_dir was not in front
of it, it made an answer up. So plain code looks the named parts up in the parsed symbols and puts their lines,
numbered, into the prompt: the same idea as preloading a file the question names, one level finer.

A word counts as naming code only when it looks like code or is marked as code: it is in backticks, has a dot
(Response.ok, app.listen), an underscore, a digit, or camelCase, or it is a capitalised word inside the sentence
that is exactly the name of a class. Plain English words are never looked up, so "get" or "request" in a
sentence does not pull in a function of that name.
"""
import re

from . import trace
from .evidence import classify
from .tasks import without_paths
from .workspace import sections

SYMBOLS, SYMBOL_BYTES = 2, 3500  # Definitions shown for one question, and the bytes they share.
INHERITORS = 6  # Classes listed as inheriting from a named class.
CONSTANTS, CONSTANT_LINES = 4, 8  # Constants whose values are shown under those definitions, and the most lines of one.
CODE_LIKE = re.compile(r"[\d_]|[a-z][A-Z]")
# Words too common in questions to say where in a long class the answer is.
COMMON = {"which", "where", "there", "their", "these", "those", "about", "after", "before", "class", "function", "method", "default",
          "value", "values", "called", "calls", "using", "should", "would", "could", "happens", "given", "explain", "defined", "inherit"}


def words(task):
    """The words of a task that may name code, in order: (word, sure). sure means it was marked or shaped as code."""
    found = [(word.strip().removesuffix("()"), True) for word in re.findall(r"`([^`\n]{1,80})`", task)]
    plain = without_paths(re.sub(r"`[^`\n]*`", " ", task))
    # Dotted names are read from the task itself: without_paths drops a word like Response.ok as if it were a file.
    found += [(word, True) for word in re.findall(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+", task)]
    for position, word in enumerate(re.findall(r"[A-Za-z_$][\w$]*", plain)):
        if CODE_LIKE.search(word):
            found.append((word, True))
        elif position and word[0].isupper() and not word.isupper():
            found.append((word, False))  # Perhaps a class, such as Session; decided against the parsed names.
    seen, unique = set(), []
    for word, sure in found:
        if word and word not in seen:
            seen.add(word)
            unique.append((word, sure))
    return unique


def table(knowledge):
    """Every parsed symbol as (path, span, is_holder): functions and methods, then the classes that hold them."""
    spans = getattr(getattr(knowledge, "dependencies", None), "spans", None) or {}
    functions = trace.functions(knowledge)
    known = {(path, span["start"], span["end"]) for path, span in functions}
    holders = [(path, span) for path in sorted(spans) for span in spans[path] if (path, span["start"], span["end"]) not in known]
    return [(path, span, False) for path, span in functions] + [(path, span, True) for path, span in holders]


def find(task, knowledge, paths=None, limit=SYMBOLS):
    """The symbols a task names: [(path, name, start, end)], at most limit, the first named first.

    paths limits the search to those files, for a task that also names the file to look in.
    """
    rows = [row for row in table(knowledge) if paths is None or row[0] in paths]
    found = []
    for word, sure in words(task):
        short = word.rsplit(".", 1)[-1]
        exact = [row for row in rows if row[1]["name"] == word]
        if exact:
            choices = exact
        elif not sure:
            # A capitalised word is taken as code only when a class has exactly that name.
            choices = [row for row in rows if row[2] and row[1]["name"].rsplit(".", 1)[-1] == word]
        else:
            choices = [row for row in rows if row[1]["name"].rsplit(".", 1)[-1] == short]
            if "." in word:
                # requests.get names a module's function, not Session.get: prefer what belongs to no class,
                # then a class whose own name is the word's first part.
                owner = word.rsplit(".", 2)[-2]
                owned = [row for row in choices if row[1]["name"].rpartition(".")[0].rsplit(".", 1)[-1] == owner]
                choices = owned or [row for row in choices if "." not in row[1]["name"]] or choices
        # Code before its tests, a function before a class of the same name, then by path, so the choice is stable.
        choices.sort(key=lambda row: (classify(row[0]) == "test", row[2], "." in row[1]["name"] and "." not in word, row[0], row[1]["start"]))
        if choices:
            path, span, _ = choices[0]
            entry = (path, span["name"], span["start"], span["end"])
            # Not twice, and not a class together with one of its own methods: the more specific one was named first.
            overlaps = any(path == other[0] and not (span["end"] < other[2] or other[3] < span["start"]) for other in found)
            if not overlaps:
                found.append(entry)
        if len(found) >= limit:
            break
    return found


def keywords(task):
    """The longer plain words of a task, lower case: what to look for inside a class too long to show whole."""
    return [word for word in dict.fromkeys(re.findall(r"[a-z]{5,}", without_paths(task).lower())) if word not in COMMON]


def constants(knowledge, workspace, text, skip=()):
    """Where the constants used in some shown code are set: [(path, first line number, its lines)], at most CONSTANTS.

    A definition often answers a question only together with a value set elsewhere: pool_connections=DEFAULT_POOLSIZE
    says nothing until DEFAULT_POOLSIZE = 10 is seen, and without it the model invents the number.
    """
    known = getattr(knowledge, "symbols", None) or {}
    found = []
    for name in dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]{3,}\b", text)):
        for path, number in known.get(name, []):
            try:
                lines = workspace.read(path).splitlines()
            except (ValueError, OSError, UnicodeError):
                continue
            line = lines[number - 1] if 0 < number <= len(lines) else ""
            # Only the line that sets it, such as NAME = 10 or const NAME = 10, and not one already on show.
            if re.match(rf"\s*(?:export\s+)?(?:(?:const|let|var|static|final|pub|public|private)\s+)*(?:[\w.<>\[\]]+\s+)?{name}\b[^=(]*=(?!=)", line) \
                    and (path, number) not in skip and classify(path) != "test":
                # A value written over several lines, such as a tuple of codes, is shown to its closing bracket.
                last = number
                while last < min(len(lines), number + CONSTANT_LINES - 1) and \
                        sum(text.count(opener) - text.count(closer) for text in lines[number - 1:last] for opener, closer in ("()", "[]", "{}")) > 0:
                    last += 1
                found.append((path, number, [text.rstrip()[:200] for text in lines[number - 1:last]]))
                break
        if len(found) >= CONSTANTS:
            break
    return found


def inheritors(knowledge, name):
    """The classes that say they inherit from a class: [(path, line number, line text)], at most INHERITORS.

    "Which classes inherit from BadParameter?" cannot be answered from BadParameter's own lines, and the model
    does not search for it; the class lines of the whole project can be checked in a moment.
    """
    short = name.rsplit(".", 1)[-1]
    if not short[:1].isupper():
        return []
    # class Child(Base, Other):  or  class Child extends Base implements Other
    line_of = re.compile(rf"^[ \t]*(?:export[ \t]+|public[ \t]+|abstract[ \t]+|final[ \t]+)*class[ \t]+\w+[ \t]*"
                         rf"(?:\([^)]*\b{re.escape(short)}\b[^)]*\)|[^{{(]*\b(?:extends|implements)\b[^{{]*\b{re.escape(short)}\b)", re.M)
    found = []
    for source, text in sorted((getattr(knowledge, "source_texts", None) or {}).items()):
        path = source.removeprefix("repo/")
        if not source.startswith("repo/") or classify(path) == "test":
            continue
        for match in line_of.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            found.append((path, number, text.splitlines()[number - 1].strip()[:200]))
            if len(found) >= INHERITORS:
                return found
    return found


def shown(workspace, found, task="", budget=SYMBOL_BYTES, knowledge=None):
    """The numbered lines of each found symbol, cut to share budget bytes. Returns (text, evidence rows, log lines).

    A definition that fits is shown whole. A longer one, usually a class, is shown as its start, its end, and the
    lines around the task's own words, the way a long file is.
    """
    parts, evidence, log = [], [], []
    share = budget // max(1, len(found))
    for path, name, start, end in found:
        try:
            lines = workspace.read(path).splitlines()
        except (ValueError, OSError, UnicodeError):
            continue
        end = min(end, len(lines))
        body = lines[start - 1:end]
        if not body:
            continue
        wanted = keywords(task)
        # Up to four lines per word, by line number inside the definition.
        focus = [number for word in wanted
                 for number in [index for index, line in enumerate(body, 1) if word in line.lower()][:4]]
        ranges = sections("\n".join(body) + "\n", share, focus, 6)
        pieces = []
        for first, last in ranges:
            first, last = start + first - 1, start + last - 1
            pieces.append(f"Lines {first}-{last}:\n" + "\n".join(f"{number:>4}| {lines[number - 1]}" for number in range(first, last + 1)))
            evidence.append({"path": path, "line": first, "end_line": last, "text": "\n".join(lines[first - 1:last])})
        whole = len(ranges) == 1 and ranges[0] == (1, len(body))
        parts.append(f"{path}: the definition of {name}, lines {start}-{end}"
                     + ("" if whole else f" (too long to show whole; read other lines of it with a range such as {start}-{min(end, start + 60)})")
                     + "\n" + "\n".join(pieces))
        log.append(f"Preloaded the definition of {name} ({path}:{start}-{end}{'' if whole else ', in parts'}) because the task names it.")
    text = "\n\n".join(parts)
    if text and knowledge is not None:
        on_show = {(row["path"], number) for row in evidence for number in range(row["line"], row["end_line"] + 1)}
        values = constants(knowledge, workspace, text, on_show)
        if values:
            text += "\n\nConstants these lines use, and where each is set:\n" + "\n".join(
                f"{path}:{number + offset}| {line}" for path, number, lines in values for offset, line in enumerate(lines))
            evidence += [{"path": path, "line": number, "end_line": number + len(lines) - 1, "text": "\n".join(lines)}
                         for path, number, lines in values]
        for _, name, _, _ in found:
            children = inheritors(knowledge, name)
            if children:
                text += f"\n\nClasses that inherit from {name.rsplit('.', 1)[-1]}:\n" + "\n".join(
                    f"{path}:{number}| {line}" for path, number, line in children)
                evidence += [{"path": path, "line": number, "end_line": number, "text": line} for path, number, line in children]
    return text, evidence, log
