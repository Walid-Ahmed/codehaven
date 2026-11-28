"""Explain an error: a pasted traceback's frames are matched to this project's files and shown to the model.

A traceback names files as they were on the machine that raised it, often by an absolute path or a path inside a
container. Each frame is matched to the project file whose path it ends with, and the lines around the frame are
put in the prompt with their numbers, innermost frame (where the error was raised) first. Frames in libraries and
the standard library are left out, but counted, so the answer can say the error passed through them.
Python, JavaScript and TypeScript, Java, Kotlin, C#, Go, Ruby, PHP, and Rust stack traces are read.
"""
import re
from pathlib import PurePosixPath

FRAME_PATTERNS = [
    re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?'),                       # Python
    re.compile(r"at (?:(?:async\s+)?([\w$.<>\[\] ]+?) \()?((?:[A-Za-z]:)?[^\s():]+\.\w+):(\d+)(?::\d+)?\)?"),  # JS, TS
    re.compile(r"at ([\w$.<>]+)\(([\w$-]+\.(?:java|kt|scala)):(\d+)\)"),           # Java, Kotlin
    re.compile(r"at ([\w.<>`]+)\(.*?\) in (\S+\.cs):line (\d+)"),                   # C#
    re.compile(r"^\s+((?:[A-Za-z]:)?/?[^\s:]+\.(?:go|rs)):(\d+)", re.M),           # Go, Rust
    re.compile(r"((?:[A-Za-z]:)?[^\s:'\"`]+\.(?:rb|php)):(\d+):in [`']([^']+)'"),   # Ruby
    re.compile(r"((?:[A-Za-z]:)?[^\s:'\"(]+\.php)\((\d+)\)"),                       # PHP
]
TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^\s+at [\w$.<>]+[ (]|File \"[^\"]+\", line \d+|"
                       r"\.(?:go|rs):\d+\b|panicked at|Exception in thread|\.cs:line \d+|\.rb:\d+:in ", re.M)
SOURCE = re.compile(r"\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|mts|cts|java|kt|kts|scala|cs|go|rs|rb|php)$")
AROUND, FRAMES = 3, 6  # Lines shown either side of a frame's line, and frames shown.
LIBRARY = re.compile(r"(?:^|/)(?:site-packages|dist-packages|node_modules|lib/python\d|\.cargo|go/pkg/mod|vendor|"
                     r"<frozen|internal/)")


def requested(task):
    """Whether a task holds a stack trace to explain."""
    return bool(TRACEBACK.search(task)) and bool(frames(task))


def frames(text):
    """[(path, line, function)] in the order the trace lists them, from every pattern that matches."""
    found = []
    for pattern in FRAME_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            # Each pattern puts the path, line, and function in its own order; tell them apart by what they look like.
            path = next(group for group in groups if group and SOURCE.search(group))
            line = int(next(group for group in groups if group and group.isdigit()))
            name = next((group for group in groups if group and group not in (path, str(line))), "")
            found.append((match.start(), path.replace("\\", "/"), line, name.strip()))
    found.sort()
    unique = []
    for _, path, line, name in found:
        if (path, line) not in [(p, n) for p, n, _ in unique]:
            unique.append((path, line, name))
    return unique


def locate(path, files):
    """The project file a frame's path means: the one whose path the frame's path ends with, longest match first."""
    parts = PurePosixPath(path).parts
    for start in range(len(parts)):
        tail = "/".join(parts[start:])
        matches = [name for name in files if name == tail or name.endswith("/" + tail)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 and start == len(parts) - 1:
            return None  # Only a bare file name is left and several files share it.
    return None


def shown(workspace, text, files):
    """(prompt text, evidence rows, note) for the frames of the trace in text that are in this project."""
    trace = frames(text)
    # Python lists the innermost frame last; other languages list it first.
    innermost_last = "Traceback (most recent call last)" in text
    ordered = list(reversed(trace)) if innermost_last else trace
    parts, rows, outside = [], [], 0
    for path, line, name in ordered:
        local = None if LIBRARY.search(path) else locate(path, files)
        if local is None:
            outside += 1
            continue
        if len(rows) >= FRAMES:
            continue
        try:
            lines = workspace.read(local).splitlines()
        except (ValueError, OSError, UnicodeError):
            continue
        if not 1 <= line <= len(lines):
            continue
        start, end = max(1, line - AROUND), min(len(lines), line + AROUND)
        body = "\n".join(f"{number:>4}{'>' if number == line else '|'} {lines[number - 1]}" for number in range(start, end + 1))
        label = "where the error was raised" if not rows else "called from here"
        parts.append(f"{local}:{line}" + (f" in {name}" if name else "") + f" ({label}; > marks the line):\n{body}")
        rows.append({"path": local, "line": start, "end_line": end, "text": "\n".join(lines[start - 1:end])})
    note = (f"{len(rows)} frame{'s' if len(rows) != 1 else ''} of the trace {'are' if len(rows) != 1 else 'is'} in this project"
            + (f"; {outside} {'are' if outside != 1 else 'is'} in libraries or files not here" if outside else "") + ".")
    return "\n\n".join(parts), rows, note


GUIDE = ("Explain this error for the user: say what went wrong, at which line of their code, and the most likely cause, "
         "using the frames shown and the error message. Cite the lines as path:line. If the cause is in a library frame "
         "that is not shown, say so. Suggest a fix, but do not claim it was applied.")
