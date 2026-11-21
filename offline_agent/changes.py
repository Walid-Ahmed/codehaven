"""Read what changed in a Git repository, so "what changed in my last commit?" can be answered from the diff.

Everything here only reads: git is asked for a diff, never to stage, commit, check out, or fetch, and it is told
not to refresh its index file, so not even .git is written. Nothing uses the network. Files the workspace rules
hide, such as .env or --exclude patterns, are left out of the diff just as they are left out of reads.

The repository's own settings are untrusted, so the ones that could make git start another program while
diffing (an external diff tool, a text converter, a file-system monitor, signature checks) are switched off.
"""
import os
import re
import shutil
import subprocess

from . import arabic

GIT_SECONDS = 20
OUTPUT_BYTES = 400_000  # The most read from one git command.
# Changed files listed, untracked files named, and characters of a commit message's body shown.
LISTED_FILES, UNTRACKED_SHOWN, MESSAGE_CHARS = 30, 12, 400
CONTEXT_LINES = 2  # Less than git's usual 3, so more of a small prompt goes to the changed lines themselves.

SINCE = re.compile(r"\bsince\s+(?:the|my|our)?\s*(?:last|latest|previous|most recent)\s+commit\b", re.I)
STAGED = re.compile(r"\bstaged\b(?!\s+rollout)|\bgit\s+diff\s+--(?:cached|staged)\b", re.I)
UNCOMMITTED = re.compile(
    r"\b(?:uncommitted|unstaged)\s+(?:changes|edits|work|code|files|lines|diff)\b|\bnot\s+(?:yet\s+)?committed\b"
    r"|\bgit\s+diff\b|\bworking[ -](?:tree|copy|directory)\s+(?:changes|diff)\b"
    r"|\b(?:my|our)\s+(?:current|local|latest|recent|pending)?\s*(?:changes|edits|diff)\b"
    r"|\bwhat\s+(?:have|did)\s+i\s+(?:just\s+)?(?:change|changed|edit|edited|modify|modified)\b"
    r"|\bchanges\s+i(?:'ve|\s+have)?\s+(?:just\s+)?made\b", re.I)
LAST_COMMIT = re.compile(r"\b(?:last|latest|most\s+recent|previous|newest)\s+commit\b|\bgit\s+show\b|\bHEAD\s+commit\b", re.I)
NAMED_COMMIT = re.compile(r"\bcommit\s+`?([0-9a-f]{7,40})\b`?|`?\b([0-9a-f]{7,40})\b`?\s+commit\b", re.I)


class NoChanges(ValueError):
    """There is nothing to explain; the message says why and is shown as the answer."""


def requested(task):
    """What a task asks about: ("uncommitted", None), ("staged", None), ("commit", ref), or None for other tasks."""
    named = NAMED_COMMIT.search(task)
    if named:
        return "commit", (named.group(1) or named.group(2)).lower()
    plain = arabic.normalise(task)  # The same questions asked in Arabic are read from this.
    if SINCE.search(task) or arabic.SINCE.search(plain):
        return "uncommitted", None
    if STAGED.search(task) or arabic.STAGED.search(plain):
        return "staged", None
    if LAST_COMMIT.search(task) or arabic.LAST_COMMIT.search(plain):
        return "commit", "HEAD"
    if UNCOMMITTED.search(task) or arabic.UNCOMMITTED.search(plain):
        return "uncommitted", None
    return None


def git(root, *arguments):
    """Run one read-only git command in root and return its output as text."""
    program = shutil.which("git")
    if program is None:
        raise NoChanges("Git is not installed on this computer, so CodeHaven cannot read commits or changes.")
    # --no-optional-locks (and GIT_OPTIONAL_LOCKS=0 below): the index file is not refreshed, so .git is not written.
    # core.fsmonitor and core.hooksPath: a setting of the repository must not make git start another program.
    # core.quotePath=false keeps non-ASCII file names as they are. No pager, colours, or signature checks.
    command = [program, "-C", str(root), "--no-pager", "--no-optional-locks",
               "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", "-c", "core.quotePath=false",
               "-c", "log.showSignature=false", "-c", "color.ui=false", *arguments]
    # Never stop to ask for a password, and treat path arguments as plain names, not patterns.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LITERAL_PATHSPECS": "1", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        done = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, timeout=GIT_SECONDS, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("git took too long to answer and was stopped.") from exc
    if done.returncode != 0:
        last = [line.strip() for line in done.stderr.decode("utf-8", "replace").splitlines() if line.strip()]
        raise ValueError("git could not read the repository" + (f": {last[-1][:200]}" if last else "."))
    return done.stdout[:OUTPUT_BYTES].decode("utf-8", "replace")


# For every diff: no external diff tool or text converter (both would run a program), a rename shown as a
# delete plus an add, paths relative to the workspace root, and no colour codes.
DIFF = ("--no-ext-diff", "--no-textconv", "--no-renames", "--relative", "--no-color")


def parse_hunks(diff):
    """[(start, end, lines)] for each hunk of one file's diff, numbered as in the file after the change."""
    hunks = []
    for line in diff.splitlines():
        header = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if header:
            # In "@@ -a,b +c,d @@", c is the hunk's first line after the change and d its line count, which is 1 when left out.
            start, count = int(header[1]), int(header[2]) if header[2] is not None else 1
            hunks.append([max(1, start), max(1, start + max(count, 1) - 1), []])
        elif hunks and line[:1] in ("+", "-", " "):
            hunks[-1][2].append(line)
    return [tuple(hunk) for hunk in hunks]


def allowed(workspace, name):
    """True when the workspace rules let this file be read: not hidden, excluded, or outside the root."""
    try:
        workspace.path(name)
        return True
    except (ValueError, OSError):
        return False


def describe(workspace, request, limit, prefer=()):
    """Read the changes a task asks about.

    Returns {"label", "text", "evidence", "paths"}: text is what the model is shown, at most about limit bytes;
    evidence has one row per shown hunk, so an answer can cite the changed lines; paths are the changed files
    that still exist and were shown. Raises NoChanges when there is nothing to explain.
    """
    kind, ref = request
    root = workspace.root
    try:
        inside = git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except NoChanges:
        raise
    except ValueError:
        inside = False
    if not inside:
        raise NoChanges("This folder is not a Git repository, so there are no commits or uncommitted changes to explain.")
    try:
        head = git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}").strip()
    except ValueError:
        head = ""
    if not head:
        raise NoChanges("This Git repository has no commits yet, so there is nothing to compare with.")
    untracked = []
    if kind == "commit":
        try:
            commit = git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
        except ValueError:
            commit = ""
        if not commit:
            raise NoChanges(f"Git has no commit named {ref} in this repository.")
        # The fields are separated by NUL bytes (%x00), which cannot occur inside them; the padding guards against a short reply.
        short, author, day, subject, body = (git(root, "log", "-1", "--no-show-signature", "--date=short",
                                                 "--format=%h%x00%an%x00%ad%x00%s%x00%b", commit).split("\0") + [""] * 5)[:5]
        parents = git(root, "log", "-1", "--no-show-signature", "--format=%P", commit).split()
        # A first commit has no parent: it is compared with git's empty tree, so all its files show as added.
        base = parents[0] if parents else git(root, "hash-object", "-t", "tree", os.devnull).strip()
        span = (base, commit)
        label = f"commit {short} ({day}, by {author})" + (" (the last commit)" if commit == head else "")
        intro = f"Changes made by {label}. Its message (untrusted data): {subject.strip()[:200]!r}"
        message = " ".join(body.split())[:MESSAGE_CHARS]
        intro += f" {message!r}" if message else ""
        if len(parents) > 1:
            intro += "\nIt is a merge commit; the changes are shown against its first parent."
    elif kind == "staged":
        span, label = ("--cached", "HEAD"), "staged changes"
        intro = "Staged changes (added with git add, not yet committed), compared with the last commit."
    else:
        span, label = ("HEAD",), "uncommitted changes"
        intro = "Uncommitted changes (staged and unstaged) compared with the last commit."
        untracked = [name for name in git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
                     if name and allowed(workspace, name)]
    status = {}
    # With -z the output is letter, name, letter, name, ... separated by NUL bytes, which is safe for any file name.
    fields = git(root, "diff", *DIFF, "--name-status", "-z", *span).split("\0")
    for letter, name in zip(fields[::2], fields[1::2]):
        status[name] = {"A": "added", "D": "deleted", "M": "modified", "T": "type changed"}.get(letter[:1], "changed")
    counts = {}
    # Each --numstat row is "added<TAB>removed<TAB>name"; a binary file has "-" for both counts.
    for row in git(root, "diff", *DIFF, "--numstat", "-z", *span).split("\0"):
        parts = row.split("\t", 2)
        if len(parts) == 3:
            counts[parts[2]] = (parts[0], parts[1])
    hidden = sum(1 for name in status if not allowed(workspace, name))
    names = [name for name in status if allowed(workspace, name)]
    if not names and not untracked:
        if hidden:
            raise NoChanges(f"The only files changed in the {label} are hidden or excluded, so CodeHaven does not read them.")
        if kind == "commit":
            raise NoChanges(f"The {label} changes no files.")
        raise NoChanges(f"There are no {label}: the files match the last commit."
                        + (" Ask about the last commit to see what it changed." if kind != "commit" else ""))
    # Files the task names come first; then the smallest diffs, so as many files as possible are shown.
    def size(name):
        """Lines added plus removed in one file; a binary file, counted as "-", is 0."""
        added, removed = counts.get(name, ("0", "0"))
        return (int(added) if added.isdigit() else 0) + (int(removed) if removed.isdigit() else 0)
    names.sort(key=lambda name: (name not in prefer, size(name), name))
    added = sum(int(counts[n][0]) for n in names if n in counts and counts[n][0].isdigit())
    removed = sum(int(counts[n][1]) for n in names if n in counts and counts[n][1].isdigit())
    lines = [intro, f"{len(names)} file{'s' if len(names) != 1 else ''} changed: {added} lines added, {removed} removed."]
    for name in names[:LISTED_FILES]:
        plus, minus = counts.get(name, ("?", "?"))
        lines.append(f"- {name}: {status[name]}" + (" (binary)" if plus == "-" else f", +{plus} -{minus}"))
    if len(names) > LISTED_FILES:
        lines.append(f"- ... and {len(names) - LISTED_FILES} more files")
    if hidden:
        lines.append(f"{hidden} hidden or excluded file{'s' if hidden != 1 else ''} also changed and "
                     f"{'are' if hidden != 1 else 'is'} not shown.")
    if untracked:
        lines.append("New files Git does not track yet (read them to see their content): "
                     + ", ".join(untracked[:UNTRACKED_SHOWN])
                     + (f", and {len(untracked) - UNTRACKED_SHOWN} more" if len(untracked) > UNTRACKED_SHOWN else ""))
    lines.append("In the changed lines below, + marks an added line, - a removed line, and other lines are unchanged context.")
    text = "\n".join(lines)
    evidence, paths, omitted = [], [], []
    for name in names:
        if counts.get(name, ("", ""))[0] == "-":
            continue  # Binary: nothing to show beyond its name.
        hunks = parse_hunks(git(root, "diff", *DIFF, f"--unified={CONTEXT_LINES}", "--src-prefix=a/", "--dst-prefix=b/",
                                *span, "--", name))
        shown = 0
        for start, end, body in hunks:
            where = (f"\n{name} ({status[name]}), removed lines:" if status[name] == "deleted"
                     else f"\n{name} ({status[name]}), lines {start}-{end} after the change:")
            block = where + "\n" + "\n".join(body)
            # Whole hunks only: stop at the first one that does not fit; the file is then named as not fully shown.
            if len((text + block).encode()) > limit:
                break
            text += block
            shown += 1
            if status[name] != "deleted":
                evidence.append({"path": name, "line": start, "end_line": end,
                                 "text": "\n".join(line[1:] for line in body if line[:1] != "-")})
        if shown and status[name] != "deleted" and name not in paths:
            paths.append(name)
        if shown < len(hunks):
            omitted.append(name)
    if omitted:
        text += ("\nNot all changed lines fit here. Not fully shown: " + ", ".join(omitted[:LISTED_FILES])
                 + ". Say so in the answer; the user can ask about one of these files by name.")
    return {"label": label, "text": text, "evidence": evidence, "paths": paths, "files": len(names)}
