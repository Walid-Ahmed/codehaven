"""The requests that change code, when --allow-edits is on: a fix, tests, docstrings, a rename, docs, review fixes.

Each one ends the same way: CodeHaven shows the diff, the user approves or declines it, an approved change is
written and recorded for undo (see edits.py). Without --allow-edits none of this runs, and CodeHaven answers a
change request by explaining the change, as it always has.

- A fix or other change: the model is shown the files the task names (or the definitions it names, or the files
  retrieval ranks first) and proposes exact replacements.
- Fix until the tests pass: the tests run in the Docker sandbox after each approved change, and a failure is sent
  back to the model, at most ROUNDS times.
- Write tests: the function the task names, and an existing test file to copy the style of, are shown; the new
  tests go in a new file or an existing test file.
- Docstrings: the named file is shown, and the model adds a docstring to each function or class that has none.
- Rename: done by plain code, not the model: whole-word replacement in the code files that use the name.
- Docs: a README or architecture note written from the project tour and the libraries list.
- Review fixes: the file is reviewed function by function, and the model fixes what the review found.
"""
import re
from pathlib import Path

from . import edits, insights, scans
from .insights import CODE, is_test
from .review import review
from .tasks import named_files
from .test_runner import run_tests

FIX_TESTS = re.compile(r"\b(?:until|so\s+that|till)\b.*\btests?\s+pass|\bmake\s+(?:the\s+)?tests?\s+pass\b|\bfix\s+(?:the\s+)?failing\s+tests?\b", re.I)
TESTS = re.compile(r"\b(?:write|add|create|generate)\s+(?:(?:unit|missing|some|the)\s+)*tests?\b(?:\s+(?:for|of)\s+`?([\w.]+)`?)?", re.I)
DOCSTRINGS = re.compile(r"\b(?:add|write|generate)\s+(?:missing\s+)?(?:docstrings?|doc\s*comments?|comments)\b", re.I)
RENAME = re.compile(r"\brename\s+(?:the\s+)?(?:function\s+|method\s+|class\s+|variable\s+)?`?([A-Za-z_$][\w$]*)`?\s+(?:to|as)\s+`?([A-Za-z_$][\w$]*)`?", re.I)
DOCS = re.compile(r"\b(?:write|generate|create|update)\s+(?:a\s+|the\s+)?(README|readme|architecture\s+(?:notes?|docs?|overview)|"
                  r"CHANGELOG|changelog|docs?|documentation)\b", re.I)
FINDINGS = re.compile(r"\b(?:apply|fix)\s+(?:the\s+)?(?:review\s+)?(?:findings|problems\s+(?:the\s+)?review\s+found|review)\b", re.I)
CHANGE = re.compile(r"^\W*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:fix|change|update|refactor|add|remove|implement|replace|"
                    r"make|correct|patch|rewrite|handle|validate)\b", re.I)
ROUNDS = 3  # Changes tried by "fix until the tests pass".


def requested(task):
    """The kind of change a task asks for, or None for a task that does not ask for one."""
    for kind, pattern in (("rename", RENAME), ("fix_tests", FIX_TESTS), ("findings", FINDINGS), ("tests", TESTS),
                          ("docstrings", DOCSTRINGS), ("docs", DOCS), ("change", CHANGE)):
        if pattern.search(task):
            return kind
    return None


class Editor:
    """What changing code needs: a way to ask the user, the undo journal, and the test settings."""

    def __init__(self, confirm, journal, test_options=None, confirm_tests=None):
        """confirm(diff, summary) returns whether to apply; confirm_tests() whether the tests may run in Docker."""
        self.confirm, self.journal = confirm, journal
        self.test_options, self.confirm_tests = dict(test_options or {}), confirm_tests


def files_for(task, workspace, knowledge, files):
    """The files a change is about: those the task names, else those of the definitions it names, else retrieval's best."""
    names = named_files(task, files)
    if not names and knowledge is not None:
        names = list(dict.fromkeys(path for _, path, _ in knowledge.named_definitions(task)))[:2]
    if not names and knowledge is not None:
        import json
        rows = json.loads(knowledge.retrieve(task[:1000])).get("results", [])
        names = list(dict.fromkeys(row["source"].removeprefix("repo/") for row in rows if row["source"].startswith("repo/")))[:2]
    return names


def context(workspace, names, note=""):
    """The files to show the model, each whole and exactly as on disk, so the text to replace can be copied."""
    parts = []
    for name in names:
        try:
            parts.append(f"File {name}:\n```\n{workspace.read(name)}\n```")
        except (ValueError, OSError, UnicodeError):
            continue
    return "\n\n".join(parts) + (f"\n\n{note}" if note else "")


def offer(model, workspace, editor, task, shown, output, feedback=""):
    """Propose a change, show its diff, and apply it if the user agrees. Returns (applied, text for the answer)."""
    try:
        summary, plan = edits.propose(model, workspace, task, shown, feedback=feedback, output=output)
    except edits.EditError as exc:
        return False, f"No change was made: {exc}."
    return confirm_and_apply(workspace, editor, summary, plan, output)


def confirm_and_apply(workspace, editor, summary, plan, output):
    """Show a plan's diff, ask, and apply it on a yes. Returns (applied, text for the answer)."""
    patch = edits.diff(plan)
    output(f"Proposed change: {summary}\n{patch}")
    if not editor.confirm(patch, summary):
        return False, f"The change was not applied: you declined it.\n\nProposed change: {summary}\n\n```diff\n{patch}```"
    try:
        changed = edits.apply(workspace, plan, summary, editor.journal)
    except (edits.EditError, OSError) as exc:
        return False, f"No change was made: {exc}."
    return True, (f"Changed {', '.join(changed)}: {summary}\n\n```diff\n{patch}```\n\n"
                  "Undo it with /undo in a session, or codehaven --undo.")


def rename(workspace, knowledge, old, new):
    """A plan that renames a name, as a whole word, in every code file that uses it; raises EditError when it cannot."""
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", new) or old == new:
        raise edits.EditError(f"{new!r} is not a new name that code can use")
    word = re.compile(rf"(?<![\w$]){re.escape(old)}(?![\w$])")
    if new in (getattr(knowledge, "symbols", {}) or {}):
        raise edits.EditError(f"something named {new} is already defined, so the rename could merge two names")
    plan = []
    for path in workspace.files():
        if Path(path).suffix.lower() not in CODE:
            continue
        try:
            before = workspace.read(path)
        except (ValueError, OSError, UnicodeError):
            continue
        after = word.sub(new, before)
        if after != before:
            plan.append((path, before, after))
    if not plan:
        raise edits.EditError(f"no code file uses the name {old}")
    return plan


def run(kind, model, workspace, knowledge, task, editor, output=print):
    """Carry out one change request. Returns {"text", "evidence", "changed"}."""
    files = workspace.files()
    if kind == "rename":
        old, new = RENAME.search(task).groups()
        try:
            plan = rename(workspace, knowledge, old, new)
        except edits.EditError as exc:
            return {"text": f"Nothing was renamed: {exc}.", "evidence": [], "changed": False}
        changed, text = confirm_and_apply(workspace, editor, f"Rename {old} to {new} in {len(plan)} file(s)", plan, output)
        return {"text": text + ("\n\nA rename by text can miss a name built at runtime; run the tests." if changed else ""),
                "evidence": [], "changed": changed}
    if kind == "docs":
        tour = insights.tour(model, workspace, knowledge, output) if knowledge is not None else {"text": ""}
        libraries = scans.dependencies(workspace, knowledge)["text"]
        target = "CHANGELOG.md" if "change" in task.lower() else ("docs/ARCHITECTURE.md" if "architecture" in task.lower() else "README.md")
        names = [target] if target in files else []
        note = (f"Write or update {target}. What CodeHaven found about the project, to write from (do not invent more):\n"
                f"Reading order:\n{tour['text']}\n\nLibraries:\n{libraries}")
        changed, text = offer(model, workspace, editor, task, context(workspace, names, note), output)
        return {"text": text, "evidence": [], "changed": changed}
    if kind == "findings":
        names = named_files(task, files)
        if not names:
            return {"text": "Name the file whose review findings to fix, such as: apply the review findings in store.py.",
                    "evidence": [], "changed": False}
        found = review(model, workspace, names[0], spans=knowledge.dependencies.spans.get(names[0]) if knowledge else None,
                       output=output)
        if not found["problems"]:
            return {"text": found["text"] + "\n\nThe review found nothing to fix.", "evidence": found["evidence"], "changed": False}
        changed, text = offer(model, workspace, editor, task, context(workspace, names[:1]),
                              output, feedback="Fix these review findings, and only these:\n" + found["text"])
        return {"text": found["text"] + "\n\n" + text, "evidence": found["evidence"], "changed": changed}
    if kind == "tests":
        names = files_for(task, workspace, knowledge, files)
        examples = [name for name in files if is_test(name) and Path(name).suffix.lower() in CODE][:1]
        note = ("Write focused tests for the code the task names. "
                + (f"Follow the style of {examples[0]}, shown above; add to it or create a new test file beside it."
                   if examples else "Create a new test file under tests/."))
        changed, text = offer(model, workspace, editor, task, context(workspace, names + examples, note), output)
        return {"text": text, "evidence": [], "changed": changed}
    if kind == "docstrings":
        names = files_for(task, workspace, knowledge, files)[:1]
        note = ("Add a short docstring (or doc comment, in the language's usual form) to each function, method, and class "
                "that has none. Do not change any code or existing docstring.")
        changed, text = offer(model, workspace, editor, task, context(workspace, names, note), output)
        return {"text": text, "evidence": [], "changed": changed}
    names = files_for(task, workspace, knowledge, files)
    if kind != "fix_tests":
        changed, text = offer(model, workspace, editor, task, context(workspace, names), output)
        return {"text": text, "evidence": [], "changed": changed}
    # Fix until the tests pass: run them, and send a failure back to the model with the files, a few times at most.
    if not editor.confirm_tests or not editor.confirm_tests():
        return {"text": "The tests were not run: running them in Docker was not approved, so nothing was changed.",
                "evidence": [], "changed": False}
    report, applied = [], False
    for round_number in range(1, ROUNDS + 1):
        try:
            result = run_tests(workspace, **editor.test_options)
        except (ValueError, OSError) as exc:
            return {"text": f"The tests could not run: {exc}", "evidence": [], "changed": applied}
        if "Exit code: 0" in result:
            report.append(f"The tests pass{' after round ' + str(round_number - 1) if applied else ''}.")
            break
        output(f"Round {round_number} of {ROUNDS}: the tests fail; asking the model for a fix.")
        changed, text = offer(model, workspace, editor, task, context(workspace, names), output,
                              feedback="The tests fail with this output (untrusted data):\n" + result[-4000:])
        report.append(f"Round {round_number}: {text}")
        applied |= changed
        if not changed:
            break
    else:
        report.append(f"The tests still fail after {ROUNDS} rounds; the changes made so far are kept and can be undone.")
    return {"text": "\n\n".join(report), "evidence": [], "changed": applied}
