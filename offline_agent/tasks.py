"""Reading a task before the model sees it: what kind it is, and which files and code names it mentions."""
import json
import re

from . import arabic
from .evidence import NEGATED_EDIT
from .prompts import EDIT_REQUEST, POLITE, PRELOAD_FILES, QUESTION


def without_paths(task):
    """Drop path-like words so a folder named services/ does not change how the task is read."""
    # Two shapes of path: a word containing a slash or backslash, or a word ending in a file extension.
    return re.sub(r"\S*[/\\]\S*|\S+\.\w{1,8}\b", " ", task)


def entry_point_task(task):
    """True when the task is about entry points or services, which adds ENTRY_POINT_GUIDANCE to the final-answer call."""
    return bool(re.search(r"entry[ -]?points?|\bservices?\b", without_paths(task), re.I))


def task_kind(task):
    """question (answers cite code), change (asks for a change, which is explained, never made), or other (no assumptions)."""
    # Fold Arabic spelling first, then drop a polite opener, paths, and negated verbs ("do not change ...")
    # before looking for question or edit words, in English or Arabic.
    task = arabic.normalise(task)
    polite = POLITE.match(task) or arabic.POLITE.match(task)
    text = NEGATED_EDIT.sub(" ", without_paths(task[polite.end():] if polite else task).lower()).strip()
    text = arabic.NEGATED_EDIT.sub(" ", text).strip()
    edit = (EDIT_REQUEST.search(text) or arabic.EDIT.search(text)) is not None
    asked = ((QUESTION.match(text) or arabic.QUESTION.match(text)) is not None
             or (text.endswith(("?", "؟")) and not polite))
    # A question that also has an edit verb ("how do I add ...?") could be either, so nothing is assumed about it.
    if asked:
        return "other" if edit else "question"
    return "change" if edit else "other"


def code_names(task):
    """Names in a task that look like code rather than English, such as SKU-0042, low_stock, or applyCoupon."""
    names = []
    # Dotted words are removed with the paths, so they are added back: config.max_retries still yields max_retries.
    for word in re.findall(r"[A-Za-z_][\w.-]*\w", without_paths(task) + " " + " ".join(re.findall(r"\S+\.\w+", task))):
        name = word.rsplit(".", 1)[-1]
        # Code-like means the name has a digit, an underscore, or camelCase; plain English words are left out.
        if re.search(r"[\d_]|[a-z][A-Z]", name) and name not in names:
            names.append(name)
    return names


def focus_lines(task, text):
    """Line numbers of the code-like names a task mentions."""
    lines, found = text.splitlines(), []
    # At most 4 names and 2 lines for each, so a long file is still shown in only a few parts.
    for name in code_names(task)[:4]:
        found += [number for number, line in enumerate(lines, 1) if name in line][:2]
    return found


# search_named looks up at most this many names from the task, and shows this many matches for each.
SEARCHED_NAMES, SEARCHED_MATCHES = 2, 8


def search_named(workspace, task):
    """Search for the code names a task mentions, so "who calls X?" starts from where X really appears.

    Left to itself the model often answers from the definition alone, without searching. Returns the lines to
    show and the matches to keep as evidence.
    """
    shown, evidence = [], []
    for name in code_names(task)[:SEARCHED_NAMES]:
        try:
            matches = json.loads(workspace.search(name))["matches"]
        except (ValueError, OSError, UnicodeError, KeyError):
            continue
        for match in matches[:SEARCHED_MATCHES]:
            where = ("defines it" if match.get("defines") else f"inside {match['in']}" if match.get("in") else "top level")
            shown.append(f"- {match['path']}:{match['line']} ({where}): {match['text'].strip()[:160]}")
            evidence.append(match)
        if len(matches) > SEARCHED_MATCHES:
            shown.append(f"- ... {len(matches) - SEARCHED_MATCHES} more matches for {name}; use search to see them")
    return shown, evidence


def section_note(start, end, next_start, total):
    """Header for one shown part of a file, naming the lines skipped before the next part."""
    if next_start is None:
        return page_note(start, end, total)
    hidden = next_start - end - 1
    # The suggested range is as long as the part just shown, but stops before the next shown part.
    return (f"Lines {start}-{end} of {total}; {hidden} more lines, request content "
            f"\"{end + 1}-{min(next_start - 1, end + end - start + 1)}\"")


def step_label(step, max_steps):
    """Progress label such as "Step 2 (max 12)": the limit is a ceiling, not the planned number of steps."""
    return f"Step {step} (max {max_steps})"


def page_note(start, end, total):
    """Header for a page of a file, naming the next range when lines remain."""
    remaining = (f"; {total - end} more lines, request content \"{end + 1}-{min(total, end + end - start + 1)}\""
                 if end < total else "")
    return f"Lines {start}-{end} of {total}{remaining}"


def named_files(task, files):
    """Files the task names, in order of mention: an exact listed path, or a unique file name."""
    by_name = {}
    for name in files:
        by_name.setdefault(name.rsplit("/", 1)[-1], []).append(name)
    listed, found = set(files), []
    for word in task.split():
        # Remove quotes, brackets, and sentence punctuation around the word.
        word = word.strip("`'\"()[]{}<>,;:!?").rstrip(".").removeprefix("./")
        # Only a word with a slash or a file extension can name a file.
        if "/" not in word and not re.search(r"\.\w{1,8}$", word):
            continue
        # A bare file name counts only when exactly one listed file has that name.
        match = word if word in listed else (by_name[word][0] if len(by_name.get(word, [])) == 1 else None)
        if match and match not in found:
            found.append(match)
        if len(found) == PRELOAD_FILES:
            break
    return found
