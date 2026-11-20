"""Long-term memory: checked answers about one repository, kept on this computer between sessions.

On by default in the codehaven command; --no-remember turns it off, and then nothing is saved to disk. Notes
live in one JSON file per repository under ~/CodeHaven/memory, never inside the repository, so the project
stays untouched, and nothing leaves the computer.

A note is only as good as the code it describes, so each one records a fingerprint of every file it cites.
A note is recalled only while all of those files are unchanged; once one changes, the note is deleted rather
than trusted. Only answers that passed the answer checks and cite lines are saved at all.
"""
import contextlib
import hashlib
import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path

from .evidence import CITATION
from .knowledge import tokens

MEMORY_FOLDER = Path.home() / "CodeHaven" / "memory"
NOTES = 200  # Per repository; the oldest go first.
QUESTION_BYTES, ANSWER_BYTES = 300, 1200
RECALLED, RECALL_BYTES = 2, 2400  # What one question may bring back, so notes never crowd out the code.
# Words that say nothing about which code a question concerns (already stemmed, as tokens() returns them).
COMMON = frozenset("""a an and are as at be by can do doe each for from how i if in into is it its of on or that the
this to us use what when where which who why will with work would you your explain describe show tell list find
code file function method class project repository thi wa ha""".split())


def clip(text, limit):
    """Cut text to at most limit UTF-8 bytes without splitting a character, marking the cut with an ellipsis."""
    raw = text.encode()
    return text if len(raw) <= limit else raw[:limit].decode("utf-8", "ignore") + " …"


def terms(text):
    """The meaningful stemmed words of a text, used to match a new question against saved notes."""
    return {word for word in tokens(text) if word not in COMMON and len(word) > 1}


def citations(answer):
    """(path, start, end) for each line citation in an answer, in order, without repeats."""
    found = []
    for path, start, end in CITATION.findall(answer):
        row = (path.removeprefix("repo/"), int(start), int(end or start))
        if row not in found:
            found.append(row)
    return found


class Memory:
    """The saved notes for one repository."""

    def __init__(self, workspace, folder=None):
        """Work out this repository's notes file and load what it holds."""
        self.workspace = workspace
        self.folder = Path(folder).expanduser() if folder else MEMORY_FOLDER
        root = str(workspace.root)
        # The file is named after the folder plus a hash of its full path, so two repositories with the same
        # folder name never share notes.
        name = re.sub(r"[^\w.-]", "_", workspace.root.name, flags=re.ASCII)[:40] or "repository"
        self.path = self.folder / f"{name}-{hashlib.sha256(root.encode()).hexdigest()[:12]}.json"
        self.notes = self.load(root)

    def load(self, root):
        """Read the notes file, keeping only well-formed notes that were saved for this exact repository path."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []  # Missing or damaged: start empty; the next save replaces it.
        notes = data.get("notes") if isinstance(data, dict) and data.get("repository") == root else None
        if not isinstance(notes, list):
            return []
        return [note for note in notes
                if isinstance(note, dict) and isinstance(note.get("question"), str) and isinstance(note.get("answer"), str)
                and isinstance(note.get("files"), dict) and note["files"]
                and all(isinstance(key, str) and isinstance(value, str) for key, value in note["files"].items())]

    def save(self):
        """Write the notes for this user only, replacing the file in one step so a crash cannot leave half of it."""
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = json.dumps({"repository": str(self.workspace.root), "notes": self.notes}, ensure_ascii=False, indent=1)
        handle, partial = tempfile.mkstemp(prefix=".memory-", dir=self.folder)  # Created readable by this user only.
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(data)
            os.replace(partial, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(partial)
            raise

    def fingerprint(self, path):
        """A hash of a file's current text, or None when the workspace rules no longer allow reading it."""
        try:
            return hashlib.sha256(self.workspace.read(path).encode()).hexdigest()
        except (ValueError, OSError, UnicodeError):
            return None

    def add(self, question, answer):
        """Save a checked answer. Returns False when it cites no readable lines, so it could never be re-checked."""
        files = {}
        for path in (row[0] for row in citations(answer)):
            if path not in files:
                mark = self.fingerprint(path)
                if mark is not None:
                    files[path] = mark
        question = clip(" ".join(question.split()), QUESTION_BYTES)
        if not files or not question:
            return False
        asked = question.lower()
        self.notes = [note for note in self.notes if note["question"].lower() != asked]  # A newer answer replaces an older one.
        self.notes.append({"question": question, "answer": clip(answer.strip(), ANSWER_BYTES), "files": files,
                           "saved": date.today().isoformat()})
        self.notes = self.notes[-NOTES:]
        self.save()
        return True

    def fresh(self, note, marks):
        """Whether every file a note cites still has the fingerprint saved with the note.

        marks caches the current fingerprints, so one recall hashes each file once however many notes cite it.
        """
        for path, mark in note["files"].items():
            if path not in marks:
                marks[path] = self.fingerprint(path)
            if marks[path] != mark:
                return False
        return True

    def recall(self, question, skip=()):
        """Notes relevant to a question whose cited files are all unchanged, best first.

        Relevant means sharing at least two meaningful words with the question, or most of its meaningful
        words when it has few. Notes found stale along the way are deleted. skip holds questions already in
        the conversation, which need no reminder.
        """
        wanted = terms(question)
        skipped = {text.lower() for text in skip}
        scored = []
        for index, note in enumerate(self.notes):
            if note["question"].lower() in skipped:
                continue
            asked = terms(note["question"])
            shared = wanted & (asked | terms(note["answer"]))
            if len(shared) >= 2 or (shared and len(shared) * 2 >= len(wanted)):
                scored.append((len(shared) + len(wanted & asked), index, note))  # Words in the old question count twice.
        marks, kept, stale, size = {}, [], [], 0
        # Best score first; between equal scores, the newer note.
        for note in (row[2] for row in sorted(scored, key=lambda row: (-row[0], -row[1]))):
            if not self.fresh(note, marks):
                stale.append(note)
                continue
            cost = len(note["question"].encode()) + len(note["answer"].encode())
            if len(kept) < RECALLED and size + cost <= RECALL_BYTES:
                kept.append(note)
                size += cost
        if stale:
            self.notes = [note for note in self.notes if not any(note is old for old in stale)]
            self.save()
        return kept

    def evidence(self, notes):
        """Evidence rows for the lines recalled notes cite, so an answer may cite them again and pass its checks."""
        rows = []
        for note in notes:
            for path, start, end in citations(note["answer"]):
                # Only files that were fingerprinted with the note, and so just checked as unchanged, may be cited again.
                if path not in note["files"]:
                    continue
                try:
                    lines = self.workspace.read(path).splitlines()
                except (ValueError, OSError, UnicodeError):
                    continue
                if 1 <= start <= end <= len(lines):
                    rows.append({"path": path, "line": start, "end_line": end, "text": lines[start - 1]})
        return rows

    @staticmethod
    def text(notes):
        """Recalled notes as the Q:/A: text that is placed in the prompt."""
        return "\n".join(f"Q: {note['question']}\nA: {note['answer']}" for note in notes)

    def forget(self):
        """Delete every saved note for this repository, and its file."""
        self.notes = []
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
