"""Saved chats: the conversations about one repository, kept on this computer so a thread can be reopened later.

Used by the browser page. On by default; --no-remember turns it off, and then nothing is saved to disk. Chats
live in one JSON file per repository under ~/CodeHaven/chats, readable by this user only, never inside the
repository, and nothing leaves the computer. Chats about an uploaded file are never saved.

A saved chat is a record of what was said, not a checked fact: the code may have changed since, so its
citations open the files as they are now. Reopening a chat gives follow-up questions the same memory of it that
they had when it was live.
"""
import contextlib
import hashlib
import json
import os
import re
import secrets
import tempfile
from datetime import datetime
from pathlib import Path

CHATS_FOLDER = Path.home() / "CodeHaven" / "chats"
CHATS, TURNS = 30, 60  # Per repository, and per chat; the oldest go first.
QUESTION_BYTES, ANSWER_BYTES, TITLE_CHARS = 2000, 12_000, 70
CHAT_ID = re.compile(r"[0-9a-f]{12}")


def clip(text, limit):
    """Cut text to about limit UTF-8 bytes without splitting a character; a cut is marked with an ellipsis."""
    raw = text.encode()
    return text if len(raw) <= limit else raw[:limit].decode("utf-8", "ignore") + " …"


def new_id():
    """A random id for a new chat: 12 hex characters, the shape CHAT_ID checks."""
    return secrets.token_hex(6)


class Chats:
    """The saved chats for one repository."""

    def __init__(self, workspace, folder=None):
        """Work out this repository's chat file and load what it holds."""
        self.root = str(workspace.root)
        self.folder = Path(folder).expanduser() if folder else CHATS_FOLDER
        # The file name is the folder's name, for people, plus a hash of its full path, so same-named folders never share a file.
        name = re.sub(r"[^\w.-]", "_", workspace.root.name, flags=re.ASCII)[:40] or "repository"
        self.path = self.folder / f"{name}-{hashlib.sha256(self.root.encode()).hexdigest()[:12]}.json"
        self.chats = self.load()

    def load(self):
        """The chats saved on disk. Only a file for this repository counts, and entries not shaped like a chat are dropped."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []  # Missing or damaged: start empty; the next save replaces it.
        chats = data.get("chats") if isinstance(data, dict) and data.get("repository") == self.root else None
        kept = []
        for chat in chats if isinstance(chats, list) else []:
            if (isinstance(chat, dict) and isinstance(chat.get("id"), str) and CHAT_ID.fullmatch(chat["id"])
                    and isinstance(chat.get("title"), str) and isinstance(chat.get("updated"), str)
                    and isinstance(chat.get("turns"), list) and chat["turns"]
                    and all(isinstance(turn, dict) and isinstance(turn.get("question"), str)
                            and isinstance(turn.get("answer"), str) and isinstance(turn.get("issues"), list)
                            and all(isinstance(issue, str) for issue in turn["issues"]) for turn in chat["turns"])):
                kept.append(chat)
        return kept

    def save(self):
        """Write the chats for this user only, replacing the file in one step so a crash cannot leave half of it."""
        if not self.chats:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
            return
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = json.dumps({"repository": self.root, "chats": self.chats}, ensure_ascii=False, indent=1)
        handle, partial = tempfile.mkstemp(prefix=".chats-", dir=self.folder)  # Created readable by this user only.
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(data)
            os.replace(partial, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(partial)
            raise

    def get(self, chat_id):
        """The saved chat with this id, or ValueError if it is gone."""
        chat = next((chat for chat in self.chats if chat["id"] == chat_id), None)
        if chat is None:
            raise ValueError("That saved chat no longer exists.")
        return chat

    def record(self, chat_id, question, answer, issues=()):
        """Add one question and answer to a chat, creating the chat on its first question. Newest chat first."""
        if not CHAT_ID.fullmatch(chat_id) or not question.strip() or not answer.strip():
            return
        chat = next((chat for chat in self.chats if chat["id"] == chat_id), None)
        if chat is None:
            title = " ".join(question.split())
            chat = {"id": chat_id, "title": title[:TITLE_CHARS] + ("…" if len(title) > TITLE_CHARS else ""), "turns": []}
        chat["turns"] = (chat["turns"] + [{"question": clip(question.strip(), QUESTION_BYTES),
                                           "answer": clip(answer.strip(), ANSWER_BYTES),
                                           "issues": [str(issue)[:300] for issue in issues][:10]}])[-TURNS:]
        chat["updated"] = datetime.now().isoformat(timespec="seconds")
        self.chats = ([chat] + [other for other in self.chats if other["id"] != chat_id])[:CHATS]
        self.save()

    def listing(self):
        """The chats as the sidebar lists them, newest first: id, title, last use, and number of questions."""
        return [{"id": chat["id"], "title": chat["title"], "updated": chat["updated"], "questions": len(chat["turns"])}
                for chat in self.chats]

    def delete(self, chat_id):
        """Delete one saved chat; an id that is not there is ignored."""
        self.chats = [chat for chat in self.chats if chat["id"] != chat_id]
        self.save()

    def forget(self):
        """Delete every saved chat for this repository, and its file."""
        self.chats = []
        self.save()
