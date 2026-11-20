"""CodeHaven in the browser: pick a repository, index it, and chat about its code.

The server listens on 127.0.0.1 only and serves one self-contained page: no scripts, styles, or fonts come
from the internet, and the page's Content-Security-Policy forbids loading any. Every request must carry the
secret token from the link the server prints, and a Host header naming this machine, so other websites open
in the same browser cannot use the server to read files. Like the rest of CodeHaven, it never changes files.

A single file can be uploaded instead of choosing a folder. The page sends it to this server only; a copy is
kept in a private temporary folder, which is the workspace for that chat, and deleted when another repository
or file is opened or the server stops. The original file is never touched.

The uploaded file can also be a picture of code, such as a screenshot. Tesseract, a program installed on this
computer, reads it into a text file in that same temporary folder, which the page offers to save. See ocr.py.

Besides chat, the page offers two study aids for one file or a group of files: a structure map, drawn from
static analysis without the model, and flashcards, written by the local model with looked-up line citations.

Unless started with --no-remember, checked answers about a repository are saved on this computer and recalled
in later sessions while the files they cite are unchanged. Uploaded files are never remembered.

With --allow-clone, and only then, a GitHub repository can be downloaded once by its address. This is the one
use of the network: git, run by this server, downloads the latest snapshot into the clone folder, and from then
on the copy opens offline like any other folder. Nothing is uploaded. CodeHaven never asks for, sees, or stores
a GitHub password or token: private repositories work only through the sign-in git already has on this machine.
"""
import contextlib
import json
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import documents, ocr, study
from .agent import Session
from .chats import Chats, new_id
from .knowledge import KnowledgeBase
from .memory import Memory
from .model import MODEL
from .semantic import Semantic
from .workspace import MAX_BYTES, MAX_FILES, Workspace

PAGE = Path(__file__).with_name("gui.html")
FILE_CONTEXT, FILE_LINES = 6, 400  # Lines shown around a citation, and the most one request returns.
# Limits: folders listed by the folder browser, bytes in a JSON request, characters kept of an uploaded file's name.
DIRECTORY_ENTRIES = 400
REQUEST_BYTES = 64_000
UPLOAD_NAME_CHARACTERS = 120
CLONE_FOLDER = Path.home() / "CodeHaven" / "repos"
CLONE_SECONDS, CLONE_BYTES = 180, 500_000_000  # A download is stopped after 3 minutes or 500 MB on disk.
# The characters accepted in a GitHub owner name and repository name; anything else is refused before git runs.
OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
REPOSITORY = re.compile(r"[A-Za-z0-9._-]{1,100}")
SEMANTIC_WAIT = 15  # Seconds indexing waits for meaning search; the rest is prepared in the background.
CLONE_OFF = "Cloning from GitHub is off, so CodeHaven stays offline. Restart it with --allow-clone to turn it on."


class Busy(Exception):
    """Another indexing run or question is still in progress."""


class App:
    """One repository and one conversation at a time, shared by the request threads."""

    def __init__(self, model, *, max_steps=12, retrieval_budget=4000, excludes=(), allow_clone=False, clone_folder=None,
                 remember=False, memory_folder=None, embed_model=None, index_folder=None, chats_folder=None):
        """Keep the settings. Nothing is indexed until open(), clone(), or open_file() is called."""
        self.chats_folder, self.chats, self.chat_id = chats_folder, None, new_id()
        self.embed_model, self.index_folder, self.semantic = embed_model, index_folder, None
        self.remember, self.memory_folder, self.memory = bool(remember), memory_folder, None
        self.model, self.max_steps, self.retrieval_budget = model, max_steps, retrieval_budget
        self.excludes = tuple(excludes)
        self.allow_clone = bool(allow_clone)
        self.clone_folder = Path(clone_folder).expanduser() if clone_folder else CLONE_FOLDER
        # One request at a time uses the model: busy is the flag, and the lock only guards reading and setting it.
        self.lock = threading.Lock()
        self.busy = False
        self.workspace = self.knowledge = self.session = None
        self.upload = self.upload_folder = None  # The uploaded file's name and its private temporary folder.
        self.picture = None  # When the upload was a picture: how reading it went, for the page to show.
        self.document = None  # When the upload was a notebook or a PDF: what was read from it.
        # While an answer runs, the queue its log lines and streamed text are put on; see ask() and emit().
        self.sink = None

    def claim(self):
        """Mark the app busy, or raise Busy if another request is still running. Always paired with release()."""
        with self.lock:
            if self.busy:
                raise Busy("CodeHaven is still working on the previous request.")
            self.busy = True

    def release(self):
        """Let the next request in."""
        with self.lock:
            self.busy = False

    def emit(self, event):
        """Pass a log line or a piece of streamed text to the answer being sent, if one is running."""
        if self.sink is not None:
            self.sink.put(event)

    def open(self, path, excludes=()):
        """Index a repository and start a fresh conversation about it."""
        self.claim()
        try:
            return self.index(path, excludes)
        finally:
            self.release()

    def index(self, path, excludes=()):
        """Index the folder at path and return the page's summary. The caller already holds the busy claim."""
        root = Path(path).expanduser()
        if not root.is_absolute():
            raise ValueError("Use the full path of the repository folder, such as /Users/you/dev/project.")
        workspace = Workspace(root, excludes=self.excludes + tuple(excludes))
        # A knowledge folder inside the repository holds the user's own reference notes; it is indexed when present.
        references = workspace.root / "knowledge"
        self.load(workspace, references if references.exists() else None)
        return self.summary()

    def clone(self, address):
        """Download a GitHub repository once (only with --allow-clone), then index the local copy.

        A repository cloned earlier is opened from disk without touching the network.
        """
        if not self.allow_clone:
            raise ValueError(CLONE_OFF)
        owner, name, ssh = github_repository(address)
        target = self.clone_folder / owner / name
        self.claim()
        try:
            if target.exists():
                if origin_repository(target) != (owner.lower(), name.lower()):
                    raise ValueError(f"{target} already exists and is not a copy of github.com/{owner}/{name}.")
                summary = self.index(str(target))
                return {**summary, "cloned": False, "source": f"github.com/{owner}/{name}"}
            download(owner, name, ssh, target)
            summary = self.index(str(target))
            return {**summary, "cloned": True, "source": f"github.com/{owner}/{name}"}
        finally:
            self.release()

    def open_file(self, name, data):
        """Chat about one uploaded file: a copy in a private temporary folder becomes the workspace.

        A picture of code is first read into text by Tesseract on this computer; the chat is about that text.
        """
        if ocr.image_suffix(data):
            return self.open_picture(upload_name(name), data)
        if documents.kind(name, data):
            return self.open_document(upload_name(name), data)
        if len(data) > MAX_BYTES:
            raise ValueError(f"Only UTF-8 text files up to {MAX_BYTES // 1000} KB can be uploaded.")
        if b"\0" in data:
            raise ValueError("Only text files and PNG or JPEG pictures of code can be uploaded; this file looks binary.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files can be uploaded.") from exc
        name = upload_name(name)
        self.claim()
        try:
            return self.open_copy(name, data)
        finally:
            self.release()

    def open_picture(self, name, data):
        """Read a picture of code into text with Tesseract, and open that text as the uploaded file."""
        self.claim()  # Before reading, so two pictures are never read at once.
        try:
            found = ocr.read_code(name, data)
            summary = self.open_copy(found["name"], found["text"].encode("utf-8"))
            self.picture = {key: found[key] for key in ("image", "lines", "check", "syntax")}
            return {**summary, "picture": self.picture}  # The summary was made before the picture's notes were kept.
        finally:
            self.release()

    def open_document(self, name, data):
        """Chat about a notebook or a PDF: its cells or pages are read into one text file on this computer."""
        self.claim()
        try:
            found = documents.read_document(name, data)
            summary = self.open_copy(upload_name(found["name"]), found["text"].encode("utf-8"))
            self.document = {key: found[key] for key in ("source", "kind", "note")}
            return {**summary, "document": self.document}
        finally:
            self.release()

    def open_copy(self, name, data):
        """Write the uploaded bytes into a new private temporary folder and make that folder the workspace."""
        folder = Path(tempfile.mkdtemp(prefix="codehaven-upload-"))  # Readable by this user only.
        try:
            (folder / name).write_bytes(data)
            workspace = Workspace(folder, excludes=self.excludes)
            workspace.read(name)  # Refuses names the workspace rules hide, such as --exclude patterns.
            # A file attached in the middle of a chat continues it, so "explain the picture I just sent" works.
            self.load(workspace, None, upload=name, upload_folder=folder, keep_chat=True)
            return self.summary()
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def load(self, workspace, references, *, upload=None, upload_folder=None, keep_chat=False):
        """Replace what is open: build the search index, memory, and saved chats for workspace, and a new Session."""
        turns = self.session.turns if keep_chat and self.session is not None else []
        semantic = None
        if self.embed_model:
            # An uploaded file lives in a temporary folder, so its vectors are kept in memory only.
            keep = self.remember and upload is None
            semantic = Semantic(self.model, self.embed_model, output=lambda line: self.emit({"type": "log", "text": str(line)}),
                                cache=Semantic.cache_path(workspace, self.embed_model, self.index_folder) if keep else None)
        knowledge = KnowledgeBase(references, workspace=workspace, context_budget=self.retrieval_budget, semantic=semantic)
        if semantic is not None:
            semantic.wait(SEMANTIC_WAIT)
        self.semantic = semantic
        # Only now, once the new index was built without an error, is the previous upload's temporary folder deleted.
        self.discard_upload()
        self.workspace, self.knowledge = workspace, knowledge
        self.upload, self.upload_folder = upload, upload_folder
        # An uploaded file lives in a temporary folder that is deleted, so there is nothing lasting to remember.
        self.memory = Memory(workspace, self.memory_folder) if self.remember and upload is None else None
        self.chats = Chats(workspace, self.chats_folder) if self.remember and upload is None else None
        if not turns:
            self.chat_id = new_id()  # A newly opened repository starts a new chat; an attached file continues one.
        self.session = Session(self.model, workspace, knowledge=knowledge, max_steps=self.max_steps, focus_file=upload,
                               memory=self.memory,
                               output=lambda line: self.emit({"type": "log", "text": str(line)}),
                               stream=lambda text: self.emit({"type": "text", "text": text}))
        # Earlier questions and answers carry over; evidence does not, because it quotes files no longer open.
        self.session.turns = list(turns)

    def discard_upload(self):
        """Delete the temporary copy of an uploaded file, if one is open."""
        if self.upload_folder is not None:
            shutil.rmtree(self.upload_folder, ignore_errors=True)
            self.upload = self.upload_folder = self.picture = self.document = None

    def close(self):
        """Clean up when the server stops: delete the uploaded file's temporary copy."""
        self.discard_upload()

    def model_name(self):
        """The name of the chat model in use, or the default name when the model object does not give one."""
        return getattr(self.model, "model", MODEL)

    def models(self):
        """Installed local chat models for the picker, and which one is in use."""
        try:
            found = self.model.local_models() if hasattr(self.model, "local_models") else []
        except ValueError:
            found = []
        return {"models": found, "current": self.model_name(), "default": MODEL}

    def use_model(self, name):
        """Switch to another installed chat model and return the refreshed list for the picker."""
        if not hasattr(self.model, "use"):
            raise ValueError("This model cannot be changed.")
        self.claim()
        try:
            self.model.use(name)
            return self.models()
        finally:
            self.release()

    def summary(self):
        """What the page needs to show the current state: model, settings, saved chats, and what is open."""
        meaning = None
        if self.semantic is not None:
            ready, total = self.semantic.coverage()
            meaning = {"model": self.embed_model, "ready": ready, "total": total, "failed": bool(self.semantic.failed)}
        base = {"model": self.model_name(), "clone": self.allow_clone, "remember": self.remember, "meaning": meaning,
                "chats": self.chats.listing() if self.chats is not None else [],
                "saved": len(self.memory.notes) if self.memory is not None else 0}
        workspace, knowledge = self.workspace, self.knowledge
        if workspace is None or knowledge is None:
            return {**base, "repository": None, "upload": None, "picture": None, "document": None}
        counts = knowledge.documents_by_source
        if self.upload is not None:
            # The temporary folder means nothing to the user; the file is readable even if its type is not indexed.
            return {**base, "repository": None, "upload": self.upload, "picture": self.picture,
                    "document": self.document, "files": 1,
                    "references": 0, "chunks": len(knowledge.entries), "limited": False}
        return {**base, "repository": str(workspace.root), "upload": None, "picture": None, "document": None,
                "files": counts["repo"],
                "git": (workspace.root / ".git").exists(),
                "references": counts["user"], "chunks": len(knowledge.entries), "limited": knowledge.limited}

    def forget(self):
        """Delete the answers saved for the open repository."""
        self.claim()
        try:
            if self.memory is not None:
                self.memory.forget()
            if self.chats is not None:
                self.chats.forget()
            if self.semantic is not None:
                self.semantic.forget()  # The saved vectors go too; they are rebuilt when the repository is indexed again.
            return self.summary()
        finally:
            self.release()

    def reset(self):
        """Start a new chat: the session forgets earlier questions, and later ones are saved under a new id."""
        if self.session is not None:
            self.claim()
            try:
                self.session.reset()
                self.chat_id = new_id()
            finally:
                self.release()

    def open_chat(self, chat_id):
        """Reopen a saved chat: the page shows it again, and follow-ups remember it as they did when it was live."""
        if self.chats is None or self.session is None:
            raise ValueError("Saved chats are off, or no repository is open.")
        self.claim()
        try:
            chat = self.chats.get(str(chat_id))
            self.session.reset()
            self.session.turns = [(turn["question"], turn["answer"]) for turn in chat["turns"]][-self.session.TURNS:]
            self.chat_id = chat["id"]
            return {"id": chat["id"], "title": chat["title"], "updated": chat["updated"], "turns": chat["turns"],
                    "remembered": len(self.session.turns), "memory": self.session.TURNS}
        finally:
            self.release()

    def delete_chat(self, chat_id):
        """Delete one saved chat and return the chats that remain."""
        if self.chats is None:
            raise ValueError("Saved chats are off, or no repository is open.")
        self.claim()
        try:
            self.chats.delete(str(chat_id))
            if self.chat_id == chat_id:
                self.chat_id = new_id()  # Further questions in the open chat start a new saved one.
            return {"chats": self.chats.listing()}
        finally:
            self.release()

    def ask(self, question):
        """Yield the events of one answer: log lines, streamed text, and a final done or error event."""
        session, chats, chat_id = self.session, self.chats, self.chat_id
        if session is None:
            raise ValueError("Choose a repository or upload a file first.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Type a question first.")
        self.claim()
        # The answer runs on a worker thread; its events reach this request thread, which writes them out, through a queue.
        events = queue.Queue()
        self.sink = events

        def work():
            """Answer the question, save the chat, and queue the final done or error event."""
            try:
                code = session.ask(question.strip())
                state = session.last_state
                answer = state.get("final_text")
                if chats is not None and answer:
                    with contextlib.suppress(OSError):  # A chat that cannot be saved must not cost the answer.
                        chats.record(chat_id, question, answer, state.get("issues", []))
                events.put({"type": "done", "exit_code": code, "steps": state.get("steps", 0),
                            "answer": answer if answer is not None else "",
                            "complete": answer is not None, "issues": state.get("issues", []),
                            "remembered": len(session.turns), "memory": session.TURNS,
                            "kept": session.saved, "saved": len(session.memory.notes) if session.memory else 0,
                            "chats": chats.listing() if chats is not None else []})
            except Exception as exc:  # Reported to the page; the server keeps running.
                events.put({"type": "error", "message": str(exc) or type(exc).__name__})
            finally:
                # Always sent, even after an error: None tells the loop below that the answer is over.
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        try:
            while (event := events.get()) is not None:
                yield event
        finally:
            # Also reached when the page goes away and the generator is closed early, so the app never stays busy.
            self.sink = None
            self.release()

    def file(self, path, start=1, end=None):
        """Lines of a repository file around a cited range, through the same rules as the agent's reads."""
        if self.workspace is None:
            raise ValueError("Choose a repository or upload a file first.")
        lines = self.workspace.read(path.removeprefix("repo/")).splitlines()
        start = max(1, int(start))
        end = max(start, int(end or start))
        first = max(1, start - FILE_CONTEXT)
        last = min(len(lines), end + FILE_CONTEXT, first + FILE_LINES - 1)
        return {"path": path.removeprefix("repo/"), "start": start, "end": end, "total": len(lines),
                "first": first, "lines": lines[first - 1:last]}

    def text(self):
        """The whole uploaded file, so the page can save the text read from a picture."""
        if self.workspace is None or self.upload is None:
            raise ValueError("Upload a file first.")
        return {"name": self.upload, "text": self.workspace.read(self.upload)}

    def opened(self):
        """The open workspace, or ValueError when nothing is open yet."""
        if self.workspace is None:
            raise ValueError("Choose a repository or upload a file first.")
        return self.workspace

    def selection(self, paths):
        """The files a study aid is about: the chosen ones, or the uploaded file when one is open."""
        return [self.upload] if self.upload is not None and not paths else paths

    def files(self):
        """Files that can be chosen for a map or flashcards: the same listing the agent sees."""
        names = self.opened().files()
        return {"files": names, "limited": len(names) >= MAX_FILES,
                "limits": {"map": study.MAP_FILES, "flashcards": study.CARD_FILES, "cards": study.CARDS,
                           "default_cards": study.DEFAULT_CARDS}}

    def structure(self, paths):
        """The structure map of the chosen files. Static analysis only, so it needs no model and no waiting turn."""
        workspace, knowledge = self.opened(), self.knowledge
        return study.structure(workspace, knowledge, self.selection(paths))

    def flashcards(self, paths, count=study.DEFAULT_CARDS):
        """Yield the events of making flashcards for the chosen files; the model is busy until they end."""
        workspace = self.opened()
        self.claim()
        try:
            yield from study.flashcards(self.model, workspace, self.selection(paths), count)
        finally:
            self.release()


def upload_name(name):
    """A safe file name for an uploaded copy, keeping the original name where it can.

    Only the last part of the name is kept, and characters other than letters, digits, dot, dash, and
    underscore become underscores, so citations such as notes_v2.py:12 stay clickable.
    """
    name = re.split(r"[\\/]", str(name or ""))[-1].strip()
    name = re.sub(r"[^\w.-]", "_", name, flags=re.ASCII)[-UPLOAD_NAME_CHARACTERS:]
    if not name.strip("._"):
        raise ValueError("The uploaded file needs a name.")
    if name.startswith("."):
        raise ValueError("Hidden files such as .env cannot be uploaded.")
    return name


def github_repository(address):
    """(owner, name, ssh) for a GitHub repository address, or ValueError.

    Accepts https://github.com/owner/name (with or without https://, .git, or a trailing /tree/... part) and
    git@github.com:owner/name.git. Addresses carrying a user name, password, or token are refused, so no secret
    can reach the clone's settings or the page's list of recent repositories.
    """
    text = str(address or "").strip()
    ssh = re.fullmatch(r"(?:ssh://)?git@github\.com[:/]([^/\s]+)/([^/\s]+?)/?", text, flags=re.IGNORECASE)
    if ssh:
        owner, name = ssh.groups()
    else:
        url = urlsplit(text if "://" in text else "https://" + text)
        if url.scheme.lower() not in ("https", "http") or "@" in url.netloc:
            raise ValueError("Use a GitHub address such as https://github.com/owner/repository, without a password "
                             "or token. Private repositories use the sign-in git already has on this computer.")
        if url.netloc.lower() not in ("github.com", "www.github.com"):
            raise ValueError("Only github.com addresses can be cloned, such as https://github.com/owner/repository.")
        parts = [part for part in url.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("Include the owner and the repository, such as https://github.com/owner/repository.")
        owner, name = parts[0], parts[1]
    name = name[:-4] if name.lower().endswith(".git") else name
    if not OWNER.fullmatch(owner) or not REPOSITORY.fullmatch(name) or name.strip(".") == "":
        raise ValueError("That does not look like a GitHub repository address.")
    return owner, name, bool(ssh)


def origin_repository(folder):
    """(owner, name) in lower case of the GitHub repository a local clone came from, read from disk only."""
    git = shutil.which("git")
    if git is None or not (folder / ".git").exists():
        return None
    try:
        found = subprocess.run([git, "-C", str(folder), "config", "--get", "remote.origin.url"], capture_output=True,
                               text=True, timeout=10, stdin=subprocess.DEVNULL)
        owner, name, _ = github_repository(found.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return owner.lower(), name.lower()


def folder_bytes(folder):
    """Bytes on disk under folder, counting links as themselves; used to stop a download that grows too large."""
    total = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, name)).st_size
    return total


def download(owner, name, ssh, target):
    """Clone the latest snapshot of one GitHub repository into target: the only network use in CodeHaven.

    git gets no password from CodeHaven and may not prompt for one, so only public repositories and the sign-in
    git already has (a credential helper or an SSH key) work. Submodules, Git LFS files, tags, history, hooks,
    and symlinks are left out, and the download stops after CLONE_SECONDS or CLONE_BYTES.
    """
    git = shutil.which("git")
    if git is None:
        raise ValueError("Cloning needs git, which is not installed on this computer.")
    url = f"git@github.com:{owner}/{name}.git" if ssh else f"https://github.com/{owner}/{name}.git"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))  # Hidden until the clone is complete.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")  # Never wait for a passphrase nobody can type.
    # Only the https and ssh transports are allowed, and the clone can run no hooks, LFS filters, or template files.
    command = [git, "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "-c", "protocol.ssh.allow=always",
               "-c", f"core.hooksPath={os.devnull}", "-c", "core.symlinks=false", "-c", "filter.lfs.required=false",
               "-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=",
               "clone", "--depth", "1", "--single-branch", "--no-tags", "--template=", "--quiet", "--", url, str(partial)]
    try:
        # git's messages go to a temporary file, not a pipe, because nothing reads them while the loop below polls.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                       env=env)
            deadline, stopped = time.monotonic() + CLONE_SECONDS, None
            # Polled rather than waited for, so that both the time limit and the size limit can stop the download.
            while process.poll() is None:
                if time.monotonic() > deadline:
                    stopped = f"The download took longer than {CLONE_SECONDS // 60} minutes and was stopped."
                elif folder_bytes(partial) > CLONE_BYTES:
                    stopped = f"The repository is larger than {CLONE_BYTES // 1_000_000} MB and was not downloaded."
                if stopped:
                    process.kill()
                    process.wait()
                    raise ValueError(stopped)
                time.sleep(0.5)
            errors.seek(0)
            message = errors.read().decode("utf-8", "replace")
        if process.returncode != 0:
            raise ValueError(clone_error(message, owner, name))
        # The finished clone appears under its real name in one step; a failed one never does.
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def clone_error(message, owner, name):
    """Turn git's error output into a plain-English message for the page."""
    lowered = message.lower()
    if "could not resolve host" in lowered or "could not resolve hostname" in lowered:
        return "GitHub could not be reached. Check the internet connection, or open a copy already on this computer."
    if any(text in lowered for text in ("repository not found", "authentication failed", "could not read username",
                                        "permission denied", "terminal prompts disabled")):
        return (f"GitHub did not share github.com/{owner}/{name}: it does not exist, or it is private and git on "
                "this computer is not signed in to an account that can read it. CodeHaven never asks for your "
                "GitHub password. Sign git in (for example with `gh auth login`), or clone it yourself and index "
                "the folder.")
    last = [line.strip() for line in message.splitlines() if line.strip()]
    return "git could not clone the repository" + (f": {last[-1]}" if last else ".")


def directories(path=""):
    """Folders inside path, for choosing a repository. Hidden folders are left out."""
    folder = Path(path).expanduser() if path else Path.home()
    if not folder.is_absolute() or not folder.is_dir():
        raise ValueError("That folder does not exist.")
    folder = folder.resolve()
    entries = []
    try:
        names = sorted(os.scandir(folder), key=lambda entry: entry.name.lower())
    except PermissionError as exc:
        raise ValueError("This folder cannot be opened.") from exc
    for entry in names:
        if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
            continue
        entries.append({"name": entry.name, "path": str(Path(entry.path)),
                        "repository": Path(entry.path, ".git").exists()})
        if len(entries) >= DIRECTORY_ENTRIES:
            break
    return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None,
            "repository": (folder / ".git").exists(), "entries": entries}


def make_handler(app, token, port_holder):
    """The request handler class for one app and token. port_holder is filled in once the server knows its port."""
    nonce_holder = {}

    class Handler(BaseHTTPRequestHandler):
        """Serves the page and its JSON API. Every request must pass authorized() before anything else happens."""

        server_version = "CodeHaven"
        sys_version = ""

        def log_message(self, *_args):
            """Log nothing."""
            pass  # Requests carry file paths and questions; keep them out of the terminal.

        def allowed_host(self):
            """Whether the Host header names this server on this machine.

            A website whose own name is made to resolve to 127.0.0.1 (DNS rebinding) still sends its own name here,
            so it is refused even though the request reaches this server.
            """
            port = port_holder["port"]
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def send(self, status, body, content_type="application/json"):
            """Send a complete response: bytes as they are (the page), anything else as JSON."""
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.security_headers()
            self.end_headers()
            self.wfile.write(data)

        def security_headers(self):
            """Headers on every response: no caching, sniffing, or framing, and only the script with this page's nonce runs."""
            nonce = nonce_holder.get("nonce", "")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            # Nothing may load from anywhere but this server: the page stays offline even if a link sneaks in.
            self.send_header("Content-Security-Policy",
                             f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

        def refuse(self, status, message):
            """Send an error as JSON, which the page shows to the user."""
            self.send(status, {"error": message})

        def authorized(self, query):
            """Check the Host header, then the secret token. Sends a 403 and returns False if either is wrong."""
            if not self.allowed_host():
                self.refuse(403, "Unknown host.")
                return False
            # The page sends the token in a header; only the first page load carries it in the link's query string.
            sent = self.headers.get("X-CodeHaven-Token") or query.get("token", [""])[0]
            # compare_digest takes the same time wherever the first difference is, so timing reveals nothing about the token.
            if not secrets.compare_digest(sent, token):
                self.refuse(403, "Open the link CodeHaven printed in the terminal.")
                return False
            return True

        def body(self):
            """The request's JSON object. Bodies over REQUEST_BYTES are refused without being read."""
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= REQUEST_BYTES:  # A negative length would make the read wait for the connection to close.
                self.close_connection = True  # The body is not read, so the connection cannot be reused.
                raise ValueError("Request too large.")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object.")
            return data

        def do_GET(self):
            """Serve the page itself and the routes that only read."""
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if not self.authorized(query):
                return
            try:
                if url.path == "/":
                    # A new nonce for every page load: it goes into the page's script tag and into the policy header.
                    nonce_holder["nonce"] = secrets.token_urlsafe(16)
                    page = PAGE.read_text(encoding="utf-8").replace("__NONCE__", nonce_holder["nonce"])
                    self.send(200, page.encode(), "text/html")
                elif url.path == "/api/status":
                    self.send(200, app.summary())
                elif url.path == "/api/models":
                    self.send(200, app.models())
                elif url.path == "/api/files":
                    self.send(200, app.files())
                elif url.path == "/api/text":
                    self.send(200, app.text())
                elif url.path == "/api/dirs":
                    self.send(200, directories(query.get("path", [""])[0]))
                elif url.path == "/api/file":
                    self.send(200, app.file(query.get("path", [""])[0], query.get("start", ["1"])[0],
                                            query.get("end", [""])[0] or None))
                else:
                    self.refuse(404, "Not found.")
            except (ValueError, OSError, UnicodeError) as exc:
                self.refuse(400, str(exc))

        def upload(self):
            """The raw bytes of an uploaded file. Bodies over the picture limit are refused without being read."""
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= ocr.IMAGE_BYTES:  # Text files have a smaller limit, checked once the bytes are seen.
                self.close_connection = True  # The body is not read, so the connection cannot be reused.
                raise ValueError(f"Only text files up to {MAX_BYTES // 1000} KB, and pictures, notebooks, and PDFs up "
                                 f"to {ocr.IMAGE_BYTES // 1_000_000} MB, can be uploaded.")
            return self.rfile.read(length)

        def do_POST(self):
            """Handle an upload, and the JSON routes that open something, change state, or start an answer."""
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            # The empty query means a POST must carry the token in its header; one in the address is not accepted.
            if not self.authorized({}):
                return
            try:
                if url.path == "/api/upload":
                    self.send(200, app.open_file(query.get("name", [""])[0], self.upload()))
                    return
                data = self.body()
                if url.path == "/api/open":
                    self.send(200, app.open(str(data.get("path", "")), data.get("excludes") or ()))
                elif url.path == "/api/clone":
                    self.send(200, app.clone(str(data.get("address", ""))))
                elif url.path == "/api/model":
                    self.send(200, app.use_model(str(data.get("name", ""))))
                elif url.path == "/api/forget":
                    self.send(200, app.forget())
                elif url.path == "/api/chat":
                    self.send(200, app.open_chat(data.get("id", "")))
                elif url.path == "/api/chat/delete":
                    self.send(200, app.delete_chat(data.get("id", "")))
                elif url.path == "/api/reset":
                    app.reset()
                    self.send(200, {"ok": True})
                elif url.path == "/api/ask":
                    self.stream(app.ask(data.get("question", "")))
                elif url.path == "/api/map":
                    self.send(200, app.structure(data.get("paths") or []))
                elif url.path == "/api/flashcards":
                    self.stream(app.flashcards(data.get("paths") or [], data.get("count", study.DEFAULT_CARDS)), drain=False)
                else:
                    self.refuse(404, "Not found.")
            except Busy as exc:
                self.refuse(409, str(exc))
            except (ValueError, OSError, UnicodeError) as exc:
                self.refuse(400, str(exc))

        def stream(self, events, drain=True):
            """Send events as lines of JSON. If the page goes away, an answer is drained so its worker thread
            can finish; work done by the generator itself, such as flashcards, is stopped instead."""
            first = next(events)  # Raises Busy or ValueError before any response is sent.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.security_headers()
            self.end_headers()
            try:
                self.wfile.write(json.dumps(first).encode() + b"\n")
                self.wfile.flush()
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                if drain:
                    for _ in events:  # The page closed; let the answer finish so the next question can start.
                        pass
            finally:
                events.close()  # Releases the model if the generator was left unfinished.

    return Handler


def make_server(app, port=0):
    """A server bound to 127.0.0.1, its secret token, and its URL. port=0 picks a free port."""
    token = secrets.token_urlsafe(24)
    port_holder = {}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, token, port_holder))
    # Request threads must not keep the program alive once the server is stopped.
    server.daemon_threads = True
    # With port=0 the port is known only now; the handler reads it for its Host check.
    port_holder["port"] = server.server_address[1]
    return server, token, f"http://127.0.0.1:{port_holder['port']}/?token={token}"


def serve(model, *, port=0, open_browser=True, repository=None, max_steps=12, retrieval_budget=4000, excludes=(),
          allow_clone=False, clone_folder=None, remember=False, memory_folder=None, embed_model=None, index_folder=None,
          chats_folder=None):
    """Run the GUI until Ctrl+C. Returns an exit code like the CLI's."""
    app = App(model, max_steps=max_steps, retrieval_budget=retrieval_budget, excludes=excludes,
              allow_clone=allow_clone, clone_folder=clone_folder, remember=remember, memory_folder=memory_folder,
              embed_model=embed_model, index_folder=index_folder, chats_folder=chats_folder)
    if repository:
        summary = app.open(str(Path(repository).expanduser().resolve()))
        print(f"Indexed {summary['files']} files from {summary['repository']}.")
    server, _, url = make_server(app, port)
    print(f"CodeHaven is running at:\n  {url}\nOnly this machine can reach it. Press Ctrl+C to stop.", flush=True)
    if allow_clone:
        print(f"Cloning from GitHub is on: git may download repositories you ask for into {app.clone_folder}. "
              "Nothing is uploaded. Restart without --allow-clone to stay fully offline.", flush=True)
    if remember:
        print("Long-term memory is on: checked answers and your chats are saved on this computer, so answers can be "
              "recalled and chats reopened in later sessions. Restart with --no-remember to save nothing.", flush=True)
    print(f"Meaning search is on, using the local embedding model {embed_model}." if embed_model else
          "Meaning search is off: answers use key-word search. To turn it on, install a local embedding model once "
          "while online (ollama pull nomic-embed-text).", flush=True)
    if open_browser:
        webbrowser.open(url)

    def stop(_signum, _frame):
        """Treat a kill signal like Ctrl+C, so the cleanup below still runs."""
        raise KeyboardInterrupt  # A plain kill also deletes the uploaded file's temporary copy.
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        app.close()
    return 0
