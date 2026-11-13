"""CodeHaven in the browser: pick a repository, index it, and chat about its code.

The server listens on 127.0.0.1 only and serves one self-contained page: no scripts, styles, or fonts come
from the internet, and the page's Content-Security-Policy forbids loading any. Every request must carry the
secret token from the link the server prints, and a Host header naming this machine, so other websites open
in the same browser cannot use the server to read files. Like the rest of CodeHaven, it never changes files.

A single file can be uploaded instead of choosing a folder. The page sends it to this server only; a copy is
kept in a private temporary folder, which is the workspace for that chat, and deleted when another repository
or file is opened or the server stops. The original file is never touched.
"""
import json
import os
import queue
import re
import secrets
import shutil
import signal
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .agent import Session
from .knowledge import KnowledgeBase
from .model import MODEL
from .workspace import MAX_BYTES, Workspace

PAGE = Path(__file__).with_name("gui.html")
FILE_CONTEXT, FILE_LINES = 6, 400  # Lines shown around a citation, and the most one request returns.
DIRECTORY_ENTRIES = 400
REQUEST_BYTES = 64_000
UPLOAD_NAME_CHARACTERS = 120


class Busy(Exception):
    """Another indexing run or question is still in progress."""


class App:
    """One repository and one conversation at a time, shared by the request threads."""

    def __init__(self, model, *, max_steps=12, retrieval_budget=4000, excludes=()):
        self.model, self.max_steps, self.retrieval_budget = model, max_steps, retrieval_budget
        self.excludes = tuple(excludes)
        self.lock = threading.Lock()
        self.busy = False
        self.workspace = self.knowledge = self.session = None
        self.upload = self.upload_folder = None  # The uploaded file's name and its private temporary folder.
        self.sink = None

    def claim(self):
        with self.lock:
            if self.busy:
                raise Busy("CodeHaven is still working on the previous request.")
            self.busy = True

    def release(self):
        with self.lock:
            self.busy = False

    def emit(self, event):
        if self.sink is not None:
            self.sink.put(event)

    def open(self, path, excludes=()):
        """Index a repository and start a fresh conversation about it."""
        self.claim()
        try:
            root = Path(path).expanduser()
            if not root.is_absolute():
                raise ValueError("Use the full path of the repository folder, such as /Users/you/dev/project.")
            workspace = Workspace(root, excludes=self.excludes + tuple(excludes))
            references = workspace.root / "knowledge"
            self.load(workspace, references if references.exists() else None)
            return self.summary()
        finally:
            self.release()

    def open_file(self, name, data):
        """Chat about one uploaded file: a copy in a private temporary folder becomes the workspace."""
        if len(data) > MAX_BYTES:
            raise ValueError(f"Only UTF-8 text files up to {MAX_BYTES // 1000} KB can be uploaded.")
        if b"\0" in data:
            raise ValueError("Only text files can be uploaded; this file looks binary.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files can be uploaded.") from exc
        name = upload_name(name)
        self.claim()
        folder = Path(tempfile.mkdtemp(prefix="codehaven-upload-"))  # Readable by this user only.
        try:
            (folder / name).write_bytes(data)
            workspace = Workspace(folder, excludes=self.excludes)
            workspace.read(name)  # Refuses names the workspace rules hide, such as --exclude patterns.
            self.load(workspace, None, upload=name, upload_folder=folder)
            return self.summary()
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            self.release()

    def load(self, workspace, references, *, upload=None, upload_folder=None):
        knowledge = KnowledgeBase(references, workspace=workspace, context_budget=self.retrieval_budget)
        self.discard_upload()
        self.workspace, self.knowledge = workspace, knowledge
        self.upload, self.upload_folder = upload, upload_folder
        self.session = Session(self.model, workspace, knowledge=knowledge, max_steps=self.max_steps, focus_file=upload,
                               output=lambda line: self.emit({"type": "log", "text": str(line)}),
                               stream=lambda text: self.emit({"type": "text", "text": text}))

    def discard_upload(self):
        """Delete the temporary copy of an uploaded file, if one is open."""
        if self.upload_folder is not None:
            shutil.rmtree(self.upload_folder, ignore_errors=True)
            self.upload = self.upload_folder = None

    def close(self):
        self.discard_upload()

    def model_name(self):
        return getattr(self.model, "model", MODEL)

    def models(self):
        """Installed local chat models for the picker, and which one is in use."""
        try:
            found = self.model.local_models() if hasattr(self.model, "local_models") else []
        except ValueError:
            found = []
        return {"models": found, "current": self.model_name(), "default": MODEL}

    def use_model(self, name):
        if not hasattr(self.model, "use"):
            raise ValueError("This model cannot be changed.")
        self.claim()
        try:
            self.model.use(name)
            return self.models()
        finally:
            self.release()

    def summary(self):
        workspace, knowledge = self.workspace, self.knowledge
        if workspace is None or knowledge is None:
            return {"model": self.model_name(), "repository": None, "upload": None}
        counts = knowledge.documents_by_source
        if self.upload is not None:
            # The temporary folder means nothing to the user; the file is readable even if its type is not indexed.
            return {"model": self.model_name(), "repository": None, "upload": self.upload, "files": 1,
                    "references": 0, "chunks": len(knowledge.entries), "limited": False}
        return {"model": self.model_name(), "repository": str(workspace.root), "upload": None, "files": counts["repo"],
                "references": counts["user"], "chunks": len(knowledge.entries), "limited": knowledge.limited}

    def reset(self):
        if self.session is not None:
            self.claim()
            try:
                self.session.reset()
            finally:
                self.release()

    def ask(self, question):
        """Yield the events of one answer: log lines, streamed text, and a final done or error event."""
        session = self.session
        if session is None:
            raise ValueError("Choose a repository or upload a file first.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Type a question first.")
        self.claim()
        events = queue.Queue()
        self.sink = events

        def work():
            try:
                code = session.ask(question.strip())
                state = session.last_state
                answer = state.get("final_text")
                events.put({"type": "done", "exit_code": code, "steps": state.get("steps", 0),
                            "answer": answer if answer is not None else "",
                            "complete": answer is not None, "issues": state.get("issues", []),
                            "remembered": len(session.turns), "memory": session.TURNS})
            except Exception as exc:  # Reported to the page; the server keeps running.
                events.put({"type": "error", "message": str(exc) or type(exc).__name__})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        try:
            while (event := events.get()) is not None:
                yield event
        finally:
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
    nonce_holder = {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "CodeHaven"
        sys_version = ""

        def log_message(self, *_args):
            pass  # Requests carry file paths and questions; keep them out of the terminal.

        def allowed_host(self):
            port = port_holder["port"]
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.security_headers()
            self.end_headers()
            self.wfile.write(data)

        def security_headers(self):
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
            self.send(status, {"error": message})

        def authorized(self, query):
            if not self.allowed_host():
                self.refuse(403, "Unknown host.")
                return False
            sent = self.headers.get("X-CodeHaven-Token") or query.get("token", [""])[0]
            if not secrets.compare_digest(sent, token):
                self.refuse(403, "Open the link CodeHaven printed in the terminal.")
                return False
            return True

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > REQUEST_BYTES:
                raise ValueError("Request too large.")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object.")
            return data

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if not self.authorized(query):
                return
            try:
                if url.path == "/":
                    nonce_holder["nonce"] = secrets.token_urlsafe(16)
                    page = PAGE.read_text(encoding="utf-8").replace("__NONCE__", nonce_holder["nonce"])
                    self.send(200, page.encode(), "text/html")
                elif url.path == "/api/status":
                    self.send(200, app.summary())
                elif url.path == "/api/models":
                    self.send(200, app.models())
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
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= MAX_BYTES:
                self.close_connection = True  # The body is not read, so the connection cannot be reused.
                raise ValueError(f"Only UTF-8 text files up to {MAX_BYTES // 1000} KB can be uploaded.")
            return self.rfile.read(length)

        def do_POST(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if not self.authorized({}):
                return
            try:
                if url.path == "/api/upload":
                    self.send(200, app.open_file(query.get("name", [""])[0], self.upload()))
                    return
                data = self.body()
                if url.path == "/api/open":
                    self.send(200, app.open(str(data.get("path", "")), data.get("excludes") or ()))
                elif url.path == "/api/model":
                    self.send(200, app.use_model(str(data.get("name", ""))))
                elif url.path == "/api/reset":
                    app.reset()
                    self.send(200, {"ok": True})
                elif url.path == "/api/ask":
                    self.stream(app.ask(data.get("question", "")))
                else:
                    self.refuse(404, "Not found.")
            except Busy as exc:
                self.refuse(409, str(exc))
            except (ValueError, OSError, UnicodeError) as exc:
                self.refuse(400, str(exc))

        def stream(self, events):
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
                for _ in events:  # The page closed; let the answer finish so the next question can start.
                    pass

    return Handler


def make_server(app, port=0):
    """A server bound to 127.0.0.1, its secret token, and its URL. port=0 picks a free port."""
    token = secrets.token_urlsafe(24)
    port_holder = {}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, token, port_holder))
    server.daemon_threads = True
    port_holder["port"] = server.server_address[1]
    return server, token, f"http://127.0.0.1:{port_holder['port']}/?token={token}"


def serve(model, *, port=0, open_browser=True, repository=None, max_steps=12, retrieval_budget=4000, excludes=()):
    """Run the GUI until Ctrl+C. Returns an exit code like the CLI's."""
    app = App(model, max_steps=max_steps, retrieval_budget=retrieval_budget, excludes=excludes)
    if repository:
        summary = app.open(str(Path(repository).expanduser().resolve()))
        print(f"Indexed {summary['files']} files from {summary['repository']}.")
    server, _, url = make_server(app, port)
    print(f"CodeHaven is running at:\n  {url}\nOnly this machine can reach it. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)

    def stop(_signum, _frame):
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
