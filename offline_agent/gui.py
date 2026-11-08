"""CodeHaven in the browser: pick a repository, index it, and chat about its code.

The server listens on 127.0.0.1 only and serves one self-contained page: no scripts, styles, or fonts come
from the internet, and the page's Content-Security-Policy forbids loading any. Every request must carry the
secret token from the link the server prints, and a Host header naming this machine, so other websites open
in the same browser cannot use the server to read files. Like the rest of CodeHaven, it never changes files.
"""
import json
import os
import queue
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .agent import Session
from .knowledge import KnowledgeBase
from .model import MODEL
from .workspace import Workspace

PAGE = Path(__file__).with_name("gui.html")
FILE_CONTEXT, FILE_LINES = 6, 400  # Lines shown around a citation, and the most one request returns.
DIRECTORY_ENTRIES = 400
REQUEST_BYTES = 64_000


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
            knowledge = KnowledgeBase(references if references.exists() else None, workspace=workspace,
                                      context_budget=self.retrieval_budget)
            self.workspace, self.knowledge = workspace, knowledge
            self.session = Session(self.model, workspace, knowledge=knowledge, max_steps=self.max_steps,
                                   output=lambda line: self.emit({"type": "log", "text": str(line)}),
                                   stream=lambda text: self.emit({"type": "text", "text": text}))
            return self.summary()
        finally:
            self.release()

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
            return {"model": self.model_name(), "repository": None}
        counts = knowledge.documents_by_source
        return {"model": self.model_name(), "repository": str(workspace.root), "files": counts["repo"],
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
            raise ValueError("Choose a repository first.")
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
                            "complete": answer is not None, "issues": state.get("issues", [])})
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
            raise ValueError("Choose a repository first.")
        lines = self.workspace.read(path.removeprefix("repo/")).splitlines()
        start = max(1, int(start))
        end = max(start, int(end or start))
        first = max(1, start - FILE_CONTEXT)
        last = min(len(lines), end + FILE_CONTEXT, first + FILE_LINES - 1)
        return {"path": path.removeprefix("repo/"), "start": start, "end": end, "total": len(lines),
                "first": first, "lines": lines[first - 1:last]}


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

        def do_POST(self):
            url = urlsplit(self.path)
            if not self.authorized({}):
                return
            try:
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
