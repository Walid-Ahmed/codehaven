"""Start-up help: say what is missing and how to get it, and start a local Ollama server when asked.

Nothing here uses the internet. The checks look for programs on this computer and ask Ollama on its loopback
address. --start-ollama runs the installed `ollama serve` bound to 127.0.0.1 with its cloud features off, the
same way scripts/start_ollama.py does, and stops it again when CodeHaven exits.
"""
import importlib.util
import os
import shutil
import subprocess
import time
import urllib.parse

from .model import MODEL

START_SECONDS = 20  # How long to wait for a server that was just started.


def running(model):
    """Whether an Ollama server answers at the model's loopback endpoint."""
    try:
        return isinstance(model.request("/api/tags").get("models"), list)
    except ValueError:
        return False


def start_ollama(model, output=print):
    """Start `ollama serve` for this run if no server is answering. Returns the process to stop later, or None.

    The server listens on the loopback address of --endpoint only and has cloud features off. A server that is
    already running is left as it is, and so is its cloud setting.
    """
    if running(model):
        output("Ollama is already running; using it as it is.")
        return None
    program = shutil.which("ollama")
    if program is None:
        raise ValueError("Ollama is not installed, or is not on PATH. Install it from https://ollama.com/, then run this again.")
    address = urllib.parse.urlsplit(model.endpoint)
    env = {**os.environ, "OLLAMA_NO_CLOUD": "1", "OLLAMA_HOST": f"{address.hostname}:{address.port or 11434}"}
    process = subprocess.Popen([program, "serve"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError(f"Ollama stopped as soon as it started (exit code {process.returncode}). Run `ollama serve` to see why.")
        if running(model):
            output(f"Started Ollama on {env['OLLAMA_HOST']} with cloud features off. It stops when CodeHaven exits.")
            return process
        time.sleep(0.25)
    stop_ollama(process)
    raise ValueError(f"Ollama did not answer within {START_SECONDS} seconds of starting.")


def stop_ollama(process):
    """Stop a server started by start_ollama, waiting briefly before forcing it."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def missing_server(model):
    """Why the model could not be reached, as lines that each say what to do next."""
    if shutil.which("ollama") is None:
        return ["Ollama: not installed. Install it from https://ollama.com/ (needed once, while online).",
                f"Then download the model once: ollama pull {model.model}"]
    if not running(model):
        return [f"Ollama: installed, but no server answers at {model.endpoint}.",
                "Start it for this run by adding --start-ollama, for example: codehaven --gui --start-ollama",
                "Or leave one running in another terminal: ollama serve"]
    try:
        installed = [found["name"] for found in model.local_models()]
    except ValueError:
        installed = []
    lines = [f"Ollama: running, but the model {model.model} is not installed.",
             f"Download it once, while online: ollama pull {model.model}"]
    if installed:
        lines.append("Or use one that is installed: --model " + installed[0]
                     + (f" (also installed: {', '.join(installed[1:4])})" if installed[1:] else ""))
    return lines


def extras(docker_image=None):
    """One line per optional tool: found, or what it adds and how to install it. None of them is needed to chat."""
    lines = []
    lines.append("Pictures of code: Tesseract found." if shutil.which("tesseract") else
                 "Pictures of code: off. Install Tesseract to turn it on (brew install tesseract, or apt install tesseract-ocr).")
    pdf = importlib.util.find_spec("pypdf") is not None or shutil.which("pdftotext")
    lines.append("PDFs: a PDF reader was found." if pdf else
                 "PDFs: off. Install one to turn it on (python -m pip install pypdf, or brew install poppler).")
    lines.append("Isolated tests: Docker found." + (f" Image expected: {docker_image}." if docker_image else "")
                 if shutil.which("docker") else
                 "Isolated tests: off. Install Docker to let CodeHaven run a project's tests in a container.")
    lines.append("Git changes: git found." if shutil.which("git") else "Git changes: off. Install git to ask about your changes.")
    return lines


def default_model_hint(model):
    """A line for --doctor when the model in use is not the tested default."""
    return (f"Note: {model.model} is not the tested default ({MODEL}); answers may differ." if model.model != MODEL else "")


# The model to suggest for a computer's memory, smallest first: (at least this many GB, model, why).
# Measured in docs/EVALS.md: the 7B coder model is the tested default; the 14B one is as often right and cites more
# carefully; the 0.5B model was right about a third of the time, so it is never suggested.
SUGGESTIONS = [
    (0, "qwen2.5-coder:3b", "smaller than the tested default; expect more wrong or unverified answers"),
    (12, MODEL, "the tested default"),
    (32, "qwen2.5:14b", "as often right as the default and more careful with citations, about half as fast"),
]


def memory_gb():
    """This computer's memory in GB, or None when it cannot be told. Read from the system, never from the network."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3)
    except (OSError, ValueError):
        pass
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            """The fields of MEMORYSTATUSEX that Windows fills in."""
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.total / 1024 ** 3)
    return None


def suggested_model(gigabytes=None):
    """(model, why) to suggest for a computer with this much memory (this one's when not given)."""
    gigabytes = memory_gb() if gigabytes is None else gigabytes
    if gigabytes is None:
        return MODEL, "the tested default (this computer's memory could not be read)"
    choice = [row for row in SUGGESTIONS if gigabytes >= row[0]][-1]
    return choice[1], f"{choice[2]}; this computer has {gigabytes} GB of memory"


def setup(model, ask=input, output=print, run=subprocess.run):
    """Guided first run: check Ollama, start it if needed, and download a model only if the user says yes.

    This is the one place CodeHaven uses the internet, and only through `ollama pull`, after asking. Returns 0 when
    CodeHaven is ready to use, 1 otherwise.
    """
    program = shutil.which("ollama")
    if program is None:
        output("Ollama is not installed. Install it from https://ollama.com/ (once, while online), then run codehaven --setup again.")
        return 1
    output("Ollama: installed.")
    server = None
    if not running(model):
        server = start_ollama(model, output)
    try:
        installed = {found["name"] for found in model.local_models()}
        suggestion, why = suggested_model()
        output(f"Suggested model for this computer: {suggestion} ({why}).")
        wanted = model.model if model.model in installed else suggestion
        if wanted not in installed:
            answer = ask(f"Download {wanted} now with ollama pull? It is a few GB and needs the internet once. [y/N] ")
            if answer.strip().lower() != "y":
                output(f"Nothing was downloaded. When you are ready: ollama pull {wanted}")
                return 1
            done = run([program, "pull", wanted], env={**os.environ, "OLLAMA_NO_CLOUD": "1"})
            if done.returncode:
                output(f"ollama pull {wanted} did not finish (exit code {done.returncode}). Check the connection and try again.")
                return 1
        output(f"Chat model: {wanted} is installed.")
        embedders = [name for name in installed if "embed" in name]
        if not embedders:
            answer = ask("Also download nomic-embed-text (about 270 MB) so search works by meaning too? [y/N] ")
            if answer.strip().lower() == "y":
                run([program, "pull", "nomic-embed-text"], env={**os.environ, "OLLAMA_NO_CLOUD": "1"})
        for line in extras():
            output(line)
        command = "codehaven --gui --start-ollama" + (f" --model {wanted}" if wanted != MODEL else "")
        output(f"Ready. From now on everything works offline. Start with: {command}")
        return 0
    finally:
        stop_ollama(server)
