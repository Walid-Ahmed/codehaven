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
