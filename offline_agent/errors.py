"""Plain-language messages for problems, so the terminal and the browser page never show a raw Python traceback.

Expected problems (Ollama not running, a model not installed, a file that cannot be read) are raised as ValueError
or OSError with a message written for the user, and are shown as they are. Anything else is a problem CodeHaven did
not expect: the user is told what happened in one sentence, that their files were not changed, and what to try. The
full traceback is printed only with CODEHAVEN_DEBUG=1 in the terminal, and always to the terminal that runs the
browser page's server, where it is useful for a bug report.
"""
import os
import sys
import traceback

DEBUG = "CODEHAVEN_DEBUG"
ADVICE = ("Your files were not changed. Try again; if it keeps happening, run `codehaven --doctor`, "
          f"or set {DEBUG}=1 to see the details for a bug report.")


def friendly(exc):
    """The message to show for exc: its own words for an expected problem, a plain explanation for any other."""
    if isinstance(exc, (ValueError, OSError, UnicodeError)) and str(exc):
        return str(exc)
    name = type(exc).__name__
    if name == "GraphRecursionError":
        return ("The question took more steps than CodeHaven allows. Ask something narrower, such as one function "
                "or one file, or raise --max-steps.")
    if isinstance(exc, MemoryError):
        return ("This computer ran out of memory. Close other programs, or use a smaller model, and try again. "
                "Your files were not changed.")
    detail = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
    return f"CodeHaven hit a problem it did not expect ({name}{': ' + detail if detail else ''}). {ADVICE}"


def debugging():
    """Whether the user asked for full tracebacks."""
    return os.environ.get(DEBUG, "").strip() not in ("", "0")


def report(exc, file=None, always=False):
    """Print the traceback of an unexpected problem to file (stderr by default) when debugging, or always."""
    if always or debugging():
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=file or sys.stderr)
