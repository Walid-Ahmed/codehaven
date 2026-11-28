"""Answers saved as Markdown, for a pull request, an issue, or the project's docs.

The model never writes files: CodeHaven's own code writes the questions and the answers as they were shown, with
their path:line citations and any failed answer checks, only when the user asks, to the file the user names.
"""
import time
from pathlib import Path


def markdown(turns, about, model=""):
    """turns: [(question, answer, issues)]. A heading per question, the answer under it, and failed checks as a quote."""
    parts = [f"# CodeHaven: {about}", time.strftime("%Y-%m-%d %H:%M") + (f" · {model}" if model else "")]
    for question, answer, issues in turns:
        parts += [f"## {' '.join(question.split())}", answer or "(no final answer)"]
        if issues:
            parts.append("> **Unverified**\n" + "\n".join(f"> - {issue}" for issue in issues))
    return "\n\n".join(parts) + "\n"


def write(path, text):
    """Write text to path, refusing to replace a file that is there, and return the path written."""
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / time.strftime("codehaven-%Y%m%d-%H%M%S.md")
    with open(target, "x", encoding="utf-8") as handle:  # "x": an existing file is never overwritten.
        handle.write(text)
    return target
