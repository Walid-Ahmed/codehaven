"""The keyword index kept between runs, so a large repository opens in a fraction of the time.

Indexing reads every file, cuts it into passages, counts the words in each, and parses its outline and imports.
For a repository of a few thousand files that takes several seconds, and almost all of it is the same as last time.
This file keeps, for each indexed file, its size, modification time, a hash of its text, the word counts of its
passages, and its parsed outline (names, line numbers, and imports). It never holds a passage's text: the file is
read again and its hash compared, so a changed file is always indexed afresh.

The file is kept in ~/CodeHaven/index beside the meaning-search vectors, readable by this user only, and is not
written with --no-remember. Nothing is ever written inside the repository. A missing, damaged, or older file only
means the index is built from scratch.
"""
import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

VERSION = 2
FOLDER = Path.home() / "CodeHaven" / "index"


def empty():
    """What load returns when nothing usable is saved: no files, so everything is indexed afresh."""
    return {"files": {}}


def path(workspace, folder=None):
    """The index file for one repository: a readable name plus a hash of its full path."""
    name = re.sub(r"[^\w.-]", "_", workspace.root.name, flags=re.ASCII)[:40] or "repository"
    mark = hashlib.sha256(str(workspace.root).encode()).hexdigest()[:12]
    return (Path(folder).expanduser() if folder else FOLDER) / f"{name}-{mark}.keywords"


def load(cache):
    """The saved index at cache, shaped like KnowledgeBase's own records, or an empty one."""
    if cache is None:
        return empty()
    try:
        with open(cache, encoding="utf-8") as handle:
            saved = json.load(handle)
        if not isinstance(saved, dict) or saved.get("version") != VERSION or not isinstance(saved.get("files"), dict):
            return empty()
        files = {key: row for key, row in saved["files"].items()
                 if isinstance(row, dict) and {"size", "mtime", "sha", "chunks"} <= row.keys()}
        return {"files": files, "names": saved.get("names"),
                "python": saved.get("python") or {}, "js": saved.get("js") or {}}
    except (OSError, ValueError, TypeError, AttributeError):
        return empty()


def save(cache, records, analysed):
    """Keep the index, without any passage text, for this user only, replacing the file in one step."""
    files = {key: {"size": row["size"], "mtime": row["mtime"], "sha": row["sha"],
                   "chunks": [[entry["chunk"], dict(entry["counts"])] for entry in row["passages"]]}
             for key, row in records.items()}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, partial = tempfile.mkstemp(prefix=".keywords-", dir=cache.parent)  # Readable by this user only.
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump({"version": VERSION, "files": files, **analysed}, file, separators=(",", ":"))
            os.replace(partial, cache)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(partial)
            raise
    except OSError:
        pass  # An index that cannot be written only costs time on the next start.


def forget(cache):
    """Delete the saved index, if there is one."""
    with contextlib.suppress(OSError):
        if cache is not None:
            cache.unlink()
