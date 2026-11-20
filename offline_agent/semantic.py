"""Search by meaning: passages are compared with the question as vectors from a local embedding model.

Keyword search finds "late fee" only where those words appear. An embedding model turns a passage into a list
of numbers that sits near passages with a similar meaning, so "where do we charge people who cancel late?" can
find a function named cancellation_penalty. CodeHaven mixes both rankings, so exact names still win.

The embedding model is a second, small Ollama model on this computer, reached over loopback like the chat model:
nothing leaves the machine. It is optional. Without one installed, or with --no-semantic, search works from key
words as before. Any failure here switches meaning search off for the session rather than failing an answer.

Vectors are kept in ~/CodeHaven/index, one file per repository, so only changed passages are embedded again.
The file holds numbers and hashes, never code text, is readable by this user only, and is not written with
--no-remember. Nothing is ever written inside the repository.
"""
import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from array import array
from operator import mul
from pathlib import Path

INDEX_FOLDER = Path.home() / "CodeHaven" / "index"
# Tried in this order when --embed-model is not given; any other installed embedding model comes after them.
PREFERRED = ("nomic-embed-text", "embeddinggemma", "mxbai-embed-large", "bge-m3", "snowflake-arctic-embed", "all-minilm")
# Some models are trained with a short label before each text, and rank noticeably worse without it.
PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: "),
            "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
            "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: ")}
BATCH = 16
PASSAGE_CHARS = 2000  # Passages are at most about 1,800 characters; this only guards against a surprise.
CANDIDATES = 24  # How many of the nearest passages take part in ranking.
FLOOR = 0.2  # Below this similarity a passage is unrelated, whatever its rank.
SAVE_EVERY = 480
FOLLOW_UP_WAIT = 3  # Seconds a question waits for passages changed since the last one.

# math.sumprod exists from Python 3.12; older versions use the slower plain sum.
dot = getattr(math, "sumprod", None) or (lambda a, b: sum(map(mul, a, b)))


def base_name(model):
    """A model's plain name, without registry path or tag: "library/nomic-embed-text:latest" gives "nomic-embed-text"."""
    return model.split(":", 1)[0].rsplit("/", 1)[-1].lower()


def choose_model(client, wanted=None):
    """The embedding model to use: the one asked for, or the first preferred one installed. None when there is none.

    Cloud-backed models are never used. A model asked for by name that is missing or cannot embed is an error.
    """
    from .model import cloud_name
    if not hasattr(client, "request"):
        return None
    try:
        listed = client.request("/api/tags").get("models", [])
    except ValueError:
        if wanted:
            raise
        return None  # Meaning search is an extra: a server that cannot list models just leaves it off.
    installed = []
    for item in listed if isinstance(listed, list) else []:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or cloud_name(name) or item.get("remote_host") or item.get("remote_model"):
            continue
        capabilities = item.get("capabilities")
        # A server that does not list capabilities leaves only the model's name to judge by.
        embeds = "embedding" in capabilities if isinstance(capabilities, list) else bool(re.search(r"embed|minilm|bge", name))
        installed.append((name, embeds))
    if wanted:
        match = next((pair for pair in installed if pair[0] in (wanted, wanted + ":latest")), None)
        if match is None:
            raise ValueError(f"Embedding model {wanted!r} is not installed. Run: ollama pull {wanted} while online.")
        if not match[1]:
            raise ValueError(f"{wanted!r} is a chat model, not an embedding model; try nomic-embed-text.")
        return match[0]
    names = [name for name, embeds in installed if embeds]
    rank = lambda name: next((i for i, known in enumerate(PREFERRED) if base_name(name) == known), len(PREFERRED))
    return min(names, key=lambda name: (rank(name), name), default=None)


def normalised(vector):
    """The vector scaled to length 1, as compact 32-bit floats, so the dot product of two is their cosine similarity."""
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return array("f", (value / length for value in vector))


class Semantic:
    """Vectors for one repository's passages, built in the background and ranked against a question."""

    def __init__(self, client, model, *, cache=None, output=None):
        """Set up for one embedding model and load the vectors saved at cache (a file path; None keeps nothing on disk)."""
        self.client, self.model = client, model
        self.cache = Path(cache) if cache else None
        self.output = output or (lambda *_: None)
        self.document_prefix, self.query_prefix = PREFIXES.get(base_name(model), ("", ""))
        # vectors: passage key -> vector, for every passage embedded so far. keys: the current passages' keys, in index order.
        self.vectors, self.keys = {}, []
        self.failed = None
        # generation counts index rebuilds, so an outdated background build can tell it was replaced. done is set while
        # no build is running.
        self.generation, self.thread, self.done = 0, None, threading.Event()
        self.done.set()
        self.lock = threading.Lock()
        if hasattr(client, "companions"):
            client.companions.add(model)  # Not "another model competing for memory": it is ours.
        self.load()

    @staticmethod
    def cache_path(workspace, model, folder=None):
        """The vectors file for one repository and model: a readable folder name plus a hash of the full path and model."""
        name = re.sub(r"[^\w.-]", "_", workspace.root.name, flags=re.ASCII)[:40] or "repository"
        mark = hashlib.sha256(f"{workspace.root}\0{model}".encode()).hexdigest()[:12]
        return (Path(folder).expanduser() if folder else INDEX_FOLDER) / f"{name}-{mark}.vectors"

    def key(self, text):
        """A passage's identity: a hash of the model and the text, so changed text or another model is embedded afresh."""
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()[:24]

    def embed(self, texts):
        """Unit-length vectors for the texts, in order, from the local embedding model. ValueError for an unusable reply."""
        result = self.client.request("/api/embed", {"model": self.model, "input": texts, "truncate": True, "keep_alive": "30m"})
        vectors = result.get("embeddings")
        if (not isinstance(vectors, list) or len(vectors) != len(texts)
                or not all(isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) for v in vectors)):
            raise ValueError("The embedding model returned no usable vectors")
        return [normalised(vector) for vector in vectors]

    def start(self, entries):
        """Give every passage its key and embed, in the background, the ones not known yet."""
        with self.lock:
            self.generation += 1
            generation = self.generation
            for entry in entries:
                entry["vector_key"] = self.key(entry["text"])
            self.keys = [entry["vector_key"] for entry in entries]
            # Keyed by hash, so identical passages are embedded once.
            missing = {}
            for entry in entries:
                if entry["vector_key"] not in self.vectors:
                    missing.setdefault(entry["vector_key"], entry["text"])
            self.done.clear()
        if not missing or self.failed:
            self.done.set()
            return
        self.thread = threading.Thread(target=self.build, args=(generation, list(missing.items())), daemon=True)
        self.thread.start()

    def build(self, generation, missing):
        """Background thread: embed the missing passages in batches, saving now and then so a stopped run keeps its work."""
        added = 0
        try:
            for offset in range(0, len(missing), BATCH):
                if generation != self.generation:
                    return  # The files changed again; a newer build has taken over.
                batch = missing[offset:offset + BATCH]
                vectors = self.embed([self.document_prefix + text[:PASSAGE_CHARS] for _, text in batch])
                for (key, _), vector in zip(batch, vectors):
                    self.vectors[key] = vector
                added += len(batch)
                if added % SAVE_EVERY == 0:
                    self.save()
            self.save()
        except (ValueError, OSError) as exc:
            self.failed = str(exc)
            self.output(f"Meaning search is off for this session ({exc}); answers use key-word search.")
        finally:
            if generation == self.generation:
                self.done.set()

    def wait(self, timeout=None, progress=None):
        """Wait for the background build. With progress, report every few seconds. Returns True once it is complete."""
        if progress is None:
            return self.done.wait(timeout)
        waited = 0
        while not self.done.wait(5):
            waited += 5
            ready, total = self.coverage()
            progress(f"Preparing meaning search: {ready} of {total} passages ready ({waited}s).")
            if timeout is not None and waited >= timeout:
                return False
        return True

    def coverage(self):
        """(passages that have a vector, all passages) for the current index; used in progress messages."""
        keys = self.keys
        return sum(1 for key in keys if key in self.vectors), len(keys)

    def rank(self, query, entries):
        """Indexes of the passages nearest in meaning to the query, nearest first, as (similarity, index)."""
        if self.failed or not self.vectors:
            return []
        self.done.wait(FOLLOW_UP_WAIT)
        try:
            wanted = self.embed([self.query_prefix + query[:PASSAGE_CHARS]])[0]
        except (ValueError, OSError) as exc:
            self.failed = str(exc)
            self.output(f"Meaning search is off for this session ({exc}); answers use key-word search.")
            return []
        scored = []
        for index, entry in enumerate(entries):
            vector = self.vectors.get(entry.get("vector_key"))
            if vector is not None and len(vector) == len(wanted):
                # Both vectors have length 1, so their dot product is the cosine similarity.
                similarity = dot(wanted, vector)
                if similarity >= FLOOR:
                    scored.append((similarity, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[:CANDIDATES]

    def load(self):
        """Read saved vectors. A missing, damaged, or mismatched file just means everything is embedded again."""
        if self.cache is None:
            return
        try:
            with open(self.cache, "rb") as handle:
                # Layout: one JSON header line (model, dimensions, keys), then every vector as raw 32-bit floats, in key order.
                header = json.loads(handle.readline(4_000_000))
                raw = handle.read()
            keys, size = header["keys"], header["dimensions"]
            if header.get("model") != self.model or not isinstance(size, int) or not 1 <= size <= 8192:
                return
            values = array("f")
            values.frombytes(raw)
            if len(values) != len(keys) * size:
                return
            self.vectors = {key: values[index * size:(index + 1) * size] for index, key in enumerate(keys) if isinstance(key, str)}
        except (OSError, ValueError, KeyError, TypeError):
            self.vectors = {}

    def save(self):
        """Keep the vectors of the current passages, for this user only, replacing the file in one step."""
        if self.cache is None:
            return
        keys = [key for key in dict.fromkeys(self.keys) if key in self.vectors]
        if not keys:
            return
        # The flat layout needs every vector to have the same length.
        size = len(self.vectors[keys[0]])
        keys = [key for key in keys if len(self.vectors[key]) == size]
        try:
            self.cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle, partial = tempfile.mkstemp(prefix=".vectors-", dir=self.cache.parent)  # Readable by this user only.
            try:
                with os.fdopen(handle, "wb") as file:
                    file.write(json.dumps({"model": self.model, "dimensions": size, "keys": keys}).encode() + b"\n")
                    for key in keys:
                        file.write(self.vectors[key].tobytes())
                os.replace(partial, self.cache)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(partial)
                raise
        except OSError:
            pass  # A cache that cannot be written only costs time on the next start.

    def forget(self):
        """Delete the saved vectors file, if there is one."""
        with contextlib.suppress(OSError):
            if self.cache is not None:
                self.cache.unlink()
