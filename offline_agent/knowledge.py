"""Offline, in-memory BM25 retrieval over local reference documents."""
import functools
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from . import arabic, index_cache
from .model import Stopped
from .workspace import MAX_FILES, Workspace
from .dependencies import PythonDependencies
from .javascript import JS_EXTENSIONS, JavaScriptDependencies
from .languages import OtherDependencies
from .exclusions import IgnoreConfigurationError

EXTENSIONS = {".md", ".txt", ".rst"}
CODE_EXTENSIONS = EXTENSIONS | JS_EXTENSIONS | {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    ".kts", ".scala", ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html",
    ".css", ".scss", ".vue", ".svelte", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".xml", ".graphql", ".gql", ".proto", ".r", ".lua",
    ".ex", ".exs", ".erl", ".hs", ".clj", ".dart",
}
MAX_DOCUMENTS = MAX_FILES  # The same bound as Workspace.files, so a mid-sized repository is indexed in full.
# The most passages and bytes of text one index holds. Building the index for a project at these limits takes
# about a quarter of a minute on a laptop; beyond them the index is reported as limited.
MAX_PASSAGES = 40_000
MAX_INDEX_BYTES = 25_000_000
BUNDLED_TOPICS = {"stategraph", "checkpoint", "checkpoints", "checkpointer", "interrupt", "interrupts"}
CODE_NAMES = {"Dockerfile", "Makefile", "CMakeLists.txt", "Justfile", "Gemfile"}


@functools.lru_cache(maxsize=200_000)
def stem(word):
    """Reduce simple English endings, so "restocking" finds restock and "pricing" finds price."""
    if len(word) > 4 and word.isascii():
        word = re.sub(r"(?:ing|ed|es|(?<!s)s)$", "", word)
        word = word[:-1] if len(word) > 3 and word.endswith("e") else word
    elif arabic.WORD.fullmatch(word):
        word = arabic.stem(word)
    return word


def tokens(text):
    """Lowercase word stems, split at underscores and camelCase, so applyDiscount and apply_discount match "apply discount"."""
    text = arabic.normalise(text)  # Arabic short vowels are not word characters, so they would split a word in two.
    return [stem(word) for word in re.findall(r"[^\W_]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).lower(), flags=re.UNICODE)]


def chunks(text):
    """Keep line citations even when a long line needs multiple chunks."""
    buffer, start, end = [], 1, 1
    size = 0
    for number, line in enumerate(text.splitlines(), 1):
        # A line over 1,600 characters is cut into pieces; max(1, ...) keeps an empty line as one empty piece.
        for offset in range(0, max(1, len(line)), 1600):
            piece = line[offset:offset + 1600]
            # Close the chunk before it passes about 1,800 characters or 30 lines, so every passage stays small.
            if buffer and (size + len(piece) + 1 > 1800 or number - start >= 30):
                yield start, end, "\n".join(buffer)
                buffer, size = [], 0
            if not buffer:
                start = number
            buffer.append(piece)
            size += len(piece) + 1
            end = number
    if buffer:
        yield start, end, "\n".join(buffer)


class KnowledgeBase:
    """A BM25 index over the bundled notes, an optional reference folder, and the repository, with its import graph."""
    def __init__(self, directory=None, *, workspace=None, context_budget=16_000, excludes=(), semantic=None, cancel=None,
                 cache=None, previous=None):
        """Read and chunk every allowed file once. The limits below keep memory and start-up time bounded.

        cancel is an optional threading.Event: once it is set, indexing stops with Stopped between two files,
        so the page's Stop button can end a long index and keep the repository that was open before.

        cache is a file outside the repository where the keyword index is kept between runs (None keeps nothing on
        disk), and previous is the index of an earlier build in this run. From either, a file whose size, modification
        time, and text are unchanged keeps its passages and parsed outline, so only changed files are worked on again.
        """
        if not 2000 <= context_budget <= 16_000:
            raise ValueError("Retrieval budget must be between 2000 and 16000 bytes")
        self.cache = Path(cache) if cache is not None else None
        known = previous if previous is not None else index_cache.load(self.cache)
        self.records = {}  # "label/name" -> size, mtime, sha, text, and passages of each indexed file, for the next build.
        self.reused = 0  # Files whose passages came from an earlier build.
        self.context_budget = context_budget
        self.semantic = semantic  # Optional search by meaning; see semantic.py. None means key words only.
        self.source_texts = {}
        self.directory = directory
        self.workspace_root = workspace.root if isinstance(workspace, Workspace) else workspace
        self.excludes = workspace.excludes if isinstance(workspace, Workspace) else tuple(excludes)
        # Reuse a supplied workspace so its cached exclusion decisions are shared.
        repository = (workspace if isinstance(workspace, Workspace) else
                      Workspace(self.workspace_root, excludes=self.excludes) if self.workspace_root is not None else None)
        if repository is not None:
            self.workspace_root = repository.root
        self.excluded_reference_roots = 0
        self.documents = 0
        self.documents_by_source = {"builtin": 0, "user": 0, "repo": 0}
        self.skipped = 0
        self.limited = False
        self.entries = []
        self.frequencies = Counter()
        # Three sources, in this order: the bundled notes, the user's reference folder, then the repository itself.
        roots = [("builtin", Path(__file__).parent / "references", EXTENSIONS)]
        if directory is not None:
            directory = Path(directory).expanduser()
            if directory.exists():
                roots.append(("user", directory, EXTENSIONS))
            else:
                raise ValueError(f"Knowledge directory does not exist: {directory}")
        if self.workspace_root is not None:
            roots.append(("repo", self.workspace_root, CODE_EXTENSIONS))
        total_bytes = 0
        seen = set()
        listings = []  # What was listed from the reference folder and the repository, for the stamp below.
        code_files, manifests = {}, {}
        for label, root, extensions in roots:
            root = Path(root).resolve(strict=True)
            # A reference folder inside the repository must obey the repository's own hide and exclude rules.
            if label == "user" and repository is not None and root.is_relative_to(repository.root):
                relative = root.relative_to(repository.root)
                if relative.parts:
                    try:
                        repository.path(relative.as_posix())
                    except IgnoreConfigurationError:
                        raise
                    except ValueError:
                        self.excluded_reference_roots += 1
                        continue
            workspace = repository if label == "repo" else Workspace(root, excludes=self.excludes)
            names = workspace.files()
            self.limited |= len(names) >= MAX_FILES
            if label != "builtin":
                listings.append((label, workspace, names))
            for name in names:
                if cancel is not None and cancel.is_set():
                    raise Stopped()
                if label == "repo" and Path(name).name == "go.mod":
                    try:
                        manifests[name] = workspace.read(name)  # Names the Go module, so its imports can be followed.
                    except (ValueError, OSError, UnicodeError):
                        pass
                if (Path(name).suffix.lower() not in extensions
                        and not (label == "repo" and Path(name).name in CODE_NAMES)):
                    continue
                # The resolved path identifies a file, so one reached from two roots, or through a link, is indexed once.
                identity = workspace.path(name).resolve()
                if label == "user" and repository is not None and identity.is_relative_to(repository.root):
                    try:
                        repository.path(identity.relative_to(repository.root).as_posix())
                    except IgnoreConfigurationError:
                        raise
                    except ValueError:
                        continue
                if identity in seen:
                    continue
                # At most MAX_PASSAGES passages and MAX_INDEX_BYTES of text are indexed; index_limited tells the caller when a limit cut in.
                if self.documents >= MAX_DOCUMENTS or len(self.entries) >= MAX_PASSAGES:
                    self.limited = True
                    break
                key = f"{label}/{name}"
                old = known["files"].get(key)
                try:
                    status = os.stat(identity)
                    # An unchanged file from this run's earlier build is not even read again: listing it already
                    # checked it, and its size and modification time are what the refresh stamp trusts too.
                    if old and "text" in old and (old["size"], old["mtime"]) == (status.st_size, status.st_mtime_ns):
                        text = old["text"]
                    else:
                        text = workspace.read(name)
                except (ValueError, OSError, UnicodeError):
                    self.skipped += 1
                    continue
                digest = old["sha"] if old and old.get("text") is text else hashlib.sha256(text.encode()).hexdigest()
                if old and old["sha"] != digest:
                    old = None
                size = len(text.encode())
                if total_bytes + size > MAX_INDEX_BYTES:
                    self.limited = True
                    continue
                total_bytes += size
                self.documents += 1
                self.documents_by_source[label] += 1
                seen.add(identity)
                if label == "repo":
                    self.source_texts["repo/" + name] = text
                    code_files[name] = text
                self.reused += old is not None
                passages = self._passages(key, text, old)
                self.records[key] = {"size": status.st_size, "mtime": status.st_mtime_ns, "sha": digest, "text": text,
                                     "passages": passages}
                for entry in passages:
                    if len(self.entries) >= MAX_PASSAGES:
                        self.limited = True
                        break
                    self.entries.append(entry)
                    self.frequencies.update(entry["counts"].keys())
        self.stamp = self.stamp_of(listings)
        self.average_length = (sum(e["length"] for e in self.entries) / len(self.entries)
                               if self.entries else 1)
        # A file's outline and imports depend on its own text and on which files exist, so with the same list of
        # files an unchanged file keeps what the last build found for it.
        names = hashlib.sha256("\0".join(sorted(code_files)).encode()).hexdigest()
        same = known.get("names") == names
        shas = {name: self.records["repo/" + name]["sha"] for name in code_files}
        reuse = {kind: {name: tuple(row[1:]) for name, row in known.get(kind, {}).items()
                        if same and name in shas and len(row) == 5 and row[0] == shas[name]} for kind in ("python", "js")}
        self.dependencies = PythonDependencies({name: text for name, text in code_files.items()
                                                if Path(name).suffix in {".py", ".pyi"}}, reuse["python"])
        # One import graph for all languages: the other analysers' results are merged into the Python one.
        js = JavaScriptDependencies(code_files, reuse["js"])
        self.analysed = {"names": names}
        for kind, found in (("python", self.dependencies), ("js", js)):
            self.analysed[kind] = {name: [shas[name], found.definitions[name], found.spans[name], found.edges[name],
                                          found.missing[name]] for name in found.edges}
        self.dependencies.edges.update(js.edges)
        self.dependencies.definitions.update(js.definitions)
        self.dependencies.spans.update(js.spans)
        self.dependencies.missing.update(js.missing)
        self.dependencies.parse_failures += js.parse_failures
        other = OtherDependencies(code_files, manifests)  # Go, Java, and Rust.
        self.dependencies.edges.update(other.edges)
        self.dependencies.definitions.update(other.definitions)
        self.dependencies.spans.update(other.spans)
        self.passages_by_source = {}
        for entry in self.entries:
            self.passages_by_source.setdefault(entry["source"], []).append(entry)
        # Where each function, class, method, and module constant is defined, so a name in the question can be looked up.
        self.symbols = {}
        for name in sorted(self.dependencies.definitions.keys() | self.dependencies.spans.keys()):
            places = [(symbol, line) for symbol, line in self.dependencies.definitions.get(name, {}).items()]
            places += [(span["name"], span["start"]) for span in self.dependencies.spans.get(name, [])]
            for symbol, line in places:
                found = self.symbols.setdefault(symbol, [])
                if (name, line) not in found:
                    found.append((name, line))
        if self.semantic is not None:
            self.semantic.start(self.entries)  # Embeds new or changed passages in the background.
        # Written only when something was worked on afresh, so an unchanged repository costs no disk write.
        if self.cache is not None and (self.reused < self.documents or known.get("names") != names):
            index_cache.save(self.cache, self.records, self.analysed)

    @staticmethod
    def _passages(source, text, old):
        """The index entries of one file's passages: reused from old, an earlier build of the same text, or made afresh."""
        if old and "passages" in old:
            return old["passages"]
        pieces = list(chunks(text))
        made = []
        # Saved on disk: the number and term counts of each passage, whose text is cut from the file again.
        rows = old["chunks"] if old and "chunks" in old else (
            (index, Counter(tokens(piece[2]))) for index, piece in enumerate(pieces))
        for index, counts in rows:
            # BM25 needs each passage's term counts and length, and for each term how many passages hold it (frequencies).
            if not counts or not 0 <= index < len(pieces):
                continue
            start, end, passage = pieces[index]
            made.append({"source": source, "start_line": start, "end_line": end, "text": passage, "chunk": index,
                         "counts": counts if isinstance(counts, Counter) else Counter(counts),
                         "length": sum(counts.values())})
        return made

    @staticmethod
    def stamp_of(listings):
        """A fingerprint of the listed files: their names, sizes, and modification times. Nothing is read."""
        digest = hashlib.sha256()
        for label, workspace, names in listings:
            for name in names:
                try:
                    status = (workspace.root / name).stat()
                    row = (label, name, status.st_size, status.st_mtime_ns)
                except OSError:
                    row = (label, name, None, None)
                digest.update(repr(row).encode())
        return digest.hexdigest()

    def current_stamp(self):
        """The fingerprint of what is on disk now, listed afresh so changed ignore rules count too."""
        listings = []
        directory = Path(self.directory).expanduser() if self.directory is not None else None
        for label, root in (("user", directory), ("repo", self.workspace_root)):
            if root is not None and Path(root).exists():
                workspace = Workspace(root, excludes=self.excludes)
                listings.append((label, workspace, workspace.files()))
        return self.stamp_of(listings)

    def refresh(self):
        """Pick up files that changed on disk since the index was built. Returns whether it was rebuilt.

        Listing the files is far cheaper than reading and indexing them again, so an unchanged project, the
        usual case between two questions, keeps its index.
        """
        try:
            if self.current_stamp() == self.stamp:
                return False
        except (ValueError, OSError):
            pass  # Whatever is wrong is reported by the rebuild below.
        previous = {"files": self.records, **self.analysed}
        self.__init__(self.directory, workspace=self.workspace_root, context_budget=self.context_budget,
                      excludes=self.excludes, semantic=self.semantic, cache=self.cache, previous=previous)
        return True

    def forget(self):
        """Delete the keyword index kept on disk, if there is one."""
        index_cache.forget(self.cache)

    def _expanded(self, entry, terms, definition_line=None):
        """Prefer complete syntactic symbols over arbitrary line chunks."""
        source = entry["source"]
        spans = self.dependencies.spans.get(source.removeprefix("repo/"), [])
        lines = self.source_texts.get(source, "").splitlines()
        choices = []
        for span in spans:
            if definition_line is not None:
                relevant = span["start"] <= definition_line <= span["end"]
            else:
                relevant = span["start"] <= entry["end_line"] and span["end"] >= entry["start_line"]
            if not relevant:
                continue
            text = "\n".join(lines[span["start"] - 1:span["end"]])
            overlap = len(terms & set(tokens(text)))
            if definition_line is not None:
                # The innermost symbol around a definition is the one defined there: a method, not its whole class.
                overlap = 0
            if definition_line is not None or overlap:
                choices.append((overlap, -(span["end"] - span["start"]), span, text))
        if not choices:
            return None
        # Most query terms wins; on a tie the shortest span, which is the innermost symbol.
        _, _, span, text = max(choices, key=lambda row: row[:2])
        expanded = dict(entry, start_line=span["start"], end_line=span["end"], text=text,
                        symbol=span["name"], symbol_complete=True)
        constants = self._constants(lines, text)
        if constants:
            expanded["constants"] = constants
        return expanded

    @staticmethod
    def _constants(lines, text):
        """Values of the file's module constants a symbol uses, since the answer is often the value, not the name."""
        found = []
        for name in dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)):
            # An assignment at the start of a line (Python, or JavaScript const/let/var/export), not an == comparison.
            definition = re.compile(rf"(?:export\s+)?(?:const\s+|let\s+|var\s+)?{name}\s*(?::[^=]+)?=(?!=)")
            number = next((number for number, line in enumerate(lines, 1) if definition.match(line) and len(line) <= 160), None)
            if number and lines[number - 1] not in text:
                found.append({"name": name, "line": number, "text": lines[number - 1].strip()})
            if len(found) >= 3:
                break
        return found

    def named_definitions(self, query):
        """Definitions of the code names a question mentions, as (name, file, line), in the order they appear.

        Only names written like code count: with an underscore, camelCase, ALL_CAPS, a dot, backticks, or (),
        or a plain word right next to "function", "method", "class", or "constant". Keyword ranking often
        misses these, because a name's definition can use it once while tests and callers repeat it."""
        mentioned = []
        for match in re.finditer(r"`([^`\n]{1,100})`|([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(\s*\(\))?", query):
            text = match.group(1) or match.group(2)
            for name in re.findall(r"[A-Za-z_$][\w$]*", text)[-1:]:  # Store.remove_item -> remove_item
                coded = (match.group(1) or match.group(3) or "." in text or "_" in name.strip("_")
                         or re.search(r"[a-z][A-Z]", name) or re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", name))
                if not coded:
                    near = re.compile(rf"\b(?:function|method|class|constant)\s+{re.escape(name)}\b"
                                      rf"|\b{re.escape(name)}\s+(?:function|method|class|constant)\b", re.I)
                    coded = near.search(query)
                if coded and name not in mentioned and not name.startswith("__"):
                    mentioned.append(name)
        found = []
        for name in mentioned:
            places = self.symbols.get(name, [])
            if 1 <= len(places) <= 3:  # A name defined in many files says little about which one is meant.
                found += [(name, path, line) for path, line in places]
        return found[:3]

    def retrieve(self, query, budget=None):
        """The passages to show the model for a query, as a JSON string that fits the context budget.

        budget, when given, is a smaller limit for this call only (at least 2000 bytes), such as the room left in a
        prompt that preloaded files already fill.

        In order: definitions of names the question mentions, the best-ranked matches (at most 4), passages of the
        files those matches import (2 levels deep), then further matches if space is left. Ranking is BM25 over
        key words, mixed with the meaning-search ranking when one is available."""
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("Retrieval query must contain 1 to 1000 characters")
        budget = self.context_budget if budget is None else max(2000, min(budget, self.context_budget))
        terms = set(tokens(query))
        scored = []
        count = len(self.entries)
        for entry in self.entries:
            score = 0.0
            for term in terms:
                frequency = entry["counts"].get(term, 0)
                if not frequency:
                    continue
                # BM25 with the usual k1 = 1.2 and b = 0.75: a rare term weighs more (inverse), repeats of a term add less
                # and less, and a passage longer than average is scaled down. 2.2 is k1 + 1.
                df = self.frequencies[term]
                inverse = math.log(1 + (count - df + 0.5) / (df + 0.5))
                norm = frequency + 1.2 * (0.25 + 0.75 * entry["length"] / self.average_length)
                score += inverse * frequency * 2.2 / norm
            if score > 0:
                scored.append((score, entry))
        near = self.semantic.rank(query, self.entries) if self.semantic is not None else []
        if near:
            # Reciprocal rank fusion: a passage's place in each ranking counts, not the two scores, which are
            # on different scales. A passage near the top of both lists beats one that leads only one of them.
            scored.sort(key=lambda pair: (-pair[0], pair[1]["source"], pair[1]["start_line"]))
            fused = {}
            # 60 is the customary fusion constant: it evens out the gap between neighbouring places. Only the top 64
            # keyword matches take part.
            for place, (_, entry) in enumerate(scored[:64]):
                fused[id(entry)] = [1 / (60 + place), entry]
            for place, (_, index) in enumerate(near):
                entry = self.entries[index]
                fused.setdefault(id(entry), [0.0, entry])[0] += 1 / (60 + place)
            scored = [(score, entry) for score, entry in fused.values()]
        # The bundled notes only cover LangGraph, so they must not crowd out the project on other questions.
        if "langgraph" not in terms and not terms & BUNDLED_TOPICS and any(not e["source"].startswith("builtin/") for _, e in scored):
            scored = [pair for pair in scored if not pair[1]["source"].startswith("builtin/")]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["source"], pair[1]["start_line"]))
        # Give different files a chance before adding further passages from one file.
        diverse, repeated, seen_sources = [], [], set()
        for pair in scored:
            source = pair[1]["source"]
            (repeated if source in seen_sources else diverse).append(pair)
            seen_sources.add(source)
        candidates = (diverse + repeated)[:64]
        result = {"results": [], "documents": self.documents, "chunks": count,
                  "documents_by_source": self.documents_by_source,
                  "skipped_documents": self.skipped, "index_limited": self.limited,
                  "excluded_reference_roots": self.excluded_reference_roots,
                  "meaning_search": bool(near),
                  "results_limited": False, "dependency_limit_reached": False,
                  "dependency_parse_failures": self.dependencies.parse_failures,
                  "context_budget_bytes": budget, "context_bytes": 0,
                  "budget_limited": False}
        identities, selected = set(), []

        def accept(entry, metadata, limit, definition_line=None):
            """Add one passage to the result if it fits within limit bytes. Returns "added", "duplicate", or "budget"."""
            expanded = self._expanded(entry, terms, definition_line)
            # Try the whole symbol first; if it is too big for the budget, fall back to the plain chunk.
            variants = [expanded, dict(entry, symbol_complete=False)] if expanded else [entry]
            for variant in variants:
                if variant is None:
                    continue
                identity = (variant["source"], variant["start_line"], variant["end_line"], variant["text"])
                if identity in identities:
                    return "duplicate"
                passage = {key: variant[key] for key in ("source", "start_line", "end_line", "text")}
                passage.update({key: variant[key] for key in ("symbol", "symbol_complete", "constants") if key in variant})
                passage.update(metadata)
                result["results"].append(passage)
                # Measured as real JSON with the passage in place; 128 bytes stay free for the counters filled in at the end.
                if len(json.dumps(result, ensure_ascii=False).encode()) <= limit - 128:
                    identities.add(identity)
                    if metadata["kind"] == "match":
                        selected.append(variant)
                    return "added"
                result["results"].pop()
            result["budget_limited"] = True
            return "budget"

        # When the top matches import other files, 40% of the budget is held back for those dependencies.
        has_dependencies = any(self.dependencies.edges.get(e["source"].removeprefix("repo/"))
                               for _, e in candidates[:4] if e["source"].startswith("repo/"))
        primary_budget = int(budget * 0.6) if has_dependencies else budget
        # A name the question mentions comes first, ahead of keyword matches, so its definition is never crowded out.
        for name, path, line in self.named_definitions(query):
            passage = next((p for p in self.passages_by_source.get("repo/" + path, [])
                            if p["start_line"] <= line <= p["end_line"]), None)
            if passage is not None:
                accept(passage, {"score": 0.0, "kind": "definition", "via": "question", "name": name},
                       primary_budget, line)
        # At most 4 keyword matches at first; the others wait in deferred until the dependencies have had their share.
        deferred = []
        for score, entry in candidates:
            if len(selected) >= 4:
                deferred.append((score, entry))
                continue
            status = accept(entry, {"score": round(score, 4), "kind": "match"}, primary_budget)
            if status == "budget":
                deferred.append((score, entry))
        # Breadth-first walk of the import graph from the matched files: 2 levels deep, at most 4 passages. [5:] drops "repo/".
        queue = [(entry["source"][5:], 0) for entry in selected if entry["source"].startswith("repo/")]
        visited = {name for name, _ in queue}
        added = 0
        for name, depth in queue:
            edges = self.dependencies.edges.get(name, {})
            if depth >= 2:
                result["dependency_limit_reached"] |= any(target not in visited for target in edges)
                continue
            for target, symbols in edges.items():
                passages = self.passages_by_source.get("repo/" + target, [])
                if not passages:
                    continue
                definitions = self.dependencies.definitions.get(target, {})
                # Show the passage holding each imported name's definition; when no name is known, the file's first passage.
                definition_lines = list(dict.fromkeys(definitions[s] for s in symbols if s in definitions)) or [None]
                for line in definition_lines:
                    passage = next((p for p in passages if line is not None
                                    and p["start_line"] <= line <= p["end_line"]), passages[0])
                    if added >= 4:
                        result["dependency_limit_reached"] = True
                        break
                    status = accept(passage, {"score": 0.0, "kind": "dependency",
                                              "via": "repo/" + name, "depth": depth + 1},
                                    budget, line)
                    if status == "added":
                        added += 1
                    elif status == "budget":
                        result["dependency_limit_reached"] = True
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
        # Reclaim unused dependency space, while avoiding unexplored new import graphs.
        for score, entry in deferred:
            if len(selected) >= 4:
                break
            name = entry["source"].removeprefix("repo/")
            if self.dependencies.edges.get(name) and name not in visited:
                continue
            accept(entry, {"score": round(score, 4), "kind": "match"}, budget)
        result["results_limited"] = len(scored) > len(selected) or result["budget_limited"]
        for _ in range(3):  # The count is part of the JSON it measures, so repeat until its own digits settle.
            result["context_bytes"] = len(json.dumps(result, ensure_ascii=False).encode())
        return json.dumps(result, ensure_ascii=False)
