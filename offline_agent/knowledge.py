"""Offline, in-memory BM25 retrieval over local reference documents."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .workspace import Workspace
from .dependencies import PythonDependencies
from .javascript import JS_EXTENSIONS, JavaScriptDependencies
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
MAX_DOCUMENTS = 1000  # The same bound as Workspace.files, so a mid-sized repository is indexed in full.
BUNDLED_TOPICS = {"stategraph", "checkpoint", "checkpoints", "checkpointer", "interrupt", "interrupts"}
CODE_NAMES = {"Dockerfile", "Makefile", "CMakeLists.txt", "Justfile", "Gemfile"}


def stem(word):
    """Reduce simple English endings, so "restocking" finds restock and "pricing" finds price."""
    if len(word) > 4 and word.isascii():
        word = re.sub(r"(?:ing|ed|es|(?<!s)s)$", "", word)
        word = word[:-1] if len(word) > 3 and word.endswith("e") else word
    return word


def tokens(text):
    """Lowercase word stems, split at underscores and camelCase, so applyDiscount and apply_discount match "apply discount"."""
    return [stem(word) for word in re.findall(r"[^\W_]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).lower(), flags=re.UNICODE)]


def chunks(text):
    """Keep line citations even when a long line needs multiple chunks."""
    buffer, start, end = [], 1, 1
    size = 0
    for number, line in enumerate(text.splitlines(), 1):
        for offset in range(0, max(1, len(line)), 1600):
            piece = line[offset:offset + 1600]
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
    def __init__(self, directory=None, *, workspace=None, context_budget=16_000, excludes=()):
        if not 2000 <= context_budget <= 16_000:
            raise ValueError("Retrieval budget must be between 2000 and 16000 bytes")
        self.context_budget = context_budget
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
        code_files = {}
        for label, root, extensions in roots:
            root = Path(root).resolve(strict=True)
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
            self.limited |= len(names) >= 1000
            for name in names:
                if (Path(name).suffix.lower() not in extensions
                        and not (label == "repo" and Path(name).name in CODE_NAMES)):
                    continue
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
                if self.documents >= MAX_DOCUMENTS or len(self.entries) >= 8000:
                    self.limited = True
                    break
                try:
                    text = workspace.read(name)
                except (ValueError, OSError, UnicodeError):
                    self.skipped += 1
                    continue
                size = len(text.encode())
                if total_bytes + size > 5_000_000:
                    self.limited = True
                    continue
                total_bytes += size
                self.documents += 1
                self.documents_by_source[label] += 1
                seen.add(identity)
                if label == "repo":
                    self.source_texts["repo/" + name] = text
                    code_files[name] = text
                for start, end, passage in chunks(text):
                    counts = Counter(tokens(passage))
                    if not counts:
                        continue
                    if len(self.entries) >= 8000:
                        self.limited = True
                        break
                    self.entries.append({"source": f"{label}/{name}", "start_line": start,
                                         "end_line": end, "text": passage,
                                         "counts": counts, "length": sum(counts.values())})
                    self.frequencies.update(counts.keys())
        self.average_length = (sum(e["length"] for e in self.entries) / len(self.entries)
                               if self.entries else 1)
        self.dependencies = PythonDependencies({name: text for name, text in code_files.items()
                                                if Path(name).suffix in {".py", ".pyi"}})
        js = JavaScriptDependencies(code_files)
        self.dependencies.edges.update(js.edges)
        self.dependencies.definitions.update(js.definitions)
        self.dependencies.spans.update(js.spans)
        self.dependencies.parse_failures += js.parse_failures
        self.passages_by_source = {}
        for entry in self.entries:
            self.passages_by_source.setdefault(entry["source"], []).append(entry)

    def refresh(self):
        """Rebuild the bounded local snapshot after an agent-approved write."""
        self.__init__(self.directory, workspace=self.workspace_root, context_budget=self.context_budget,
                      excludes=self.excludes)

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
            if definition_line is not None or overlap:
                choices.append((overlap, -(span["end"] - span["start"]), span, text))
        if not choices:
            return None
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
            definition = re.compile(rf"(?:export\s+)?(?:const\s+|let\s+|var\s+)?{name}\s*(?::[^=]+)?=(?!=)")
            number = next((number for number, line in enumerate(lines, 1) if definition.match(line) and len(line) <= 160), None)
            if number and lines[number - 1] not in text:
                found.append({"name": name, "line": number, "text": lines[number - 1].strip()})
            if len(found) >= 3:
                break
        return found

    def retrieve(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("Retrieval query must contain 1 to 1000 characters")
        terms = set(tokens(query))
        scored = []
        count = len(self.entries)
        for entry in self.entries:
            score = 0.0
            for term in terms:
                frequency = entry["counts"].get(term, 0)
                if not frequency:
                    continue
                df = self.frequencies[term]
                inverse = math.log(1 + (count - df + 0.5) / (df + 0.5))
                norm = frequency + 1.2 * (0.25 + 0.75 * entry["length"] / self.average_length)
                score += inverse * frequency * 2.2 / norm
            if score > 0:
                scored.append((score, entry))
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
                  "results_limited": False, "dependency_limit_reached": False,
                  "dependency_parse_failures": self.dependencies.parse_failures,
                  "context_budget_bytes": self.context_budget, "context_bytes": 0,
                  "budget_limited": False}
        identities, selected = set(), []

        def accept(entry, metadata, limit, definition_line=None):
            expanded = self._expanded(entry, terms, definition_line)
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
                if len(json.dumps(result, ensure_ascii=False).encode()) <= limit - 128:
                    identities.add(identity)
                    if metadata["kind"] == "match":
                        selected.append(variant)
                    return "added"
                result["results"].pop()
            result["budget_limited"] = True
            return "budget"

        has_dependencies = any(self.dependencies.edges.get(e["source"].removeprefix("repo/"))
                               for _, e in candidates[:4] if e["source"].startswith("repo/"))
        primary_budget = int(self.context_budget * 0.6) if has_dependencies else self.context_budget
        deferred = []
        for score, entry in candidates:
            if len(selected) >= 4:
                deferred.append((score, entry))
                continue
            status = accept(entry, {"score": round(score, 4), "kind": "match"}, primary_budget)
            if status == "budget":
                deferred.append((score, entry))
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
                definition_lines = list(dict.fromkeys(definitions[s] for s in symbols if s in definitions)) or [None]
                for line in definition_lines:
                    passage = next((p for p in passages if line is not None
                                    and p["start_line"] <= line <= p["end_line"]), passages[0])
                    if added >= 4:
                        result["dependency_limit_reached"] = True
                        break
                    status = accept(passage, {"score": 0.0, "kind": "dependency",
                                              "via": "repo/" + name, "depth": depth + 1},
                                    self.context_budget, line)
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
            accept(entry, {"score": round(score, 4), "kind": "match"}, self.context_budget)
        result["results_limited"] = len(scored) > len(selected) or result["budget_limited"]
        for _ in range(3):
            result["context_bytes"] = len(json.dumps(result, ensure_ascii=False).encode())
        return json.dumps(result, ensure_ascii=False)
