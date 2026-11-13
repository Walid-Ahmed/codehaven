# Repository RAG and your own reference documents

The agent has a local reference knowledge base. You can add manuals, team coding
standards, architecture notes, API descriptions, and worked examples. Retrieval
adds relevant passages to the model's context; it does not retrain the model.
No URLs are fetched, no embedding model is downloaded, and indexing never calls
Ollama or any network service. qwen2.5-coder:7b remains the only model.

## Automatic repository RAG

Point `--workspace` at a local repository:

```sh
offline-agent --workspace /path/to/repo 'Explain how authentication works'
```

No copy step or separate index command is required. The selected repository's
supported source code and text documentation are automatically indexed alongside
the bundled guide and your extra references. The initial task retrieves relevant
passages; the model can call `retrieve` for more. Code citations use a `repo/`
prefix, for example `repo/src/auth.py:1-30`. Reference documents under your chosen
knowledge directory keep the `user/` prefix; the same physical file is not indexed
twice. Startup reports repository and reference counts and warns about index limits.

Supported code includes Python, JavaScript/TypeScript, Go, Rust, Java, C/C++, C#,
Ruby, PHP, Swift, Kotlin, shell scripts, SQL, HTML/CSS, Vue/Svelte, and other common
text source formats. Common JSON/YAML/TOML/XML configuration files and build files
such as Dockerfile and Makefile are included. See CODE_EXTENSIONS and CODE_NAMES
in offline_agent/knowledge.py for the exact set. Arbitrary unknown extensions,
images, compiled artifacts, and notebooks are not included.

This is a bounded snapshot, not a guarantee every repository file is indexed.
The existing hidden-path, excluded-directory, UTF-8, file-size, and global index
limits still apply. Indexing honors `.gitignore`, `.agentignore`, and CLI exclusions.
Eligible visible files may still contain private data. Only select repositories you intend to expose to
your local model. Larger repositories may need `search`/`read` for files outside
the index. A directory need not have Git metadata to work.

CodeHaven never writes files. The index is rebuilt at the start of each run and
between questions in a session, so your own edits are picked up on the next
question (there is no filesystem watcher). Earlier conversation excerpts remain
snapshots; retrieve or read again when validating changed code.

## Add documents

Create a knowledge folder inside the project you are working on, provided its
ignore rules permit that folder:

```sh
mkdir -p /path/to/project/knowledge
```

Copy your own `.md`, `.txt`, or `.rst` documents into that folder. For example:

```text
my-project/
  knowledge/
    langgraph/
      state-graph.md
      conditional-edges.md
    team-style-guide.md
    architecture.txt
```

Then ask a coding question:

```sh
offline-agent --workspace /path/to/project \
  'Use my references to explain how to route a LangGraph action'
```

Startup tells you where to add references and how many documents were indexed.
The task automatically retrieves relevant passages. The model can request more
using the `retrieve` tool, with its question in `content` and an empty `path`.
Results include source names and line ranges, such as
`user/team-style-guide.md:12-25`. The model is instructed to cite those sources;
review its citations because generation can still make mistakes.

## Use a separate reference library

```sh
offline-agent --workspace /path/to/project \
  --knowledge-base /path/to/my-reference-library \
  'Consult my architecture references before proposing a change'
```

The supplied directory must exist. Documents can be outside the project, but only
the user-configured reference root is indexed. The model cannot select a different
root. Knowledge retrieval reads documents; it never edits that reference library.
If references are inside the workspace, its ignore rules also apply to them.

## LangGraph manuals

Project-authored LangGraph documentation notes ship with the package and are always
included: `langgraph.md` (CodeHaven workflow), `langgraph-graphs.md` (construction),
`langgraph-checkpoints.md` (persistence), and `langgraph-approvals.md` (interrupts).
Each topic includes official provenance links and a review date. These are offline
reference notes, not the full official manual, and do not enable new agent features.
Try asking: `codehaven 'Explain LangGraph checkpointing using the bundled references'`.
To add a full manual, place text or
Markdown copies that you already have permission to use in your local knowledge
folder. Split long manuals into smaller files (one topic/page per file), each no
larger than 64 KB. Include the manual version and source URL in the document text
for provenance. URLs remain text; the agent never follows them. There is no
website downloader or automatic documentation updater.

PDF, Word, HTML, and images are not supported by this first version. Export their
useful content to Markdown or UTF-8 plain text before adding it. Only add documents
you intend to expose to your local coding model. In this repository `knowledge/`
is ignored by Git and therefore not indexed. To keep personal references outside
Git while using them for RAG, use an external library via `--knowledge-base`.
See [file exclusions](EXCLUSIONS.md) for rule scope and precedence.

## How local RAG works

At startup, the agent reads eligible repository and reference files and builds an in-memory BM25
keyword index. It breaks documents into passages of up to 30 lines / about 1,800
characters and ranks them by query term overlap and rarity. It returns up to four
keyword-matched passages plus up to four Python/JavaScript/TypeScript dependency passages with citations
and a shared 16 KB JSON cap. There are no embeddings or semantic
vector search, so queries using the document's actual terminology work best.

When a question names code, such as `MAX_GUESTS`, `apply_discount`, `Store.remove_item`, `applyCoupon`,
or "the stem function", the definition of that name comes first, as a passage with `kind: definition`,
`via: question`, and `name`. Keyword ranking alone often missed these, because tests and callers repeat a
name more often than the line that defines it. A word counts as a name only when it is written like code
(underscore, camelCase, ALL_CAPS, a dot, backticks, or `()`) or sits next to "function", "method", "class",
or "constant", and only when it is defined in one to three indexed Python/JavaScript/TypeScript files.
At most three definitions are added, within the same byte budget.

The index is rebuilt for each new task. External edits and deletions are picked
up on the next run. No database, persistent index, account, or API key is required.
Indexing considers at most 1,000 candidate files per root, 1,000 documents total across references and repository code,
5 MB of document text, and 8,000 passages. Hidden/excluded paths, symlinks, binary
files, invalid UTF-8, and files over 64 KB are excluded. Metadata flags report
index/result limits; a no-match result is not proof the full library has no answer.
The skipped-document count covers read failures among eligible text candidates,
not every excluded path.

References are untrusted data. The model is instructed to use them as evidence,
not follow embedded commands. This is guidance rather than a prompt-injection
security boundary. CodeHaven cannot write files, and test runs still need approval. Project tests now run in isolated Docker containers with no external network
and a read-only project snapshot; see [isolated tests](ISOLATED_TESTS.md).
RAG itself performs no network requests.

## Dependency-aware Python retrieval

Python repository files (`.py` and `.pyi`) are parsed with Python's standard AST
parser at indexing time. No project module is imported or executed. When a
keyword-matched passage comes from a Python file, retrieval follows its statically
resolved local imports, even if the dependency has no query keyword overlap.

For example, a query matching `orders.py` can return a definition from
`pricing.py` imported with `from pricing import calculate_total`. Absolute imports,
relative imports (`from .pricing ...`, `from ..models ...`), aliases, local package
initializers, and common repository-root / `src/` layouts are supported. A sibling
module is also considered as a fallback for script-style layouts. Resolution only
considers indexed repository paths and never searches installed packages, the
network, or arbitrary files elsewhere on the device.

For a named import, the retriever prefers the passage containing the imported
top-level function, class, or simple assigned constant. If no such definition can
be found, it uses the dependency's first passage. A passage may be selected for
each imported symbol, within the global dependency budget; not every definition
or entire body is guaranteed to fit. Module imports default to the first passage
unless query terms identify a more relevant parsed symbol there.

Expansion follows up to two hops from the keyword-matched files, adding at most
four dependency passages. Cycles and repeated passages are deduplicated. The total
keyword and dependency output shares the configurable JSON budget (16 KB by default). Dependency results
have `kind: dependency`, `via` (the importing source), and `depth`; normal results
have `kind: match`. `dependency_limit_reached` reports depth/count/output limits;
`dependency_parse_failures` counts Python and JavaScript/TypeScript parse failures. A syntax
error does not prevent that file's text from participating in keyword retrieval.

This is static import assistance, not a complete Python interpreter or call graph.
Conditional/function-local imports may be included even when not executed. Custom
`sys.path`, dynamic imports, star-import name resolution, runtime monkey-patching,
ambiguous package layouts, reverse caller lookup, and third-party dependencies are
not resolved. JavaScript/TypeScript support is described below; other languages retain keyword-only retrieval. Syntax unsupported by
the Python version running the agent is skipped by dependency analysis. Index
limits may omit import targets; absence of a dependency passage is not evidence
that no dependency exists. Use search/read for missing context.

The dependency graph refreshes with the local index on each new task. It does not add internet access or another model.

## JavaScript and TypeScript dependencies

Local Tree-sitter grammars analyze `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, `.tsx`,
`.mts`, and `.cts` source files. No Node.js runtime is required, and project code
is never evaluated during indexing. Install/update the Python package once while
online (`python -m pip install -e .`); parsers are local runtime libraries and do
not fetch grammar files, documentation, or models.

Supported dependencies are relative ES imports (including aliases, defaults, type
imports, namespace imports, and side effects), relative re-exports, and literal
single-argument `require('./module')` calls. Resolution considers exact indexed
paths, common source extensions, directory index files, and `.js` to `.ts`/`.tsx`
source substitutions (`.mjs` to `.mts`, `.cjs` to `.cts`).

Named and default function declarations, arrow/function variable declarations,
methods, classes, interfaces, and type aliases have source ranges. Local named
export aliases and named re-export chains can guide definition retrieval.
CommonJS require edges are supported but do not implement full `module.exports`
symbol resolution. Bare package names, tsconfig paths, bundler aliases, package.json
exports, computed paths, and dynamic `import()` are not resolved. Comments and
strings that resemble imports do not create edges. Syntax errors leave the file
available for keyword retrieval but skip its dependency/symbol analysis. The
parse-failure count combines Python and JavaScript/TypeScript failures.

## Retrieval budgets and complete symbols

The CLI default retrieval budget is 4,000 UTF-8 bytes (library default: 16,000), configurable from 2,000 to
16,000 using `--retrieval-budget BYTES`. Library callers can pass
`KnowledgeBase(..., context_budget=8000)`. The setting persists across index refresh.
`context_budget_bytes` and `context_bytes` describe the cap and actual serialized
JSON size. `budget_limited` records budget restrictions during candidate selection;
it can be true even when deferred candidates later fit after reclaiming space.

Keyword ranking is still BM25. Selection considers the first match from different
files before repeated passages from the same file, examining up to 64 ranked
candidates and keeping at most four keyword results. If those matches have local
imports, 40% of the budget is initially reserved for dependencies. Unused space
can then accommodate deferred matches whose dependency graphs were already
considered (or have no known dependencies). Up to four dependency passages across
two hops share the same total budget. Oversized passages are skipped while smaller
candidates are still considered; the first over-budget passage no longer ends all
retrieval.

Python AST and Tree-sitter source ranges let the retriever expand a relevant chunk
to a complete function/method/class or supported TS declaration. Python decorators
are included. For named imports, it targets the corresponding definition, and can
include several imported definitions from the same module within the four-passage
limit. Anonymous default JS function/arrow exports are supported. Duplicate full
symbol ranges are not repeated.

Full symbols are preferred if they fit at selection time. Otherwise, retrieval
falls back to the original bounded chunk and marks `symbol_complete: false`.
Complete symbols have `symbol_complete: true` and a `symbol` name. Ordinary text
passages or code without a recognized symbol may omit these fields. A complete
symbol can still depend on imports or module-level state not present in its body;
read/search/retrieve remain available for more context. Long single-line chunks
and enclosing blocks may remain partial. No function is guaranteed to fit.

These are retrieval byte budgets, not exact model token counts. System prompts,
file listings, read results, model outputs, and conversation history are outside
this cap. There is no automatic conversation compaction yet. Existing index/file
limits still apply. This upgrade does not add semantic embeddings, network access,
or another model.
