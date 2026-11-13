# CodeHaven technical report

CodeHaven (version 0.9.1) is a read-only, offline assistant for understanding code. It indexes a repository on your machine, answers questions about it with a local model served by Ollama (`qwen2.5-coder:7b` by default), and cites the lines it used. It never changes files and makes no network requests at runtime.

This report describes how it works: the agent loop, how it finds relevant code (RAG), how answers are checked, and what the retrieval measurements show.

## Architecture

CodeHaven runs one agent, built as a LangGraph `StateGraph` ([agent.py](../offline_agent/agent.py)). At each step the local model returns one JSON action, and the graph sends it to the matching node:

| Action | What it does |
|---|---|
| `read` | Reads a project file, in pages for long files |
| `search` | Finds literal text in project files and names the function each match is in |
| `retrieve` | Fetches ranked passages from the local index (see [RAG](#retrieval-rag)) |
| `run_tests` | Runs unittest or pytest in an isolated Docker container, after approval |
| `finish` | Returns the final answer |

```text
START -> model -> read, search, retrieve, or run_tests -> model
             |-> invalid reply -> correction message -> model (once)
             \-> finish -> END
                    tool at action limit -> limit -> END
```

Requests to find FastAPI entry points take a separate discovery path first: a broad search for `FastAPI(`, then one cited summary. The graph state lives in memory only; there are no persistent checkpoints. The last step is always kept for the answer, and an action repeated identically is skipped instead of run again.

There is one agent, not several. Work a small model does unreliably, such as looking up a name before answering, is done by deterministic code before the model's first step.

## Retrieval (RAG)

RAG: yes. Before the model answers, CodeHaven fetches relevant passages from your repository and adds them to the prompt. They are found three ways:

* Keyword search (BM25). An in-memory index ranks passages by the words they share with the question.
* Definition lookup (new). If the question names something in the code, like `MAX_GUESTS` or `Store.remove_item`, its definition comes first.
* Import following. It adds the functions a matched file imports, up to two hops away.

The model can also call `retrieve` later with its own query. Each passage carries its source and line range, so answers can cite it.

### How each method works

**Keyword search.** [knowledge.py](../offline_agent/knowledge.py) splits each file into passages of up to 30 lines (about 1,800 characters) and ranks them with BM25. Words are lowercased, `camelCase` and `snake_case` names are split, and common endings are trimmed, so `applyDiscount` matches "apply discount" and "restocking" matches `restock`. Different files are preferred before more passages from the same file.

**Definition lookup.** When the index is built, CodeHaven records where each function, class, method and module constant is defined, using Python's `ast` module and Tree-sitter for JavaScript and TypeScript. A word in the question counts as a code name only when it is written like code (underscore, camelCase, `ALL_CAPS`, a dot, backticks or `()`), or when it sits next to "function", "method", "class" or "constant". Only names defined in one to three files are looked up, and at most three definitions are added. For a method, the passage is the method itself, not its whole class.

**Import following.** For Python, JavaScript and TypeScript, CodeHaven adds the definitions that matched files import from other local files, up to two hops and four passages. Package imports and path aliases are not resolved.

**Whole functions.** Passages are widened to the complete function or class when it fits, and carry the values of up to three module constants it uses, such as `MAX_GUESTS = 4`.

### Limits

| Limit | Value |
|---|---|
| Retrieved context per call | 4,000 bytes by default (`--retrieval-budget`, 2,000 to 16,000) |
| Documents indexed | 1,000 |
| Indexed text | 5 MB, at most 8,000 passages |
| Largest file indexed | 64 KB |

Hidden files, `.gitignore` and `.agentignore` matches, symlinks, binary files and invalid UTF-8 are skipped. The index is rebuilt at the start of each run, so edits you make yourself are picked up.

### No vector database

CodeHaven uses no embeddings and no vector database. Matching is by words and code structure, not by meaning, and nothing is saved to disk. This keeps it fast and free of extra models and stored indexes.

The cost is that questions using none of the code's words, such as "where do we handle login?" when the code says `session`, can miss. The measurements below show this is the main remaining gap. They also show that most misses on the eval questions would not be fixed by embeddings either.

## Answer checks

Answers to questions are an overview plus up to 10 points, each naming a part of the code. CodeHaven finds that part's real lines and prints `path:start-end` itself, because small models describe code well but make up line numbers. A final answer must cite lines that were actually read, searched or retrieved. Answers with missing or unsupported citations are shown as **unverified** and the run exits with code 2.

## Offline and read-only

- Ollama must be at a loopback address, and cloud-backed, remote-backed and embedding-only models are refused.
- LangSmith tracing is forced off for every run (`tracing_context(enabled=False)`), so environment variables cannot turn uploads on.
- The browser page listens on 127.0.0.1 only, with a per-session token.
- The workspace layer has no write method, and no graph node writes files.
- Tests run in Docker with networking disabled, against a read-only snapshot, with a 30-second timeout.

## Retrieval measurements

Measured on 2026-11-10 by checking whether the first retrieval for each question contains the line that holds the answer. These numbers use retrieval only, with no model, at the default 4,000-byte budget.

| Question set | Keyword search only | With definition lookup |
|---|---|---|
| Eval questions, `evals/sample_repo` and `evals/bookings_repo` (30) | 20 | 20 |
| Eval questions reworded in plain English (15) | 12 | 12 |
| Questions about CodeHaven's own code, using its names (11) | 4 | 8 |
| The same questions in plain English (11) | 0 | 0 |

What they show:

- **The eval repositories are too small to test retrieval.** With 6 to 12 files, most of each repository comes back for every question. CodeHaven's own repository (68 indexed files) shows the real gaps.
- **Definition lookup doubled the hits for questions that name code.** Keyword ranking had missed definitions because tests and callers repeat a name more often than the line that defines it.
- **The 10 eval misses are not wording problems.** Five are entries deep in long data lists (a 600-row catalog, a 450-row holiday list), four ask who calls a function, and one is a line outside any function.
- **Plain-English questions on a real repository are the main gap.** A prototype that asked the model to rewrite the question into free keywords made no difference. A second prototype, where the model picks up to five names from a list of the repository's own symbols, found all 15 reworded eval questions, up from 12. It took about 100 seconds per question, but another model was loaded in Ollama at the same time, so its real speed is unknown.

## Known limitations and next steps

- **Plain-English questions.** Time the name-picking step with the 7B model loaded on its own; ship it if it is fast enough. Embeddings, cached on disk per repository and run locally, remain the fallback.
- **Long data lists.** Index each passage with its file's path, so a question that names `inventory/catalog.py` and `SKU-0042` reaches the right rows.
- **"Who calls X?"** Retrieval favours a function's definition over its callers. The pre-step search already finds callers for named functions; retrieval could add the passages that call a named function.
- **Small models.** Results shift when prompts change, so one pass or fail on an eval is weak evidence. All tuning and measurement used `qwen2.5-coder:7b`; other models may answer differently.
