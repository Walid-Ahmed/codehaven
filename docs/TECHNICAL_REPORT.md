# CodeHaven technical report

CodeHaven (version 0.10.0) is a read-only, offline assistant for understanding code. It indexes a repository on your machine, answers questions about it with a local model served by Ollama (`qwen2.5-coder:7b` by default), and cites the lines it used. It never changes your files, and at runtime it talks only to Ollama on `127.0.0.1`. The one exception is opt-in: with `--allow-clone`, the browser page can download a GitHub repository you ask for.

This report describes how it is built: the parts and how a question moves through them, the agent loop and its tools, how relevant code is found (RAG), how answers are checked, what is stored, and what the retrieval measurements show. For how to use it, see the [README](../README.md).

## Contents

1. [The design in one page](#the-design-in-one-page)
2. [How a question is answered](#how-a-question-is-answered)
3. [The agent loop](#the-agent-loop)
4. [Tools](#tools)
5. [Purpose-built helpers](#purpose-built-helpers)
6. [Retrieval (RAG)](#retrieval-rag)
7. [Answer checks](#answer-checks)
8. [Keeping prompts small](#keeping-prompts-small)
9. [What is stored](#what-is-stored)
10. [Questions in Arabic](#questions-in-arabic)
11. [Offline and read-only guarantees](#offline-and-read-only-guarantees)
12. [The browser page](#the-browser-page)
13. [Code map](#code-map)
14. [Testing and evaluation](#testing-and-evaluation)
15. [Retrieval measurements](#retrieval-measurements)
16. [Known limitations and next steps](#known-limitations-and-next-steps)

## The design in one page

![CodeHaven architecture: the browser page and terminal feed a request router, which sends a request to a purpose-built helper or to the LangGraph agent loop with its five actions; local knowledge supplies passages; answer checks run before the cited answer is shown.](assets/codehaven-architecture.svg)

Three ideas shape the design. Each one comes from watching what a 7B model actually does.

**One agent, not several.** A small model that plans, delegates, and merges results spends its few steps on coordination and loses track. CodeHaven runs a single loop in which the model picks one action at a time.

**Plain code does what a small model does unreliably.** The model describes code well, but it invents line numbers, skips searches it should run, and answers "looks fine" when shown a whole file. So line numbers are looked up in the real file, the names in a question are searched before the model's first step, and a review shows one function per call.

**A request with a known shape gets its own helper.** "Explain my last commit" always starts with `git diff`. "Which functions have no tests?" needs no model at all. A router sends these to purpose-built code, and only open questions reach the agent loop.

## How a question is answered

The same path serves the terminal and the browser page. It starts in `run_state` in [agent.py](../offline_agent/agent.py).

1. **Recall.** In a session, up to two saved answers that share key words with the question are recalled from long-term memory, along with a short record of the conversation so far.
2. **Route.** The question is matched against a few fixed patterns, in this order:

   | The question asks for | What handles it | Uses the model |
   |---|---|---|
   | Git changes ("my last commit", "uncommitted changes") | [changes.py](../offline_agent/changes.py) reads the diff, then the agent explains it | Yes |
   | A review of a named file | [review.py](../offline_agent/review.py), one function per call | Yes |
   | Functions with no tests | [insights.py](../offline_agent/insights.py) | No |
   | A project tour | [insights.py](../offline_agent/insights.py), one sentence per file | Yes |
   | FastAPI entry points | A `discover` node searches for `FastAPI(` before the model runs | Yes |
   | Anything else | The agent loop | Yes |

3. **Prepare the prompt.** For an open question, plain code gathers what the model would otherwise spend steps finding:
   - a sample of up to 40 file paths, ranked by the words they share with the question;
   - the first retrieved passages (see [Retrieval](#retrieval-rag));
   - up to two files the question names, preloaded with line numbers;
   - or, when no file is named, a search for the code names in the question, such as `apply_discount`.
4. **Loop.** The agent loop runs until the model returns `finish` or the step limit is reached.
5. **Check.** The final answer goes through the [answer checks](#answer-checks).
6. **Remember.** A checked answer that cites code is saved to long-term memory, unless `--no-remember` is set.

The question is also classified as a *question*, a *change request*, or *other*. A change request such as "fix `Store.low_stock`" gets the same read-only treatment with one extra instruction: explain what to change and where, and never claim a change was made.

## The agent loop

The loop is a LangGraph `StateGraph` built in `build_graph` in [agent.py](../offline_agent/agent.py). At each step the model returns one JSON object naming an action, and a conditional edge sends it to the matching node.

```text
START -> model -> read, search, retrieve, or run_tests -> model
   |         |-> invalid reply -> correction message -> model (once)
   |         |-> repeated action -> repeat -> model (finish only)
   |         \-> finish -> END
   \-> discover -> model         tool at action limit -> limit -> END
```

| Node | What it does |
|---|---|
| `model` | Calls the local model, validates the reply, and on `finish` builds and checks the answer |
| `read`, `search`, `retrieve`, `run_tests` | Run one tool and add its result to the conversation |
| `discover` | Runs the entry-point search before the model's first step |
| `repeat` | Replaces an action that was already tried with a request to finish |
| `limit` | Ends the run when the step limit is reached |

The state holds the conversation messages, the current reply, the step count, the exit code, and the evidence gathered so far. It lives in memory only; there are no persistent checkpoints.

The guards exist because of specific failures seen in real runs:

- **Step limit.** 12 model replies by default (`--max-steps`, at most 30). The last step is reserved for the answer, so a run always has room to finish.
- **Repeated actions.** An action identical to an earlier one is skipped anywhere in a task, not only back to back, so two alternating lookups cannot use every step.
- **Invalid replies.** Bad JSON, an unknown action (including `edit` or `write`, which do not exist), or a reply cut off at the token limit runs no tool. The model gets one correction message; a second invalid reply in a row ends the task.
- **Repetition.** At temperature 0 a small model can repeat the same sentences until it reaches the token limit. Replies are streamed, so the loop is detected and stopped, and the text before the repetition is kept.

Exit codes: 0 finished, 1 error, 2 step limit or unverified answer, 130 interrupted.

## Tools

The model can ask for five actions. None of them writes to your project.

| Action | What it does | Limits |
|---|---|---|
| `read` | Reads a UTF-8 file, in pages for long files | 64 KB per file; a page fits in half the prompt budget and names the next range |
| `search` | Finds literal text and names the function each match is in | 50 matching lines, 16,000 bytes, 1,000 files |
| `retrieve` | Fetches ranked passages from the local index | 4,000 bytes by default (`--retrieval-budget`) |
| `run_tests` | Runs unittest or pytest in a Docker container, after you approve | Networking off, read-only snapshot, 30-second timeout |
| `finish` | Returns the answer and the code parts it relies on | 10 points, 3,000 characters |

File access goes through [workspace.py](../offline_agent/workspace.py), which rejects path traversal, hidden paths, and symlinks, and applies `.gitignore`, `.agentignore`, and `--exclude` rules ([exclusions.py](../offline_agent/exclusions.py)). The class has no write method. Tests run through [test_runner.py](../offline_agent/test_runner.py); there is no fallback to running them on the host.

## Purpose-built helpers

These answer one kind of request each. Those that use the model make small, bounded calls with their own short prompt and JSON schema.

| Helper | Module | How it works |
|---|---|---|
| Git changes | [changes.py](../offline_agent/changes.py) | Runs only `git diff`, `git log`, `git rev-parse`, and `git ls-files`. The diff must fit in half the prompt budget; named files come first, then the smallest changes. |
| File review | [review.py](../offline_agent/review.py) | Up to 12 functions, one model call each. The model states what the function promises and what it does before giving a verdict. |
| Test gaps | [insights.py](../offline_agent/insights.py) | Lists public functions whose names appear in no test file. No model. |
| Project tour | [insights.py](../offline_agent/insights.py) | Orders up to 8 files by the import graph; the model adds one sentence per file. |
| Structure map | [study.py](../offline_agent/study.py) | Draws calls inside a file, or imports between files, from parsed code. No model. |
| Flashcards | [study.py](../offline_agent/study.py) | The model writes 3 to 6 cards per file and names the part each is about; the lines are looked up. |
| Picture to code | [ocr.py](../offline_agent/ocr.py) | Tesseract reads the picture; indentation is rebuilt from word positions. No model. |
| Notebooks and PDFs | [documents.py](../offline_agent/documents.py) | Converts a notebook or PDF to a text file that can be read like any other. No model. |

## Retrieval (RAG)

Before the model answers, CodeHaven fetches relevant passages from the repository and adds them to the prompt. The model can also call `retrieve` later with its own query. Each passage carries its source and line range, so answers can cite it. The index is built in [knowledge.py](../offline_agent/knowledge.py).

Passages are found four ways:

- **Keyword search (BM25).** Ranks passages by the words they share with the question.
- **Definition lookup.** If the question names something in the code, such as `MAX_GUESTS` or `Store.remove_item`, its definition comes first.
- **Import following.** Adds the functions a matched file imports, up to two hops away.
- **Meaning search (optional).** With a local embedding model installed, passages are also ranked by how close their meaning is to the question.

### How each method works

**Keyword search.** Each file is split into passages of up to 30 lines (about 1,800 characters) and ranked with BM25. Words are lowercased, `camelCase` and `snake_case` names are split, and common endings are trimmed, so `applyDiscount` matches "apply discount" and "restocking" matches `restock`. Different files are preferred before more passages from the same file.

**Definition lookup.** When the index is built, CodeHaven records where each function, class, method, and module constant is defined, using Python's `ast` module and Tree-sitter for JavaScript and TypeScript. A word in the question counts as a code name only when it is written like code (underscore, camelCase, `ALL_CAPS`, a dot, backticks, or `()`), or when it sits next to "function", "method", "class", or "constant". Only names defined in one to three files are looked up, and at most three definitions are added. For a method, the passage is the method itself, not its whole class.

**Import following.** For Python, JavaScript, TypeScript, Go, Java, and Rust, CodeHaven adds the definitions that matched files import from other local files, up to two hops and four passages. Python uses `ast` ([dependencies.py](../offline_agent/dependencies.py)), JavaScript and TypeScript use Tree-sitter ([javascript.py](../offline_agent/javascript.py)), and Go, Java, and Rust use line patterns ([languages.py](../offline_agent/languages.py)). Package imports and path aliases are not resolved.

**Meaning search.** [semantic.py](../offline_agent/semantic.py) embeds every passage with a second, small Ollama model (the first installed of `nomic-embed-text`, `embeddinggemma`, and others, or `--embed-model`). The 24 nearest passages above a similarity of 0.2 are mixed with the keyword ranking by reciprocal rank fusion: a passage's place in each list counts, not the two scores, which are on different scales. Exact names therefore still win. Passages are embedded in the background, so a question uses whatever is ready. If the embedding model fails, search goes back to key words for the session.

**Whole functions.** Passages are widened to the complete function or class when it fits, and carry the values of up to three module constants it uses, such as `MAX_GUESTS = 4`.

### Limits

| Limit | Value |
|---|---|
| Retrieved context per call | 4,000 bytes by default (`--retrieval-budget`, 2,000 to 16,000) |
| Documents indexed | 1,000 |
| Indexed text | 5 MB, at most 8,000 passages |
| Largest file indexed | 64 KB |

Hidden files, `.gitignore` and `.agentignore` matches, symlinks, binary files, and invalid UTF-8 are skipped. The keyword index is rebuilt at the start of each run and between questions, so edits you make yourself are picked up.

## Answer checks

An answer is an overview plus up to 10 points, and each point names a part of the code: a function, a class, `Class.method`, a route, or a file. CodeHaven finds that part's real lines and prints `path:start-end` itself. The checks are in [evidence.py](../offline_agent/evidence.py) and `with_points` in [agent.py](../offline_agent/agent.py).

- **Line numbers are looked up, not trusted.** Python files are parsed, never run; other names match the first line that mentions them.
- **Citations must be supported.** Every `path:line` in the final answer must fall inside code that was read, searched, or retrieved during the run. Up to 50 evidence records (4,000 bytes) are kept for this.
- **Gaps are shown.** A missing or unsupported citation, or a name that cannot be found, marks the answer **Unverified**, and the run exits with code 2.

These checks confirm that an answer points at real code that was read. They do not prove the explanation is right, so the cited lines are still worth reading.

## Keeping prompts small

A 7B model has a small context window, and long prompts are slow on a laptop. Several bounds keep each request short ([model.py](../offline_agent/model.py)):

- **Prompt budget.** Conversation content is limited to 12,000 UTF-8 bytes by default (`--prompt-budget`).
- **Compaction.** When the budget is reached, older tool results are replaced with one-line stubs, oldest first. Messages are dropped only if that is not enough. The system rules and the task always stay.
- **Session history.** A follow-up question sees the last two answers nearly whole, up to six older questions as one sentence each, and the locations cited so far. Tool output is not carried over.
- **Output limit.** 1,024 tokens per reply by default (`--max-output-tokens`).
- **Stable prefix.** The system prompt and task stay at the start of the conversation, so Ollama can reuse its prompt cache between steps.

## What is stored

Nothing is ever written inside your repository. What CodeHaven keeps is under `~/CodeHaven`, readable by your user only, and `--no-remember` turns all of it off.

| Folder | Holds | Module |
|---|---|---|
| `memory/` | Checked, cited answers, one file per repository. Each records a fingerprint of the files it cites and is deleted when one of them changes. Up to 200 per repository. | [memory.py](../offline_agent/memory.py) |
| `chats/` | Chats from the browser page. Up to 30 per repository, 60 questions each. | [chats.py](../offline_agent/chats.py) |
| `index/` | Embedding vectors and hashes for meaning search. Never code text. | [semantic.py](../offline_agent/semantic.py) |
| `repos/` | Repositories downloaded with `--allow-clone`. | [gui.py](../offline_agent/gui.py) |

## Questions in Arabic

A question asked in Arabic is answered in Arabic. The support is in [arabic.py](../offline_agent/arabic.py) and adds no model and no dependency.

- **Detecting the language.** Quoted text, paths, and code names are set aside, then Arabic and Latin words are counted. A task is Arabic when it has at least two Arabic words and they are at least half as many as the Latin ones, or when its one Arabic word is not outnumbered by Latin words.
- **One spelling.** `normalise` removes short vowels and stretching, folds the forms of alef into one, writes ى as ي and ة as ه, and turns Arabic-Indic digits into Western ones. Every pattern is written in that spelling.
- **Routing.** Arabic patterns sit beside the English ones for the question type (question, change request, other) and for Git changes, file review, test gaps, and the project tour, so an Arabic request reaches the same helper.
- **Search.** The index tokenizer normalises Arabic words and drops "ال" and common endings, so keyword search matches Arabic comments, strings, and documentation. Code names stay in Latin letters, so an Arabic question finds code through the names it mentions, or through meaning search with a multilingual embedding model.
- **The page.** [gui.html](../offline_agent/gui.html) is one page in two languages. Its English text is the key each Arabic sentence is looked up by, in a table inside the page, so nothing is loaded from anywhere. Choosing Arabic translates the page once as it loads, sets it right to left, and is remembered in the browser's own storage. A test checks that every sentence the page writes has an Arabic one.
- **Fixed sentences.** The review, test-gap, and tour reports are written by code, not the model, so each has an Arabic wording chosen by the language of the question. Counts use the Arabic forms for one, two, 3 to 10, and 11 or more.
- **The prompt.** One instruction is added to the last message: answer in Arabic, and keep paths, code names, and citations exactly as written. English tasks get nothing added, so their prompts, and the eval results, are unchanged.
- **The page.** Arabic text is laid out right to left; code and citations are isolated left to right, so `store.py:12-20` is never reordered.

The answer checks are the same for both languages. Answer quality in Arabic has not been measured: the evals contain English questions only.

## Offline and read-only guarantees

- **Loopback only.** The Ollama endpoint must be a numeric loopback address. Proxies and redirects are disabled.
- **Local models only.** Cloud-backed, remote-backed, and embedding-only models are refused as chat models.
- **No tracing.** Every run is wrapped in `tracing_context(enabled=False)`, so LangSmith environment variables cannot turn uploads on.
- **No write path.** The workspace layer has no write method, and no graph node writes files. The evals fail any task that changes a file.
- **Isolated tests.** Tests run in Docker with networking disabled, against a read-only snapshot.
- **One opt-in exception.** `--allow-clone` lets the browser page download a GitHub repository you ask for. Nothing is uploaded, and without the flag the server refuses to clone.

## The browser page

`codehaven --gui` starts a small HTTP server from the standard library ([gui.py](../offline_agent/gui.py)) and serves one self-contained page ([gui.html](../offline_agent/gui.html)). The page loads nothing from the internet; its Content-Security-Policy blocks outside scripts, styles, and fonts.

- The server listens on `127.0.0.1` only.
- Every request must carry a secret token from the link printed in the terminal and name this machine in its Host header, so another website open in your browser cannot use the server to read your files.
- Answers stream to the page as the model writes them.
- **Stop** ends an answer or flashcards in progress. The server closes its connection to Ollama ([model.py](../offline_agent/model.py)), which stops generation and also ends the wait for a model that is still loading. A stopped question is not remembered or saved.
- An uploaded file is copied to a private temporary folder and deleted when you open something else or stop the server.

## Code map

All code is in [offline_agent/](../offline_agent).

| Module | Role |
|---|---|
| [cli.py](../offline_agent/cli.py) | Command-line entry point and flags |
| [gui.py](../offline_agent/gui.py), [gui.html](../offline_agent/gui.html) | The browser page and its local server |
| [agent.py](../offline_agent/agent.py) | Routing, prompt preparation, the LangGraph loop, and sessions |
| [model.py](../offline_agent/model.py) | The Ollama client: loopback checks, streaming, repetition guard, compaction |
| [workspace.py](../offline_agent/workspace.py), [exclusions.py](../offline_agent/exclusions.py) | Safe, read-only file listing, reading, and search |
| [knowledge.py](../offline_agent/knowledge.py) | The index: passages, BM25, definition lookup, retrieval budget |
| [semantic.py](../offline_agent/semantic.py) | Meaning search with a local embedding model |
| [dependencies.py](../offline_agent/dependencies.py), [javascript.py](../offline_agent/javascript.py), [languages.py](../offline_agent/languages.py) | Import graphs and symbol spans per language |
| [evidence.py](../offline_agent/evidence.py) | Retained evidence and the final answer checks |
| [changes.py](../offline_agent/changes.py), [review.py](../offline_agent/review.py), [insights.py](../offline_agent/insights.py), [study.py](../offline_agent/study.py) | Purpose-built helpers |
| [documents.py](../offline_agent/documents.py), [ocr.py](../offline_agent/ocr.py) | Notebooks, PDFs, and pictures of code |
| [memory.py](../offline_agent/memory.py), [chats.py](../offline_agent/chats.py) | Long-term memory and saved chats |
| [arabic.py](../offline_agent/arabic.py) | Questions asked in Arabic: language detection, spelling, and request patterns |
| [test_runner.py](../offline_agent/test_runner.py) | Isolated test runs in Docker |
| [evaluation.py](../offline_agent/evaluation.py) | The eval harness |

## Testing and evaluation

**Unit tests.** `python3 -m unittest discover -s tests` runs more than 300 tests in about 15 seconds. They use fake model replies and a loopback HTTP fixture, so they need no model and no network. They check the workflow, the limits, and the safety rules. They say nothing about answer quality.

**Evals.** `python3 -m offline_agent.evaluation` runs fixed questions against the real local model: 12 tasks over [evals/sample_repo](../evals/sample_repo), and a second set over [evals/bookings_repo](../evals/bookings_repo) to check that a change is not fitted to the first. Each answer is checked by regular expressions, and a task fails if any file in its copy of the repository changed. The evals use keyword search only, so results stay comparable from run to run.

Results shift when prompts change, so one pass or fail is weak evidence. All tuning and measurement used `qwen2.5-coder:7b`.

## Retrieval measurements

Measured on 2026-11-10, before meaning search was added, by checking whether the first retrieval for each question contains the line that holds the answer. These numbers use retrieval only, with no model, at the default 4,000-byte budget.

| Question set | Keyword search only | With definition lookup |
|---|---|---|
| Eval questions, `evals/sample_repo` and `evals/bookings_repo` (30) | 20 | 20 |
| Eval questions reworded in plain English (15) | 12 | 12 |
| Questions about CodeHaven's own code, using its names (11) | 4 | 8 |
| The same questions in plain English (11) | 0 | 0 |

What they show:

- **The eval repositories are too small to test retrieval.** With 6 to 12 files, most of each repository comes back for every question. CodeHaven's own repository (68 indexed files at the time) shows the real gaps.
- **Definition lookup doubled the hits for questions that name code.** Keyword ranking had missed definitions because tests and callers repeat a name more often than the line that defines it.
- **The 10 eval misses are not wording problems.** Five are entries deep in long data lists (a 600-row catalog, a 450-row holiday list), four ask who calls a function, and one is a line outside any function.
- **Plain-English questions on a real repository were the main gap.** Keyword search found none of the 11. This is the gap meaning search was added to close in version 0.10. It has not yet been measured on these question sets.

## Known limitations and next steps

- **Measure meaning search.** Repeat the table above with an embedding model installed, to learn how much of the plain-English gap it closes.
- **Long data lists.** Index each passage with its file's path, so a question that names `inventory/catalog.py` and `SKU-0042` reaches the right rows.
- **"Who calls X?"** Retrieval favours a function's definition over its callers. The search that runs before the model's first step already finds callers for named functions; retrieval could add the passages that call a named function.
- **Reviews see one function at a time.** A review cannot judge how functions work together, and it can miss a bug or flag correct code.
- **Line-pattern parsing.** Go, Java, and Rust imports are read with line patterns, so unusual layouts and macros can be missed.
- **Arabic is unmeasured.** Add Arabic questions to the evals to learn how well the 7B model answers them.
- **Small models.** Results shift when prompts change, and other models may answer differently from `qwen2.5-coder:7b`.
- **Not an OS sandbox.** File access relies on path checks, which are not hardened against files being moved while CodeHaven runs.
