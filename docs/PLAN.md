# Implementation plan and roadmap

## Product goal

Build a coding assistant that runs with a model on the user's own device, with no paid API or runtime internet dependency. Deliver a small CLI first and expand it after real usage. "Free" describes the absence of a required paid service; hardware, electricity, and model licenses remain the user's responsibility.

## Version 0.1: inspect, propose, review

Original v0.1 architecture (superseded by v0.2 below):

```text
User task + workspace
        |
Python CLI --> bounded agent loop --> local Ollama HTTP API --> installed model
                       |
                validated action
                       |
           read file / preview diff / finish
                       |
             user confirms each write
```

1. Start with Python 3.10+ and no runtime packages for a portable baseline.
2. Use Ollama as the first inference adapter, with the model fixed to `qwen2.5-coder:7b` and no fallback.
3. Enumerate bounded project context; let the model select files to read.
4. Use a JSON action protocol independent of native tool-calling support.
5. Show diffs and require per-file consent in write mode.
6. Validate paths, content size, action shape, step count, and local endpoint.
7. Add offline unit tests, transport integration coverage, and GitHub CI.
8. Publish source and usage guidance in a new repository.

Acceptance criteria: preview mode never writes; declined writes never apply; confirmed writes match the preview; traversal and symlinks are rejected; remote endpoints are rejected; a fake model can complete the read/write/finish workflow; the application reports unavailable Ollama clearly. A live model smoke test depends on an available local service and installed model and must be reported separately from fixture tests.

## Version 0.2: LangGraph orchestration

Implemented a typed in-memory StateGraph with model, read, write, and limit nodes. Conditional edges dispatch validated actions and stop at finish or the model-action limit. The existing workspace restrictions and per-file confirmation remain. LangGraph and LangSmith are installed during setup, with tracing explicitly disabled during runs. There is no hosted graph service, checkpoint persistence, or automatic retry. Inference remains restricted to local `qwen2.5-coder:7b`.

## Next

Since v0.9 CodeHaven is read-only and focused on understanding code; editing was removed (see Version 0.9). Next, in order:

- Run the evals again on the converted question sets, and record a new baseline for the default model.
- Reduce padded answers. The model often adds invented reference lists after a correct answer; they are flagged as unverified, but they make answers noisy.
- Evaluate on real repositories. Both eval repositories are small, and the questions are short.
- Add a Stop button to the browser page for an answer in progress.

## Version 0.2.1: approved unittest execution

Added a run_tests graph node using fixed unittest discovery under tests/, per-run approval, a 30-second timeout, bounded 16 KB output, and POSIX process-group cleanup. Results feed back into the model conversation. This is explicitly limited to trusted projects on macOS/Linux and is not an isolated runner; test code retains user file/network permissions. Model selection remains fixed to qwen2.5-coder:7b.

## Isolated verification loop (shipped in v0.7)

Add explicitly approved test commands in an OS-isolated runner, bounded output and timeouts, failure feedback, and clear test evidence. General command execution and confinement of untrusted tests require a separate security design; a fixed command alone is insufficient. Keep network disabled for the runner.

## Later versions

Consider llama.cpp or another local inference adapter, a desktop UI, resumable sessions, repository indexing, and editor integration. Keep the transport boundary and workspace layer separate so these additions do not require rewriting the agent loop.

## Initial validation record

Automated tests exercise simulated model actions and a local HTTP server. See the repository history and CI for results. Actual local inference quality and hardware requirements are not established by these tests.

Local validation on November 1, 2026: all 9 automated tests passed on Python 3.13.3. The installed `qwen2.5-coder:14b` model passed the doctor check, proposed `hello.py` in preview mode, and completed a separate confirmed-write run in a temporary directory. The generated addition function passed two manually executed assertions. This is a small smoke test, not a general coding-quality benchmark. Python 3.10 and Linux checks are configured in CI but were not run locally.

Current model policy: use only `qwen2.5-coder:7b`. The earlier 14b smoke test above is historical and does not validate the current 7b configuration. The CLI no longer accepts a model-selection option.

Version 0.2 validation on November 1, 2026: all 21 tests passed on Python 3.13.3 with LangGraph 1.2.19, and `pip check` reported no broken requirements. A real `qwen2.5-coder:7b` run read a disposable Python file and finished with an explanation in two model actions. Tests also cover the 30-action boundary, approved writes at the limit, and disabled tracing with inherited tracing environment flags. Other Python versions remain covered by the CI configuration rather than local verification.

Version 0.2.1 validation: all 29 automated tests passed on Python 3.13.3. The real local qwen2.5-coder:7b model requested run_tests on a disposable unittest project, received exit code 0 with one passing test, and reported that result. Runner tests cover failures, timeouts, output truncation, missing test directories, rejected custom commands, denied approval, and result feedback into the graph.

## Version 0.2.2: literal file content search

Added a read-only search graph node and scoped Workspace.search. Model arguments reuse content for the query and path for optional file/directory scope. Search is literal and case-sensitive, bounded to 1,000 candidate files, 50 matching lines, and 16 KB of JSON, using existing workspace path/read restrictions. Snippets carry relative paths and 1-based line numbers. No external search program or dependency is needed.

Validation: all 40 automated tests passed on Python 3.13.3, including literal matching, scope, excluded paths, binary/encoding handling, snippets, candidate/result limits, and graph feedback. A live local qwen2.5-coder:7b run called search and correctly reported orders.py at line 1 in a disposable project.

## Version 0.3: offline reference RAG

Added local BM25 passage retrieval with citations, automatic task context, and a retrieve graph node. Ships a project-authored LangGraph guide and accepts user references under workspace/knowledge or an explicit --knowledge-base directory. No extra model, network client, persistent store, or vector database is added. Supported reference formats are Markdown, UTF-8 text, and reStructuredText. See KNOWLEDGE_BASE.md for bounds and setup.

Validation: all 52 tests passed on Python 3.13.3. Coverage includes BM25 ranking, line citations, local document filtering and bounds, index refresh on a new task, initial retrieval context, explicit retrieve routing, and no network connections during reference loading/retrieval. The built wheel includes the bundled guide. A real local qwen2.5-coder:7b run retrieved a user policy through the tool, and a final-answer smoke test displayed the policy with the correct user/team.md:1-1 citation. Final answers now display content as well as the model's short status message.

## Version 0.3.1: automatic repository RAG

The CLI now includes the selected workspace's supported source code, documentation,
and configuration/build files in the local BM25 knowledge base. Citations use repo/
paths. User reference documents and the bundled guide remain available, with
physical-path deduplication. Startup reports source counts and index limits.
Approved agent writes rebuild the local index before the next retrieval. The same
bounded limits apply across all roots. Git-ignore support was pending in v0.3.1
and shipped in v0.6 below.

Validation: all 60 automated tests passed on Python 3.13.3, including repository
language coverage, source citations, default CLI indexing, deduplication, filtering,
refresh after approved writes, and no network connections during indexing. A live
local 7b CLI run answered shipping_total(10) as 15 and cited repo/orders.py:1-2.
An earlier live run repeated actions until its limit; the successful run also
proposed an unnecessary preview edit before finishing. Prompt guidance now tells
the model to finish questions from available context and reserve writes for change
requests. These smoke tests do not establish general model reliability.

## Version 0.4: dependency-aware Python retrieval

Added static AST-based local import resolution over already indexed repository
Python files. Retrieval expands keyword matches with up to four linked dependency
passages across two hops, selecting named-definition passages when available.
Imports are not executed. Cycles are deduplicated and the existing shared 16 KB
context budget remains. Syntax errors and expansion limits are reported in result
metadata. Other languages and dynamic/runtime dependency analysis remain out of
scope for this version.

Validation: all 73 automated tests passed on Python 3.13.3. New coverage includes
absolute/relative imports, aliases, src layouts, package initializers, definition
passage selection, cycles, two-hop and context limits, syntax failures, refresh,
and excluded targets. Static analysis never imports project modules or opens
network connections. A live local qwen2.5-coder:7b test correctly answered that
checkout_total(10) returns 17 using the imported fee function, and cited
repo/pricing.py:1-2.

## Version 0.5: complete symbols, context allocation, and JS/TS imports

Added local Tree-sitter JS/TS parsing for relative ES imports/re-exports and literal
require calls, source-extension/index resolution, and named/default definition
ranges. Python AST ranges include decorators. Retrieval prefers complete symbols,
marks partial fallbacks, considers file diversity, reserves dependency space,
continues after oversized candidates, and exposes a configurable 2–16 KB byte
budget with actual-size metadata. Multiple imported definitions can be included.
The existing two-hop/four-dependency limits and local-only 7b model remain.

Validation: all 90 tests passed on Python 3.13.3, covering JS/TS imports, re-exports,
JSX/TSX, cycles, parser failures, complete decorated functions, long imported
functions, multiple imported symbols, budget limits/fallbacks, diversity, and CLI
configuration. Editable installation and pip check passed. Local-only parser
versions tested: tree-sitter 0.25.2, JavaScript grammar 0.25.0, TypeScript grammar
0.23.2. A two-file TypeScript retrieval included the full imported function across
40+ lines within a 6000-byte budget. The live local 7b model eventually answered
17, but first proposed an unnecessary edit in preview mode and misstated a loop
count in its explanation. No files were written. Retrieval validation does not
establish reliable action selection or fully accurate generated explanations.

## Version 0.6: consistent exclusions

Added root/nested Git-style .gitignore and .agentignore matching through local
pathspec, plus repeatable deny-only --exclude patterns. Workspace listing, search,
direct file access, all RAG roots, and dependency context share filtering. An
in-workspace reference root cannot bypass its parent repository's rules. Negations
work within each layer; ignored parents, Git ignores, and built-in restrictions
cannot be bypassed by higher-layer inclusions. Global/ancestor Git settings are
outside this version's root-scoped policy. See EXCLUSIONS.md.

Validation: all 110 tests passed on Python 3.13.3 with pathspec 0.12.1, including
nested negation/parent pruning, CLI patterns, shared listing/search/index/direct
access behavior, ignored dependency targets, reference-root bypass prevention,
refresh, configuration error handling, and no runtime Git/network calls. An
optional fixture compared representative pattern results against local Git's
check-ignore successfully. Editable package installation passed.

## Version 0.7: isolated unittest and pytest

Replaced direct host test execution with mandatory Docker Linux-container isolation.
Local image inspection pins the image ID; no pulls, builds, or dependency installs
occur during runs. A filtered bounded snapshot is mounted read-only, networking is
disabled, the root filesystem is read-only, and tests run as an unprivileged user
with dropped capabilities and resource limits. Temporary writes are isolated.
Clean Docker config avoids inherited proxy/context settings. Tests use a fixed
unittest or pytest command selected by the user, with approval, output/time bounds,
and force-removal of the whole container. Missing isolation fails closed. CI adds
real container integration coverage; setup and limits are in ISOLATED_TESTS.md.

Validation: all 122 tests passed with real container integration enabled on Python
3.13.3 / Docker Desktop 29.8.2. Five integration cases exercised unittest and pytest
pass/fail results, dropped capabilities/no-new-privileges, read-only root/project
mounts, only loopback active, blocked outbound connection, inaccessible host
canary/excluded files/socket/environment secret, permitted temporary writes,
timeout/output limits, and removal of containers with detached children. No
sandbox containers remained after validation. The local image was prepared with
pytest 9.2.0. Package installation, dependency checks, compilation, and whitespace
checks passed. CI adds a separate Linux Docker integration job; its result is
separate from this local validation.

## Version 0.8: reliable edits and evidence from real runs

Reading full transcripts of the local model changed the priorities. The model usually wrote a correct change but not in the requested SEARCH/REPLACE format, so it failed the parser repeatedly. Added:

- Edits that are simply the new version of some lines, merged into the file with checks for ambiguous, unrelated, misplaced, or destructive changes.
- Whole-file write guards, atomic writes, and byte-exact line endings.
- Read-only question tasks, and symbol-based citations for every question.
- Automatic search for code names in the task, with enclosing functions in results.
- File previews showing the start, the end, and the lines a task names.
- Word stems, camelCase splitting, and constant values in retrieval.
- Loop detection across a whole task, a length limit on the `message` field, and a reminder for change tasks that end without a change.
- Warnings when another Ollama client forces model reloads.
- Eval reports with run logs, raw replies, changed files, and reload counts.

Validation on November 7, 2026, on an Apple M4 with 24 GB, using `qwen2.5-coder:7b` with no other Ollama client running:

| Task set | Before (Nov 1) | v0.8 |
|---|---|---|
| `evals/tasks.json`, the set tuned against | 7/12, mean 5.58 steps | 10/12, mean 1.83 steps |
| `evals/bookings_tasks.json`, tasks 1-8, seen during tuning | not run | 7/8 |
| `evals/bookings_tasks.json`, tasks 9-14, seen during tuning | not run | 5/6 |
| `evals/bookings_tasks.json`, tasks 15-20, written after the last code change and never tuned against | not run | 6/6 |

Tasks 1-8 scored 4/8 on their first run, and tasks 9-14 scored 4/6, before later fixes. Those first scores are the fairer measure of how the earlier code handled tasks it had not seen. The last six tasks are a small sample on a small repository and include no multi-file change, so 6/6 does not establish general reliability.

Timing: the 12 main tasks took 513 seconds. The Nov 1 run took 3,013 seconds, but it is not known whether another Ollama client was active then, so that gap cannot all be credited to these changes. With a competing client, steps measured 35-113 seconds of model loading each.

Remaining failures in the final run: `fix-low-stock` (see Next), `slugify-docstring` and `f-comment` (a correct line in the wrong place, reported as success, so these answer checks do not catch it), and `h-holiday-date` (the placement check refused the change correctly, but the model resent it unchanged).

## Version 0.9: read-only code understanding, browser chat, and model choice

CodeHaven is now a read-only assistant for understanding code. The `edit` and `write` actions, `--apply`,
and all code that wrote files were removed, not switched off, so no path in the program can change a
file. The removed editing work remains in Git history up to commit 40cf5c8 if a later version brings
editing back. Change requests are answered with an explanation: what to change, where (with cited lines),
and the new code. `run_tests` stays, because tests run in Docker against a read-only snapshot.

Also added:

- `codehaven --gui`, a local browser page to pick a repository, index it, and chat about it, with
  clickable citations, an offline logo and slogan, and a model picker.
- Any installed local chat model through `--model` or the page. `qwen2.5-coder:7b` stays the default and
  the tested model. Cloud names, remote-backed models, and embedding-only models are refused.
- Points that name whole files cite those files, even when the model pairs a file name with the wrong
  folder, and names that are not definitions, such as table rows, match the line that mentions them.

Evals: the change tasks became questions that check the same understanding (for example, the
`low_stock` bug and its fix, or the price of `SKU-0042`), five redundant ones were dropped, and the
harness now fails any task that leaves a file changed. Earlier results in this file are for those
earlier task sets and are not comparable with runs on the new ones.
