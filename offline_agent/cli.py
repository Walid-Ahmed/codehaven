"""The codehaven command: one task, an interactive session, or the browser GUI."""
import argparse
import contextlib
import sys
import time
from functools import partial
from pathlib import Path

from . import coder, doctor, edits, errors, export, index_cache, ocr, slash
from .agent import Session, run
from .chats import Chats
from .knowledge import KnowledgeBase
from .memory import Memory
from .model import MODEL, Ollama
from .semantic import Semantic, choose_model
from .test_runner import DEFAULT_IMAGE
from .workspace import Workspace, combined

SEMANTIC_WAIT = 60  # Seconds a command waits for meaning search before answering with what is ready.
EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def write_stream(text):
    """Print a piece of the streamed answer at once, without waiting for the end of the line."""
    sys.stdout.write(text)
    sys.stdout.flush()


def interactive(session, first_task=None, prompt=input):
    """Read tasks until exit. One failed or interrupted task does not end the session."""
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401  (gives input() line editing and arrow-key history, where Python has it)
    print("Interactive session. Type a task, /reset to forget earlier answers, /export [file] to save the answers as "
          "Markdown, or exit to quit.", flush=True)
    task = first_task
    answered = []  # (question, answer, issues) of this session, for /export.
    while True:
        if task is None:
            try:
                task = prompt("\ncodehaven> ").strip()
            except EOFError:
                print()
                return 0
        if task.lower() in EXIT_WORDS:
            return 0
        if task == "/reset":
            session.reset()
            print("Session history cleared.", flush=True)
        elif task == "/undo":
            editor = getattr(session, "options", {}).get("editor")
            if editor is None:
                print("Nothing to undo: changing code is off. Start with --allow-edits to turn it on.", flush=True)
            else:
                try:
                    print(f"Undid: {edits.undo(session.workspace, editor.journal)}", flush=True)
                except (edits.EditError, OSError) as exc:
                    print(f"Error: {exc}", file=sys.stderr, flush=True)
        elif task == "/export" or task.startswith("/export "):
            if not answered:
                print("Nothing to export yet: ask a question first.", flush=True)
            else:
                try:
                    written = export.write(task[len("/export"):].strip() or ".",
                                           export.markdown(answered, str(getattr(session.workspace, "root", "")),
                                                           getattr(session.model, "model", "")))
                    print(f"Saved {len(answered)} answer{'s' if len(answered) != 1 else ''} to {written}.", flush=True)
                except FileExistsError:
                    print("That file already exists; CodeHaven does not overwrite it. Give another name.", file=sys.stderr, flush=True)
                except OSError as exc:
                    print(f"Error: could not save the answers ({exc}).", file=sys.stderr, flush=True)
        elif task.startswith("/") and task.split()[0] not in ("/reset", "/undo", "/export"):
            try:
                expanded = slash.expand(task)
            except ValueError as exc:
                print(exc, file=sys.stderr, flush=True)
            else:
                if expanded is None:
                    print(slash.help_text(["/reset: forget earlier answers", "/export [file]: save the answers as Markdown",
                                           "/undo: undo the last change (with --allow-edits)", "exit: quit"]), flush=True)
                else:
                    task = expanded
                    continue  # Asked below, as if it had been typed in full.
        elif task:
            try:
                session.ask(task)
                state = getattr(session, "last_state", None) or {}
                if state.get("final_text"):
                    answered.append((task, state["final_text"], state.get("issues", [])))
            except (ValueError, OSError) as exc:
                print(f"Error: {exc}", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                print("\nStopped this task.", file=sys.stderr, flush=True)
            except Exception as exc:  # An unexpected problem ends this task, not the session.
                errors.report(exc)
                print(f"Error: {errors.friendly(exc)}", file=sys.stderr, flush=True)
        task = None


def read_picture(picture, out=None):
    """Write the code in a picture to a new file beside it. Needs Tesseract, not the model."""
    try:
        source = Path(picture).expanduser()
        if source.stat().st_size > ocr.IMAGE_BYTES:
            raise ValueError(f"Only pictures up to {ocr.IMAGE_BYTES // 1_000_000} MB can be read.")
        target = Path(out).expanduser() if out else None
        found = ocr.read_code(source.name, source.read_bytes(), target.name if target else None)
        target = target or source.with_name(found["name"])
        try:
            with open(target, "x", encoding="utf-8") as handle:  # "x" refuses to replace a file that is there.
                handle.write(found["text"])
        except FileExistsError:
            raise ValueError(f"{target} already exists, and CodeHaven never changes your files. "
                             "Choose a new name with --out.") from None
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Read {found['lines']} lines from {source.name} into {target}.\n"
          "Reading a picture can get characters wrong, so check the text before you run it.")
    if found["check"]:
        print("Compare with the picture: line " + ", ".join(map(str, found["check"])) + ".")
    if found["syntax"]:
        print(f"Not valid Python yet ({found['syntax']}).")
    elif target.suffix == ".py":
        print("It is valid Python.")
    return 0


def main(argv=None):
    """Parse the command line and run the chosen mode. Returns the exit code."""
    parser = argparse.ArgumentParser(description=f"CodeHaven — Your code. Your machine. No API fees. A read-only assistant for "
                                                 f"understanding code with a local model ({MODEL} by default). "
                                                 "It never changes your files.")
    parser.add_argument("task", nargs="?", help="Question about the code, in plain English or Arabic")
    parser.add_argument("--workspace", help="Project folder; defaults to the current folder (with --gui, none)")
    parser.add_argument("--also", action="append", default=[], metavar="FOLDER",
                        help="Another repository to open with --workspace, so questions are answered across both; repeatable")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Additional Git-style exclusion; repeat for multiple patterns")
    parser.add_argument("--knowledge-base", help="Local reference directory; defaults to workspace/knowledge")
    parser.add_argument("--retrieval-budget", type=int, default=4000, metavar="BYTES",
                        help="Retrieval JSON budget in UTF-8 bytes (2000..16000; default 4000)")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Maximum response tokens per action (256..8192; default 1024)")
    parser.add_argument("--prompt-budget", type=int, default=12000, metavar="BYTES",
                        help="Conversation content budget (8000..64000 UTF-8 bytes; default 12000)")
    parser.add_argument("--test-runner", choices=("unittest", "pytest"), default="unittest")
    parser.add_argument("--test-image", default=DEFAULT_IMAGE, help="Prepared local Docker image for isolated tests")
    parser.add_argument("--allow-python", action="store_true",
                        help="Let the agent ask to run a short Python snippet, shown to you for approval first, in the same "
                             "Docker sandbox as the tests (no network, read-only copy of the project)")
    parser.add_argument("--test-docker-socket", help="Absolute local Unix Docker socket; remote daemons are unsupported")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=MODEL, help=f"Installed local Ollama model (default {MODEL}, the tested one)")
    parser.add_argument("--max-steps", type=int, default=12, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--doctor", action="store_true",
                        help="Check the local server, the model, and the optional tools, and say how to get what is missing")
    parser.add_argument("--setup", action="store_true",
                        help="Guided first run: check Ollama, suggest a model for this computer's memory, and download it "
                             "with ollama pull only if you say yes (the only step that uses the internet)")
    parser.add_argument("--start-ollama", action="store_true",
                        help="If no Ollama server is running, start one for this run on 127.0.0.1 with cloud features "
                             "off, and stop it on exit")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Ask several tasks in one session; follow-ups see earlier answers")
    parser.add_argument("--no-retry", action="store_true",
                        help="Do not ask the model to correct an answer that fails its checks (one extra step, on by default)")
    parser.add_argument("--allow-edits", action="store_true",
                        help="Let CodeHaven change code when you ask it to (fix, write tests, add docstrings, rename, write "
                             "docs). Each change is shown as a diff and applied only if you answer y; --undo puts it back")
    parser.add_argument("--undo", action="store_true", help="Undo the last change CodeHaven made in this workspace, then exit")
    parser.add_argument("--export", metavar="FILE",
                        help="Also save the question and its answer, with citations, as Markdown to FILE (a new file; "
                             "an existing one is never replaced). In --interactive, type /export [FILE] instead")
    parser.add_argument("--no-stream", action="store_true", help="Print the final answer only once it is complete")
    parser.add_argument("--gui", action="store_true", help="Open a local page in your browser to pick a repository and chat")
    parser.add_argument("--port", type=int, default=0, help="Port for --gui on 127.0.0.1 (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="With --gui, print the link instead of opening it")
    parser.add_argument("--allow-clone", action="store_true",
                        help="With --gui, allow downloading GitHub repositories you ask for (the only network use; "
                             "off by default, nothing is uploaded)")
    parser.add_argument("--clone-dir", metavar="FOLDER",
                        help="With --allow-clone, where downloaded repositories are kept (default ~/CodeHaven/repos)")
    parser.add_argument("--remember", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep checked answers about this repository on this computer and recall them in later "
                             "sessions while the files they cite are unchanged (on by default; --no-remember saves "
                             "nothing to disk)")
    parser.add_argument("--memory-dir", metavar="FOLDER",
                        help="Where saved answers are kept (default ~/CodeHaven/memory)")
    parser.add_argument("--chats-dir", metavar="FOLDER",
                        help="Where the browser page keeps saved chats (default ~/CodeHaven/chats; nothing is kept "
                             "with --no-remember)")
    parser.add_argument("--forget", action="store_true",
                        help="Delete the answers saved for this workspace, then exit")
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=True,
                        help="Also search by meaning with a local Ollama embedding model, when one is installed "
                             "(on by default; --no-semantic uses key words only)")
    parser.add_argument("--embed-model", metavar="NAME",
                        help="Installed local embedding model for --semantic (default: the first found of "
                             "nomic-embed-text, embeddinggemma, mxbai-embed-large, ...)")
    parser.add_argument("--index-dir", metavar="FOLDER",
                        help="Where the keyword index and meaning-search vectors are kept (default ~/CodeHaven/index; nothing is kept "
                             "with --no-remember)")
    parser.add_argument("--ocr", metavar="PICTURE",
                        help="Read the code in a PNG or JPEG picture, such as a screenshot, into a new file beside it, "
                             "then exit. Offline; needs Tesseract installed on this computer")
    parser.add_argument("--out", metavar="FILE", help="With --ocr, the file to write (default: the picture's name with .py)")
    args = parser.parse_args(argv)
    # --ocr and --forget do not use the model, so they are handled before Ollama is contacted.
    if args.ocr:
        return read_picture(args.ocr, args.out)
    if args.undo:
        try:
            workspace = Workspace(args.workspace or ".", excludes=args.exclude)
            print(f"Undid: {edits.undo(workspace, edits.journal_path(workspace))}")
            return 0
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if args.forget:
        try:
            memory = Memory(Workspace(args.workspace or ".", excludes=args.exclude), args.memory_dir)
            count = len(memory.notes)
            memory.forget()
            chats = Chats(memory.workspace, args.chats_dir)
            saved_chats = len(chats.chats)
            chats.forget()
            index_cache.forget(index_cache.path(memory.workspace, args.index_dir))
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Forgot {count} saved answer{'s' if count != 1 else ''} and {saved_chats} saved "
              f"chat{'s' if saved_chats != 1 else ''} for {memory.workspace.root}.")
        return 0
    if not 2000 <= args.retrieval_budget <= 16000:
        parser.error("--retrieval-budget must be between 2000 and 16000 bytes")
    if args.setup:
        try:
            return doctor.setup(Ollama(model=args.model, endpoint=args.endpoint), output=partial(print, flush=True))
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if not args.doctor and not args.interactive and not args.gui and not args.task:  # --undo returned above.
        # Typed at a terminal with nothing to ask yet, the likely wish is to start asking.
        if not sys.stdin.isatty():
            parser.error("a task is required unless --doctor, --interactive, --gui, or --forget is used")
        args.interactive = True
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    server = model = None
    try:
        model = Ollama(model=args.model, endpoint=args.endpoint, max_output_tokens=args.max_output_tokens,
                       prompt_budget=args.prompt_budget)
        if args.start_ollama:
            server = doctor.start_ollama(model, partial(print, flush=True))
        try:
            model.check_model()
        except ValueError as exc:
            # The GUI can start with another installed model, which the page lets the user change.
            try:
                available = [m["name"] for m in model.local_models()] if args.gui else []
            except ValueError:
                available = []
            if not available:
                # Say what is missing and the command that fixes it, not only that the request failed.
                print(f"Error: {exc}", file=sys.stderr)
                for line in doctor.missing_server(model):
                    print("  " + line, file=sys.stderr)
                return 1
            model.use(available[0])
            print(f"{args.model} is not available; starting with {available[0]}. Choose another model in the page.")
        embed_model = choose_model(model, args.embed_model) if args.semantic else None
        if embed_model and hasattr(model, "companions"):
            model.companions.add(embed_model)  # Ours, so not reported as another program's model.
        others = model.other_loaded_models()
        if others:
            print("Warning: Ollama also has " + ", ".join(others) + " loaded. If both models do not fit in memory, Ollama "
                  "reloads them in turn, which can add a minute or more to every step. Check with `ollama ps`.")
        if args.doctor:
            print(f"Local Ollama is ready: {model.model}")
            print(f"Meaning search: {embed_model}" if embed_model else "Meaning search: off (--no-semantic)" if not args.semantic
                  else "Meaning search: off, no embedding model is installed. To turn it on, run once while online: "
                       "ollama pull nomic-embed-text")
            suggestion, why = doctor.suggested_model()
            print(f"Suggested model for this computer: {suggestion} ({why})." if suggestion != model.model else
                  f"The model in use is the one suggested for this computer ({why}).")
            for line in doctor.extras(args.test_image) + [doctor.default_model_hint(model)]:
                if line:
                    print(line)
            return 0
        if args.gui:
            # Imported only here, so the web server is loaded only when --gui asks for it.
            from .gui import serve
            return serve(model, port=args.port, open_browser=not args.no_browser, repository=args.workspace,
                         max_steps=args.max_steps, retrieval_budget=args.retrieval_budget, excludes=args.exclude,
                         allow_clone=args.allow_clone, clone_folder=args.clone_dir,
                         remember=args.remember, memory_folder=args.memory_dir,
                         embed_model=embed_model, index_folder=args.index_dir, chats_folder=args.chats_dir,
                         allow_edits=args.allow_edits)
        workspace = (combined([args.workspace or ".", *args.also], excludes=args.exclude) if args.also
                     else Workspace(args.workspace or ".", excludes=args.exclude))
        if workspace.scopes:
            print("Opened together: " + ", ".join(workspace.scopes) + f" (paths are shown from {workspace.root}).")
        references = Path(args.knowledge_base).expanduser() if args.knowledge_base else workspace.root / "knowledge"
        semantic = None
        if embed_model:
            semantic = Semantic(model, embed_model, output=partial(print, flush=True),
                                cache=Semantic.cache_path(workspace, embed_model, args.index_dir) if args.remember else None)
        print(f"Indexing {workspace.root} ...", flush=True)
        started = time.monotonic()
        knowledge = KnowledgeBase(references if args.knowledge_base or references.exists() else None,
                                  workspace=workspace, context_budget=args.retrieval_budget, semantic=semantic,
                                  cache=index_cache.path(workspace, args.index_dir) if args.remember else None)
        reused = f", {knowledge.reused} unchanged since last time" if knowledge.reused else ""
        print(f"Indexed {knowledge.documents} files in {time.monotonic() - started:.1f}s{reused}.", flush=True)
        if semantic is not None:
            # A first run embeds every passage; later runs only the changed ones. Never hold a question up for long.
            complete = semantic.wait(SEMANTIC_WAIT, partial(print, flush=True))
            ready, total = semantic.coverage()
            print(f"Meaning search: on with {embed_model}, {ready} of {total} passages ready"
                  + ("." if complete else "; the rest are prepared in the background and saved for next time."))
        print(f"Knowledge base: {knowledge.documents_by_source['repo']} repository files, "
              f"{knowledge.documents_by_source['user']} user references, {len(knowledge.entries)} chunks. "
              "Add your own .md, .txt, or .rst references to " + str(references))
        if knowledge.limited:
            print("Index limits reached; retrieval covers only part of the available files. Use search/read for more.")
        if knowledge.excluded_reference_roots:
            print("Reference directory excluded by workspace rules; it was not indexed. "
                  "Use a permitted directory or an external reference library.")
        print(f"Workspace: {workspace.root}\nRead-only: CodeHaven never changes your files.")
        options = dict(max_steps=args.max_steps, knowledge=knowledge, retry_unverified=not args.no_retry,
                       output=partial(print, flush=True), stream=None if args.no_stream else write_stream,
                       test_options={"runner": args.test_runner, "image": args.test_image,
                                     "docker_socket": args.test_docker_socket},
                       confirm_tests=lambda: input(
                           f"Run {args.test_runner} in Docker with network disabled and project read-only? [y/N] "
                       ).strip().lower() == "y",
                       confirm_code=(lambda _code: input("Run this snippet in Docker with network disabled and the project "
                                                         "read-only? [y/N] ").strip().lower() == "y")
                       if args.allow_python else None)
        if args.allow_edits:
            options["editor"] = coder.Editor(
                confirm=lambda _diff, _summary: input("Apply this change? [y/N] ").strip().lower() == "y",
                journal=edits.journal_path(workspace), test_options=options["test_options"],
                confirm_tests=options["confirm_tests"])
            print("Changing code is on: each change is shown as a diff and applied only if you answer y. "
                  "Undo the last one with /undo or codehaven --undo.")
        memory = Memory(workspace, args.memory_dir) if args.remember else None
        if memory is not None:
            print(f"Long-term memory: {len(memory.notes)} saved answer{'s' if len(memory.notes) != 1 else ''} for this "
                  f"workspace in {memory.folder}. Checked answers are saved there; use --no-remember to save nothing, "
                  "or --forget to delete them.")
        if args.interactive:
            return interactive(Session(model, workspace, memory=memory, **options), args.task)
        if args.export:
            session = Session(model, workspace, memory=memory, **options)
            code = session.ask(args.task)
            state = session.last_state
            try:
                written = export.write(args.export, export.markdown([(args.task, state.get("final_text") or "", state.get("issues", []))],
                                                                    str(workspace.root), model.model))
                print(f"Saved the answer to {written}.")
            except FileExistsError:
                print(f"Error: {args.export} already exists; CodeHaven does not overwrite it.", file=sys.stderr)
                return 1
            return code
        return run(model, workspace, args.task, memory=memory, **options)
    except (ValueError, OSError, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        # The usual exit code for a program stopped by Ctrl+C (128 + SIGINT's number, 2).
        return 130
    except Exception as exc:  # Anything unexpected: a plain sentence, not a traceback.
        errors.report(exc)
        print(f"Error: {errors.friendly(exc)}", file=sys.stderr)
        return 1
    finally:
        doctor.stop_ollama(server)
