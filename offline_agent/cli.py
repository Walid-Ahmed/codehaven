import argparse
from functools import partial
import sys
from pathlib import Path
from .knowledge import KnowledgeBase
from .agent import Session, run
from .model import MODEL, Ollama
from .workspace import Workspace
from .test_runner import DEFAULT_IMAGE

EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def write_stream(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def interactive(session, first_task=None, prompt=input):
    """Read tasks until exit. One failed or interrupted task does not end the session."""
    print("Interactive session. Type a task, /reset to forget earlier answers, or exit to quit.", flush=True)
    task = first_task
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
        elif task:
            try:
                session.ask(task)
            except (ValueError, OSError) as exc:
                print(f"Error: {exc}", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                print("\nStopped this task.", file=sys.stderr, flush=True)
        task = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"CodeHaven — Your code. Your machine. No API fees. Powered by local {MODEL}")
    parser.add_argument("task", nargs="?", help="Coding task in plain English")
    parser.add_argument("--workspace", default=".")
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
    parser.add_argument("--test-docker-socket", help="Absolute local Unix Docker socket; remote daemons are unsupported")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--apply", action="store_true", help="Allow writing after per-file confirmation")
    parser.add_argument("--max-steps", type=int, default=12, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--doctor", action="store_true", help="Check local server and installed model")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Ask several tasks in one session; follow-ups see earlier answers")
    parser.add_argument("--no-stream", action="store_true", help="Print the final answer only once it is complete")
    args = parser.parse_args(argv)
    if not 2000 <= args.retrieval_budget <= 16000:
        parser.error("--retrieval-budget must be between 2000 and 16000 bytes")
    if not args.doctor and not args.interactive and not args.task:
        parser.error("a task is required unless --doctor or --interactive is used")
    try:
        model = Ollama(endpoint=args.endpoint, max_output_tokens=args.max_output_tokens,
                       prompt_budget=args.prompt_budget)
        model.check_model()
        others = model.other_loaded_models()
        if others:
            print("Warning: Ollama also has " + ", ".join(others) + " loaded. If both models do not fit in memory, Ollama "
                  "reloads them in turn, which can add a minute or more to every step. Check with `ollama ps`.")
        if args.doctor:
            print(f"Local Ollama is ready: {MODEL}")
            return 0
        workspace = Workspace(args.workspace, excludes=args.exclude)
        references = Path(args.knowledge_base).expanduser() if args.knowledge_base else workspace.root / "knowledge"
        knowledge = KnowledgeBase(references if args.knowledge_base or references.exists() else None,
                                  workspace=workspace, context_budget=args.retrieval_budget)
        print(f"Knowledge base: {knowledge.documents_by_source['repo']} repository files, "
              f"{knowledge.documents_by_source['user']} user references, {len(knowledge.entries)} chunks. "
              "Add your own .md, .txt, or .rst references to " + str(references))
        if knowledge.limited:
            print("Index limits reached; retrieval covers only part of the available files. Use search/read for more.")
        if knowledge.excluded_reference_roots:
            print("Reference directory excluded by workspace rules; it was not indexed. "
                  "Use a permitted directory or an external reference library.")
        print(f"Workspace: {workspace.root}\nFile edits: {'confirm each write' if args.apply else 'preview only'}")
        options = dict(apply=args.apply, max_steps=args.max_steps, knowledge=knowledge,
                       output=partial(print, flush=True), stream=None if args.no_stream else write_stream,
                       test_options={"runner": args.test_runner, "image": args.test_image,
                                     "docker_socket": args.test_docker_socket},
                       confirm=lambda path: input(f"Write {path}? [y/N] ").strip().lower() == "y",
                       confirm_tests=lambda: input(
                           f"Run {args.test_runner} in Docker with network disabled and project read-only? [y/N] "
                       ).strip().lower() == "y")
        if args.interactive:
            return interactive(Session(model, workspace, **options), args.task)
        return run(model, workspace, args.task, **options)
    except (ValueError, OSError, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
