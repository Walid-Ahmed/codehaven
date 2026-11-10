"""Local evaluation harness: run fixed tasks on a sample repository and score the local model.

Usage: python -m offline_agent.evaluation [--only ID] [--output PATH]

Each task runs on a fresh temporary copy of the sample repository, with test runs declined.
CodeHaven is read-only, so a task that leaves any file changed fails. The model is reached only through loopback Ollama,
so evaluation is as offline as normal use. Results record pass/fail, exit code, steps, tokens,
time, reloads, changed files, and each task's run log, so a failure can be explained and
prompt or tool changes can be compared run to run.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import Session
from .knowledge import KnowledgeBase
from .model import MODEL, Ollama
from .workspace import Workspace

EVALS = Path(__file__).resolve().parents[1] / "evals"
RETRIEVAL_BUDGET = 4000  # The CLI default, so evaluation measures what users actually run.
RELOAD_SECONDS = 5
RAW_REPLY_CHARS = 4000  # Raw replies show what a rejected action actually said.


class Metered:
    """Wrap a model and add up the calls, tokens, and model time Ollama reports."""

    def __init__(self, model):
        self.model = model
        self.prompt_budget = getattr(model, "prompt_budget", 12_000)
        self.last_metrics = {}
        self.calls = self.prompt_tokens = self.generated_tokens = self.reloads = 0
        self.model_seconds = 0.0
        self.replies = []

    def chat(self, messages, schema):
        self.calls += 1
        try:
            return self.model.chat(messages, schema)
        finally:
            self.replies.append(str(getattr(self.model, "last_raw", ""))[:RAW_REPLY_CHARS])
            metrics = getattr(self.model, "last_metrics", None)
            self.last_metrics = metrics if isinstance(metrics, dict) else {}
            self.prompt_tokens += self.last_metrics.get("prompt_tokens") or 0
            self.generated_tokens += self.last_metrics.get("generated_tokens") or 0
            self.model_seconds += self.last_metrics.get("total_seconds") or 0
            if self.calls > 1 and (self.last_metrics.get("load_seconds") or 0) >= RELOAD_SECONDS:
                self.reloads += 1


def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must hold a non-empty JSON list")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not isinstance(task.get("task"), str):
            raise ValueError(f"Each task needs string id and task fields: {task!r:.120}")
        if task["id"] in seen:
            raise ValueError(f"Duplicate task id: {task['id']}")
        seen.add(task["id"])
        followups = task.get("followups", [])
        if not isinstance(followups, list):
            raise ValueError(f"Task {task['id']}: followups must be a list")
        for turn in [task] + followups:
            if not isinstance(turn, dict) or not isinstance(turn.get("task"), str):
                raise ValueError(f"Task {task['id']}: each follow-up needs a string task")
            checks = turn.get("checks")
            if not isinstance(checks, dict) or not checks.get("answer") or set(checks) != {"answer"}:
                raise ValueError(f"Task {task['id']} needs answer checks, and only those: CodeHaven never changes files")
            for pattern in checks["answer"]:
                re.compile(pattern)
    return tasks


def score(task, answer):
    """Return the answer regexes (case-insensitive) that the answer fails."""
    return [f"answer lacks /{pattern}/" for pattern in task["checks"]["answer"] if not re.search(pattern, answer, re.I | re.S)]


def changed_files(before, after):
    """Relative paths whose content differs between two directory trees."""
    def tree(root):
        return {path.relative_to(root).as_posix(): path.read_bytes() for path in Path(root).rglob("*")
                if path.is_file() and "__pycache__" not in path.parts}
    old, new = tree(before), tree(after)
    return sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))


def evaluate_task(task, model, repo):
    """Run one task, and its follow-up questions in the same session, on a fresh copy of the repository."""
    with tempfile.TemporaryDirectory(prefix="codehaven-eval-") as temporary:
        root = Path(temporary) / "repo"
        shutil.copytree(repo, root)
        workspace = Workspace(root)
        metered, log = Metered(model), []
        started = time.monotonic()
        session = Session(metered, workspace, knowledge=KnowledgeBase(None, workspace=workspace, context_budget=RETRIEVAL_BUDGET),
                          max_steps=task.get("max_steps", 12), confirm_tests=lambda: False, output=log.append)
        turns = [task] + task.get("followups", [])
        answers, failures, exit_code, steps = [], [], 0, 0
        for number, turn in enumerate(turns, 1):
            label = f"turn {number}: " if len(turns) > 1 else ""
            answer = ""
            try:
                code = session.ask(turn["task"])
                state = session.last_state
                steps += state.get("steps", 0)
                exit_code = max(exit_code, code)
                reply = state.get("reply") or {}
                if reply.get("action") == "finish":
                    answer = reply.get("content") or reply.get("message") or ""
            except ValueError as exc:
                exit_code, steps = 1, max(steps, metered.calls)
                failures.append(f"{label}error: {exc}")
            failures += [label + failure for failure in score(turn, answer)]
            answers.append(answer if len(turns) == 1 else f"Q{number}: {turn['task']}\nA{number}: {answer}")
        changed = changed_files(repo, root)
        failures += [f"{name} changed, but CodeHaven must never change files" for name in changed]
    return {"id": task["id"], "passed": not failures, "exit_code": exit_code, "steps": steps, "changed": changed,
            "turns": len(turns), "reloads": metered.reloads, "log": [str(line) for line in log], "replies": metered.replies,
            "prompt_tokens": metered.prompt_tokens, "generated_tokens": metered.generated_tokens,
            "model_seconds": round(metered.model_seconds, 2), "seconds": round(time.monotonic() - started, 2),
            "failures": failures, "answer": "\n\n".join(answers)[:4000]}


def summarize(results):
    count = len(results)
    return {"tasks": count, "passed": sum(r["passed"] for r in results),
            "verified": sum(r["exit_code"] == 0 for r in results), "reloads": sum(r.get("reloads", 0) for r in results),
            "mean_steps": round(sum(r["steps"] for r in results) / count, 2) if count else 0,
            "prompt_tokens": sum(r["prompt_tokens"] for r in results),
            "generated_tokens": sum(r["generated_tokens"] for r in results),
            "seconds": round(sum(r["seconds"] for r in results), 1)}


def table(results):
    width = max([len("TASK")] + [len(r["id"]) for r in results])
    lines = [f"{'TASK':<{width}}  PASS  EXIT  STEPS  PROMPT TOK  GEN TOK  SECONDS  NOTES"]
    for r in results:
        lines.append(f"{r['id']:<{width}}  {'yes' if r['passed'] else 'NO':<4}  {r['exit_code']:<4}  {r['steps']:<5}  "
                     f"{r['prompt_tokens']:>10,}  {r['generated_tokens']:>7,}  {r['seconds']:>7.1f}  "
                     + "; ".join(r["failures"])[:120])
    return "\n".join(lines)


def evaluate(tasks, model, repo, output=print):
    results = []
    for number, task in enumerate(tasks, 1):
        output(f"[{number}/{len(tasks)}] {task['id']}: {task['task']}")
        results.append(evaluate_task(task, model, repo))
        result = results[-1]
        output(f"    {'PASS' if result['passed'] else 'FAIL'} in {result['steps']} steps, {result['seconds']:.1f}s"
               + (f" ({'; '.join(result['failures'])})" if result["failures"] else ""))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a local model on fixed CodeHaven tasks (offline)")
    parser.add_argument("--tasks", default=str(EVALS / "tasks.json"))
    parser.add_argument("--repo", default=str(EVALS / "sample_repo"))
    parser.add_argument("--only", action="append", default=[], metavar="ID", help="Run only this task id; repeatable")
    parser.add_argument("--output", default=str(EVALS / "results" / "latest.json"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=MODEL, help="Installed local Ollama model to score (default: the tested one)")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--prompt-budget", type=int, default=12000)
    args = parser.parse_args(argv)
    try:
        tasks = load_tasks(args.tasks)
        if args.only:
            unknown = set(args.only) - {task["id"] for task in tasks}
            if unknown:
                parser.error("unknown task id: " + ", ".join(sorted(unknown)))
            tasks = [task for task in tasks if task["id"] in args.only]
        if not Path(args.repo).is_dir():
            raise ValueError(f"Sample repository not found: {args.repo}")
        model = Ollama(model=args.model, endpoint=args.endpoint, max_output_tokens=args.max_output_tokens, prompt_budget=args.prompt_budget)
        model.check_model()
        others = model.other_loaded_models()
        if others:
            print("Warning: Ollama also has " + ", ".join(others) + " loaded, so timings may not be representative.")
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = evaluate(tasks, model, args.repo, output=lambda line: print(line, flush=True))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    summary = summarize(results)
    print("\n" + table(results))
    print(f"\nPassed {summary['passed']}/{summary['tasks']}; {summary['verified']} also passed CodeHaven's own answer checks; "
          f"mean steps {summary['mean_steps']}; "
          f"tokens {summary['prompt_tokens']:,} prompt / {summary['generated_tokens']:,} generated; {summary['seconds']}s")
    report = {"model": args.model, "started": started, "tasks_file": args.tasks,
              "settings": {"max_output_tokens": args.max_output_tokens, "prompt_budget": args.prompt_budget},
              "summary": summary, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results written to {output}")
    if summary["reloads"]:
        print(f"Warning: the model was reloaded {summary['reloads']} times between steps, so these timings are not "
              "representative. Something else was competing for Ollama's memory.")
    return 0 if summary["passed"] == summary["tasks"] else 1


if __name__ == "__main__":
    sys.exit(main())
