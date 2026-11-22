"""Download the real open-source repositories the real-repository evals run on. Run once, while online.

Usage: python3 scripts/fetch_eval_repos.py [NAME ...]

This is a developer tool, not part of CodeHaven: the installed package never runs it, and CodeHaven itself stays
offline. It is the only step of the evals that uses the internet. git downloads one pinned release of each
repository in evals/real/repos.json into evals/real/repos/NAME, which Git ignores, and the download is refused
unless it is exactly the commit recorded there, so the questions always meet the code they were written for.
After that the evals run offline, like the others:

    python3 -m offline_agent.evaluation --suite requests
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REAL = Path(__file__).resolve().parents[1] / "evals" / "real"
SECONDS = 300  # A download is stopped after 5 minutes.


def git(*arguments, cwd=None):
    """Run git without prompts, hooks, or LFS downloads, and return what it printed."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    done = subprocess.run(["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.symlinks=false", *arguments],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=SECONDS, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        raise ValueError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "git failed")
    return done.stdout.strip()


def fetch(entry, folder):
    """Download one pinned release into folder/NAME. Returns a line saying what happened."""
    name, target = entry["name"], folder / entry["name"]
    marker = target / ".codehaven-eval-commit"  # Hidden, so CodeHaven never indexes it.
    if marker.is_file() and marker.read_text().strip() == entry["commit"]:
        return f"{name}: already here at {entry['tag']}."
    folder.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=folder))
    try:
        git("clone", "--depth", "1", "--single-branch", "--no-tags", "--template=", "--quiet",
            "--branch", entry["tag"], "--", entry["url"], str(partial))
        commit = git("rev-parse", "HEAD", cwd=partial)
        if commit != entry["commit"]:
            raise ValueError(f"{entry['tag']} is commit {commit}, not the recorded {entry['commit'] or '(none)'}")
        shutil.rmtree(partial / ".git")  # Only the files are needed, and CodeHaven must not see a second repository.
        (partial / ".codehaven-eval-commit").write_text(commit + "\n")
        if target.exists():
            shutil.rmtree(target)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return f"{name}: downloaded {entry['tag']} ({commit[:10]})."


def main(argv=None):
    """Download every repository in the list, or only the ones named. Returns 0 when all are in place."""
    names = list(sys.argv[1:] if argv is None else argv)
    entries = json.loads((REAL / "repos.json").read_text(encoding="utf-8"))
    unknown = set(names) - {entry["name"] for entry in entries}
    if unknown:
        print("Unknown repository: " + ", ".join(sorted(unknown)), file=sys.stderr)
        return 1
    failed = 0
    for entry in entries:
        if names and entry["name"] not in names:
            continue
        try:
            print(fetch(entry, REAL / "repos"), flush=True)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            failed += 1
            print(f"{entry['name']}: not downloaded: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
