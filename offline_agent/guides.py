"""Two whole-project views for a newcomer: an onboarding guide, and Mermaid diagrams drawn from the parsed code.

- Onboarding guide: how to run the project and its tests (read from its own config files), what it depends on,
  where its tests live, and the reading order of the project tour. Only the tour uses the model, for one sentence
  per file; everything else is read from the files and cited.
- Diagrams: the import graph between the project's folders or files, or the calls below a named function, as a
  Mermaid flowchart. Mermaid is plain text that GitHub, GitLab, many editors, and mermaid.live draw as a picture,
  so the diagram can be pasted into a README or a pull request. Nothing uses the model, and nothing is run.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from . import insights, scans, trace

ONBOARDING = re.compile(r"\bonboard(?:ing)?\b|\b(?:i'?m|i\s+am)\s+new\s+(?:to|on)\b|\bnew\s+(?:developer|contributor|team\s+member)\b"
                        r"|\bgetting[- ]started\s+guide\b|\bhow\s+do\s+i\s+get\s+started\b", re.I)
DIAGRAM = re.compile(r"\b(?:mermaid|diagram|flowchart|draw\s+(?:the\s+)?(?:architecture|structure|modules?|imports?|call))\b", re.I)
CALLS = re.compile(r"\bcalls?\b.*?\bof\s+`?([A-Za-z_$][\w$.]*\w)`?|\b(?:call\s+(?:graph|flow|tree)|calls)\s+(?:for|from|of)\s+"
                   r"`?([A-Za-z_$][\w$.]*\w)`?", re.I)
NODES = 24  # The most boxes one diagram draws, so it stays readable.


def requested(task):
    """("onboarding", None), ("diagram", function name or None), or None."""
    if ONBOARDING.search(task):
        return "onboarding", None
    if DIAGRAM.search(task):
        found = CALLS.search(task)
        return "diagram", next((group for group in found.groups() if group), None) if found else None
    return None


def run(kind, name, model, workspace, knowledge, output=print):
    """The answer for one kind: {"text", "evidence"}."""
    if kind == "onboarding":
        return onboarding(model, workspace, knowledge, output)
    return call_diagram(knowledge, name) if name else import_diagram(knowledge)


def node_id(text):
    """A Mermaid node id: letters, digits, and underscores only."""
    return "n_" + re.sub(r"\W", "_", text)


def label(text):
    """A Mermaid node label, quoted so slashes and dots are kept."""
    return '"' + text.replace('"', "'") + '"'


def import_diagram(knowledge):
    """Which files import which: the NODES files with the most import links, so a large project shows its core."""
    edges = getattr(getattr(knowledge, "dependencies", None), "edges", None) or {}
    pairs = [(source, target) for source, targets in edges.items() for target in targets
             if source != target and not insights.is_test(source) and not insights.is_test(target)]
    if not pairs:
        return {"text": "No imports between this project's files were found, so there is no diagram to draw. Imports are "
                        "followed for Python, JavaScript, TypeScript, Go, Java, Rust, C, C++, C#, Ruby, PHP, and Kotlin.",
                "evidence": []}
    degree = Counter(path for pair in pairs for path in pair)
    nodes = {path for path, _ in degree.most_common(NODES)}
    chosen = sorted(pair for pair in set(pairs) if pair[0] in nodes and pair[1] in nodes)
    lines = ["```mermaid", "flowchart LR"]
    lines += [f"  {node_id(node)}[{label(node)}]" for node in sorted(nodes)]
    lines += [f"  {node_id(source)} --> {node_id(target)}" for source, target in chosen]
    lines.append("```")
    total = len(degree)
    note = (f"Which files import which, from the parsed code: the {len(nodes)} most connected of {total} files, tests left out. "
            "Paste it into GitHub, an editor with Mermaid, or mermaid.live to see it drawn.")
    return {"text": note + "\n\n" + "\n".join(lines), "evidence": []}


def call_diagram(knowledge, name):
    """The calls below one function, as the trace finds them, drawn as a flowchart."""
    every = trace.functions(knowledge) if knowledge is not None else []
    places = trace.locate(name, every)
    if not places:
        return {"text": f"No function or method named {name} was found in the parsed code.", "evidence": []}
    edges = getattr(getattr(knowledge, "dependencies", None), "edges", None) or {}
    start = places[0]
    seen, queue, arrows = {(start[0], start[1]["name"])}, [(start, 0)], []
    for (path, span), depth in queue:
        if depth >= trace.DEPTH or len(seen) >= NODES:
            continue
        lines = knowledge.source_texts.get("repo/" + path, "").splitlines()
        for target in trace.callees(path, span, lines, every, edges):
            key = (target[0], target[1]["name"])
            arrows.append(((path, span["name"]), key))
            if key not in seen and len(seen) < NODES:
                seen.add(key)
                queue.append((target, depth + 1))
    rows = ["```mermaid", "flowchart TD"]
    rows += [f"  {node_id(path + '_' + symbol)}[{label(symbol + chr(10) + path)}]" for path, symbol in sorted(seen)]
    rows += [f"  {node_id(a[0] + '_' + a[1])} --> {node_id(b[0] + '_' + b[1])}" for a, b in dict.fromkeys(arrows) if b in seen]
    rows.append("```")
    path, span = start
    text = (f"What {span['name']} ({path}:{span['start']}-{span['end']}) calls, {trace.DEPTH} calls deep, from the parsed code. "
            "Calls through a variable or inheritance are not followed.\n\n" + "\n".join(rows))
    return {"text": text, "evidence": [{"path": path, "line": span["start"], "end_line": span["end"], "text": span["name"]}]}


RUNNERS = [  # (file, how to read the commands it defines)
    ("package.json", "scripts"), ("Makefile", "make"), ("pyproject.toml", "python"), ("tox.ini", "tox"),
    ("go.mod", "go"), ("Cargo.toml", "cargo"), ("Gemfile", "ruby"), ("composer.json", "composer"),
]


def commands(workspace):
    """[(command, where it comes from)] to install, run, and test the project, read from its own files."""
    listed, found = set(workspace.files()), []
    for name, kind in RUNNERS:
        if name not in listed:
            continue
        try:
            text = workspace.read(name)
        except (ValueError, OSError, UnicodeError):
            continue
        lines = text.splitlines()
        if kind == "scripts":
            try:
                scripts = (json.loads(text).get("scripts") or {})
            except ValueError:
                scripts = {}
            for script in [key for key in ("install", "start", "dev", "build", "test", "lint") if key in scripts]:
                line = next((number for number, row in enumerate(lines, 1) if f'"{script}"' in row), 1)
                found.append((f"npm run {script}" if script not in ("test", "start") else f"npm {script}", f"{name}:{line}"))
        elif kind == "make":
            for number, row in enumerate(lines, 1):
                target = re.match(r"^([A-Za-z][\w-]*):(?!=)", row)
                if target and target[1] in ("install", "setup", "run", "build", "test", "check", "lint", "dev"):
                    found.append((f"make {target[1]}", f"{name}:{number}"))
        elif kind == "python":
            found.append(("python -m pip install -e .", f"{name}:1"))
            test = "python -m pytest" if "pytest" in text else "python -m unittest discover"
            line = next((number for number, row in enumerate(lines, 1) if "pytest" in row), 1)
            found.append((test, f"{name}:{line}"))
        elif kind == "tox":
            found.append(("tox", f"{name}:1"))
        elif kind == "go":
            found += [("go build ./...", f"{name}:1"), ("go test ./...", f"{name}:1")]
        elif kind == "cargo":
            found += [("cargo build", f"{name}:1"), ("cargo test", f"{name}:1")]
        elif kind == "ruby":
            found.append(("bundle install", f"{name}:1"))
        elif kind == "composer":
            found.append(("composer install", f"{name}:1"))
    return found


def onboarding(model, workspace, knowledge, output=print):
    """A start-here guide: how to run and test it, what it uses, where the tests are, and what to read first."""
    parts, evidence = ["Onboarding guide for this project, from its own files. Every path:line can be opened to check it."], []
    steps = commands(workspace)
    if steps:
        parts += ["", "How to set it up, run it, and test it:"]
        for command, where in steps:
            parts.append(f"- `{command}` ({where})")
            path, line = where.rsplit(":", 1)
            evidence.append({"path": path, "line": int(line), "end_line": int(line), "text": command})
    tests = defaultdict(int)
    for path in workspace.files():
        if insights.is_test(path) and PurePosixPath(path).suffix.lower() in insights.CODE:
            tests[str(PurePosixPath(path).parent)] += 1
    if tests:
        parts += ["", "Where the tests are: " + ", ".join(f"{folder}/ ({count} files)" for folder, count in
                                                        sorted(tests.items(), key=lambda item: -item[1])[:4]) + "."]
    libraries = scans.dependencies(workspace, knowledge)
    declared = [line for line in libraries["text"].splitlines() if line.startswith("- `")][:10]
    if declared:
        parts += ["", "What it is built on (the first libraries it declares or imports):"] + declared
        evidence += libraries["evidence"][:10]
    tour = insights.tour(model, workspace, knowledge, output)
    parts += ["", "What to read first:", tour["text"]]
    evidence += tour["evidence"]
    parts += ["", "For a picture of how the parts connect, ask for an architecture diagram."]
    return {"text": "\n".join(parts), "evidence": evidence}
