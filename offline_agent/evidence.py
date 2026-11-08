"""Bounded source evidence and conservative, deterministic answer checks."""
import json
import re
from collections import Counter
from pathlib import PurePosixPath

LIMIT = 4000
# Start at a token boundary and require an alphabetic extension, so URLs and host:port pairs are not citations.
CITATION = re.compile(r"(?<![\w./:-])(?:repo/)?([\w./-]+\.[A-Za-z]\w*):(\d+)(?:-(\d+))?(?![\d/])")
EDIT_VERB = r"(?:edit|modif|chang|creat|fix|implement|add|writ|delet)\w*"
NEGATED_EDIT = re.compile(rf"\b(?:do\s+not|don['’]?t|never|without)\s+{EDIT_VERB}(?:\s*(?:,|/|or|and)\s*{EDIT_VERB})*")


def classify(path, snippet=""):
    parts = PurePosixPath(path).parts
    name = parts[-1].lower() if parts else ""
    if "tests" in parts or "test" in parts or name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    if "stub" in path.lower():
        return "stub"
    if name.endswith((".md", ".rst", ".txt")):
        return "reference"
    if "FastAPI(" in snippet or path.endswith("/app/main.py"):
        return "service"
    return "source"


def retain(previous, matches):
    """Preserve citations before snippets when evidence reaches its byte limit."""
    records = [dict(item) for item in previous]
    limited = False
    for match in matches:
        item = {"path": match["path"], "line": match["line"],
                "kind": classify(match["path"], match.get("text", "")),
                "snippet": match.get("text", "")[:64]}
        if "end_line" in match:
            item["end_line"] = match["end_line"]
        if match.get("in"):
            item["in"] = match["in"]
        old = next((old for old in records if (old["path"], old["line"]) == (item["path"], item["line"])), None)
        if old is not None:
            if "end_line" in item:
                old["end_line"] = item["end_line"]
            continue
        if len(records) >= 50:
            limited = True
            break
        records.append(item)
    while len(json.dumps(records, ensure_ascii=False).encode()) > LIMIT:
        with_snippet = next((row for row in records if row.get("snippet")), None)
        if with_snippet is not None:
            with_snippet["snippet"] = ""
        else:
            records.pop()
            limited = True
    return records, limited


def discovery_task(task):
    """Only plan explicit FastAPI entry-point discovery, never edits or general explanations."""
    text = task.lower()
    return ("fastapi" in text and bool(re.search(r"entry[ -]?points?", text))
            and bool(re.search(r"\b(find|list|identify|locate|discover)\b", text))
            and not re.search(r"\b(edit|modify|change|create|fix|implement|add|write|delete)\b", NEGATED_EDIT.sub("", text)))


def mentions(line, path):
    """Match a whole path token, so app/main.py does not match tests/app/main.py."""
    path = re.escape(path.removeprefix("repo/"))
    return re.search(rf"(?<![\w./-])(?:repo/)?{path}(?![\w/-]|\.\w)", line) is not None


def check_answer(answer, evidence, *, discovery=False, require_citation=True):
    """Flag detectable gaps, not a semantic proof of truth.

    require_citation is for answers about the code. A report that a change was made needs no line citations,
    and flagging those made every such report look unverified.
    """
    issues = []
    repeated = Counter(line.strip() for line in answer.splitlines() if len(line.strip()) >= 40)
    if any(count >= 3 for count in repeated.values()):
        issues.append("The answer repeats the same substantial text three or more times.")
    services = {row["path"].removeprefix("repo/") for row in evidence if row.get("kind") == "service"}
    if len(services) > 1 and re.search(
        r"\b(?:the|only|single|sole)\s+(?:fastapi\s+)?entry[ -]?point\s+(?:is|was|can be|has been)", answer, re.I
    ):
        issues.append("The answer implies a single entry point, but multiple service candidates were found.")
    citations = CITATION.findall(answer)
    valid = set()
    for path, start, end in citations:
        path = path.removeprefix("repo/")
        start, end = int(start), int(end or start)
        if any(row["path"].removeprefix("repo/") == path and row["line"] <= start <= end <= row.get("end_line", row["line"])
               for row in evidence):
            valid.add(path)
        else:
            issues.append(f"Citation not supported by retained evidence: {path}:{start}-{end}.")
    if require_citation and evidence and not valid:
        issues.append("Repository claims require path:line citations supported by observed evidence.")
    if discovery:
        missing = services - valid
        if missing:
            issues.append(f"Missing supported citations for {len(missing)} discovered service candidate(s).")
        section = None
        for line in answer.splitlines():
            heading = line.strip().strip('#*: ').lower()
            if heading in ('service candidates', 'services', 'service entry points'):
                section = 'service'
            elif heading in ('tests', 'test apps', 'test candidates'):
                section = 'test'
            elif heading in ('stubs', 'stub apps', 'stub candidates'):
                section = 'stub'
            elif line.strip() and not line.lstrip().startswith(('-', '*')):
                section = None
            if section:
                for row in evidence:
                    if mentions(line, row['path']) and row['kind'] != section:
                        issues.append(f"Classification mismatch: {row['path']} is labeled {row['kind']} in evidence, not {section}.")
    return issues


def inventory(evidence, limited=False):
    lines = ["Observed source candidates (heuristic classification; not proof of deployment):"]
    for kind in ('service', 'test', 'stub', 'source', 'reference'):
        records = [row for row in evidence if row['kind'] == kind]
        if records:
            lines.append(f"{kind.capitalize()} candidates:")
            lines.extend(f"- [{row['path']}:{row['line']}]" for row in records)
    lines.append("Coverage is limited to the searched literal and eligible files." +
                 (" Search/evidence limits were reached." if limited else " No search/evidence limit was reached."))
    return '\n'.join(lines)
