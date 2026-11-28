"""Three whole-project answers made by plain code, with no model: the libraries a project uses, a security and
secrets scan, and what a change to one function or class would affect.

Each finding cites the line it comes from, and the lines are kept as evidence, so the answer can be checked like
any other. Nothing is run, imported, or installed.

- Dependencies: the packages the project's manifests declare (pyproject.toml, requirements*.txt, package.json,
  go.mod, Cargo.toml, Gemfile, composer.json, *.csproj), and the outside modules its code imports, with how many
  files import each. An import no manifest declares is pointed out.
- Security scan: hard-coded keys and passwords, eval and exec, shell commands built from strings, and SQL built from
  strings, found with line patterns. A pattern finds candidates, not proof: read each cited line. A secret's value
  is never shown, only its first characters.
- Impact: for a named function, method, or class, the places that call it (with the function each is in), the
  files that import it, and the tests that mention it.
"""
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from . import trace
from .insights import CODE, is_test

DEPENDENCIES = re.compile(r"\b(?:which|what|list|show)\b.*\b(?:librar(?:y|ies)|dependenc(?:y|ies)|packages|third[- ]party)\b"
                          r"|\b(?:external|third[- ]party)\s+(?:librar(?:y|ies)|dependenc(?:y|ies)|packages|code)\b"
                          r"|^\W*dependenc(?:y|ies)(?:\s+list)?\W*$", re.I)
SECURITY = re.compile(r"\bsecurity\s+(?:scan|check|review|audit|issues?|risks?|problems?)\b|\b(?:scan|check|look)\b.*\b(?:secrets?|"
                      r"vulnerabilit(?:y|ies)|security)\b|\bhard[- ]?coded\s+(?:keys?|secrets?|passwords?|credentials?|tokens?)\b"
                      r"|\b(?:any|find)\s+(?:secrets|vulnerabilities|injection)\b", re.I)
NAME = r"`?([A-Za-z_$][\w$.]*\w)`?(?:\(\))?"
IMPACT = re.compile(rf"\bwhat\s+(?:would\s+|will\s+|could\s+|might\s+)?(?:breaks?|break|be\s+affected|is\s+affected|depends\s+on)\b.*?"
                    rf"\b(?:change|changing|modify|modifying|edit|editing|remove|removing|rename|renaming|delete|deleting)\s+"
                    rf"(?:the\s+)?(?:function\s+|method\s+|class\s+)?{NAME}"
                    rf"|\b(?:impact|effect|blast\s+radius)\s+of\s+(?:changing|modifying|removing|renaming)\s+"
                    rf"(?:the\s+)?(?:function\s+|method\s+|class\s+)?{NAME}"
                    rf"|\bwho\s+(?:depends\s+on|uses)\s+(?:the\s+)?(?:function\s+|method\s+|class\s+)?{NAME}", re.I)
# Findings listed, places of one kind listed, and the longest line shown.
FINDINGS, PLACES, LINE_CHARS = 40, 12, 140


def requested(task):
    """("dependencies", None), ("security", None), ("impact", name), or None for any other task."""
    if (found := IMPACT.search(task)) and (name := next((group for group in found.groups() if group), None)):
        return "impact", name.rstrip(".")
    if SECURITY.search(task):
        return "security", None
    if DEPENDENCIES.search(task):
        return "dependencies", None
    return None


def run(kind, name, workspace, knowledge):
    """The answer for one kind of scan: {"text", "evidence"}."""
    if kind == "dependencies":
        return dependencies(workspace, knowledge)
    if kind == "security":
        return security(workspace)
    return impact(workspace, knowledge, name)


def shown(text):
    """One line of code, trimmed to fit a list."""
    text = " ".join(text.split())
    return text if len(text) <= LINE_CHARS else text[:LINE_CHARS - 1] + "…"


def evidence(path, line, text):
    """One evidence row: the cited line, so the answer's citations are checked against what was read."""
    return {"path": path, "line": line, "end_line": line, "text": text}


# Manifest file name (or suffix) -> how to find the declared packages: a pattern whose first group is the name.
MANIFESTS = [
    (re.compile(r"^requirements[\w.-]*\.txt$"), re.compile(r"^\s*([A-Za-z0-9][\w.-]*)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;]|$)")),
    (re.compile(r"^go\.mod$"), re.compile(r"^\s*(?:require\s+)?([\w.-]+\.[a-z]{2,}/[\w./-]+)\s+v[\d.]")),
    (re.compile(r"^Gemfile$"), re.compile(r"""^\s*gem\s+['"]([\w.-]+)['"]""")),
    (re.compile(r"\.csproj$"), re.compile(r"""<PackageReference\s+Include=['"]([\w.-]+)['"]""")),
]
# A Python requirement inside pyproject.toml: a quoted string in a dependencies list, such as "httpx>=0.27".
PYPROJECT_ITEM = re.compile(r"""^\s*(?:dependencies\s*=\s*\[)?\s*['"]([A-Za-z0-9][\w.-]*)\s*(?:\[[^\]]*\])?\s*[<>=!~;,'"]""")
# Import names that are spelled differently from the package that provides them.
ALIASES = {"yaml": "pyyaml", "pil": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python", "bs4": "beautifulsoup4",
           "dateutil": "python-dateutil", "jose": "python-jose", "jwt": "pyjwt", "dotenv": "python-dotenv",
           "google": "google-api-python-client", "attr": "attrs", "magic": "python-magic", "serial": "pyserial"}


def normal(name):
    """A package name in the form that two spellings of it share: lower case, with - _ . the same."""
    return re.sub(r"[-_.]+", "-", name.lower())


def declared(workspace):
    """[(package, manifest path, line)] for every package the project's manifests declare."""
    found = []
    for path in workspace.files():
        file = PurePosixPath(path).name
        try:
            if file not in ("pyproject.toml", "package.json", "Cargo.toml", "composer.json") and not any(
                    pattern.search(file) for pattern, _ in MANIFESTS):
                continue
            text = workspace.read(path)
        except (ValueError, OSError, UnicodeError):
            continue
        lines = text.splitlines()
        if file in ("package.json", "composer.json"):
            try:
                data = json.loads(text)
            except ValueError:
                continue
            sections = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies", "require", "require-dev")
            for section in sections:
                for package in (data.get(section) or {}) if isinstance(data, dict) else {}:
                    if package == "php" or package.startswith("ext-"):
                        continue
                    line = next((number for number, row in enumerate(lines, 1) if f'"{package}"' in row), 1)
                    found.append((package, path, line))
            continue
        table = ""
        for number, row in enumerate(lines, 1):
            if file in ("pyproject.toml", "Cargo.toml"):
                if re.match(r"^\s*\[", row):
                    table = row.strip()
                    continue
                if file == "Cargo.toml":
                    match = re.match(r"^\s*([\w-]+)\s*=", row) if "dependencies" in table else None
                else:
                    inside = table in ("[project]", "[project.optional-dependencies]") or "dependencies" in table
                    match = PYPROJECT_ITEM.match(row) if inside and not re.match(r"^\s*(?:name|version|requires-python)\s*=", row) else None
            else:
                pattern = next(item for name_pattern, item in MANIFESTS if name_pattern.search(file))
                match = pattern.search(row) if not row.lstrip().startswith(("#", "//", "-")) else None
            if match:
                found.append((match[1], path, number))
    return found


def dependencies(workspace, knowledge):
    """The libraries the project declares and the outside modules its code imports."""
    packages = declared(workspace)
    missing = getattr(getattr(knowledge, "dependencies", None), "missing", None) or {}
    used = defaultdict(list)  # top-level module -> [(path, line)]
    for path, rows in missing.items():
        for row in rows:
            if row.get("local"):
                continue
            module = row["module"]
            top = module.split("/")[0] if not module.startswith("@") else "/".join(module.split("/")[:2])
            top = top.split(".")[0] if PurePosixPath(path).suffix in (".py", ".pyi") else top
            if (path, row["line"]) not in used[top]:
                used[top].append((path, row["line"]))
    known = {normal(package) for package, _, _ in packages}
    rows, lines = [], []
    if packages:
        # Manifests at the top of the project list every package; ones in subfolders, such as examples, are summarised.
        by_file = defaultdict(list)
        for package, path, line in packages:
            by_file[path].append((package, line))
        top = sorted(path for path in by_file if "/" not in path)
        for path in top:
            lines.append(f"Declared in {path} ({len(by_file[path])}):")
            for package, line in by_file[path][:FINDINGS]:
                lines.append(f"- `{package}` ({path}:{line})")
                rows.append(evidence(path, line, package))
        nested = sorted(path for path in by_file if "/" in path)
        if nested:
            lines.append(("" if top else "") + f"Manifests in subfolders ({len(nested)}):")
            for path in nested[:PLACES]:
                package, line = by_file[path][0]
                lines.append(f"- {path}: {len(by_file[path])} package{'s' if len(by_file[path]) != 1 else ''}, "
                             f"first `{package}` at {path}:{line}")
                rows.append(evidence(path, line, package))
    else:
        lines.append("No manifest (pyproject.toml, requirements.txt, package.json, go.mod, Cargo.toml, Gemfile, "
                     "composer.json, or .csproj) declares any packages.")
    if used:
        lines.append("")
        lines.append(f"Imported by the code from outside the repository ({len(used)}), most widely used first:")
        for module, places in sorted(used.items(), key=lambda item: (-len({p for p, _ in item[1]}), item[0]))[:FINDINGS]:
            files = len({path for path, _ in places})
            path, line = places[0]
            note = "" if normal(module) in known or normal(ALIASES.get(module.lower(), module)) in known or not packages \
                else "; not declared in a manifest"
            lines.append(f"- `{module}`: {files} file{'s' if files != 1 else ''}, first at {path}:{line}{note}")
            rows.append(evidence(path, line, module))
    lines.append("")
    lines.append("Found from the manifests and the parsed imports, without running anything. Standard-library and "
                 "built-in modules are not listed.")
    return {"text": "\n".join(lines), "evidence": rows}


SECRET_VALUE = r"""['"]([^'"\s]{8,})['"]"""
RULES = [
    ("Hard-coded secret", re.compile(r"\b(AKIA[0-9A-Z]{16})\b")),
    ("Hard-coded secret", re.compile(r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)")),
    ("Hard-coded secret", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{20,})\b")),
    ("Hard-coded secret", re.compile(r"(?i)\b[\w-]*(?:password|passwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|token)"
                                     r"[\w-]*['\"]?\s*[:=]\s*" + SECRET_VALUE)),
    ("Code run from a string", re.compile(r"(?<![\w.])(?:eval|exec)\s*\((?!\s*\))|\bnew\s+Function\s*\(")),
    ("Shell command", re.compile(r"\bshell\s*=\s*True\b|\bos\.(?:system|popen)\s*\(|\bchild_process\.exec(?:Sync)?\s*\(|"
                                 r"(?<![\w.])exec(?:Sync)?\s*\(\s*(?:`[^`]*\$\{|['\"][^'\"]*['\"]\s*\+)")),
    ("SQL built from a string", re.compile(r"\.(?:execute|executemany|raw|query)\s*\(\s*(?:f['\"]|['\"][^'\"]*['\"]\s*(?:%|\+)|"
                                           r"['\"][^'\"]*['\"]\s*\.format\(|`[^`]*\$\{)", re.I)),
    ("Unsafe loading", re.compile(r"\bpickle\.loads?\s*\(|\byaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)|\bmarshal\.loads\s*\(")),
    ("TLS check turned off", re.compile(r"\bverify\s*=\s*False\b|rejectUnauthorized\s*:\s*false")),
]
# Values that are plainly placeholders, not real secrets.
PLACEHOLDER = re.compile(r"(?i)^(?:x+|\*+|changeme|change_me|example|sample|dummy|test|testing|password|secret|your[\w-]*|"
                         r"<[^>]*>|\$\{[^}]*\}|%\([^)]*\)s|\{\{[^}]*\}\})$|example|placeholder|xxxx|dummy")
SCANNED = CODE | {".env", ".ini", ".cfg", ".toml", ".yaml", ".yml", ".json", ".properties", ".sh", ".bash", ".zsh", ".tf"}


def masked(text, value):
    """The line with a secret's value replaced by its first 4 characters and stars."""
    return text.replace(value, value[:4] + "*" * min(12, max(4, len(value) - 4))) if value else text


def security(workspace):
    """Lines that look like a security risk, most serious kind first, each cited and with any secret masked."""
    findings = []
    for path in workspace.files():
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in SCANNED and PurePosixPath(path).name not in ("Dockerfile",):
            continue
        try:
            lines = workspace.read(path).splitlines()
        except (ValueError, OSError, UnicodeError):
            continue
        for number, line in enumerate(lines, 1):
            if len(line) > 2000:
                continue
            for kind, rule in RULES:
                match = rule.search(line)
                if not match:
                    continue
                value = match.group(1) if match.groups() else ""
                if kind == "Hard-coded secret" and value and (PLACEHOLDER.search(value) or re.search(r"(?i)environ|getenv|process\.env", line)):
                    continue
                findings.append((kind, path, number, masked(line, value if kind == "Hard-coded secret" else "")))
                break
    order = [kind for kind, _ in RULES]
    findings.sort(key=lambda row: (order.index(row[0]), is_test(row[1]), row[1], row[2]))
    if not findings:
        return {"text": "No line matched the security patterns (hard-coded keys and passwords, eval and exec, shell "
                        "commands, SQL built from strings, unsafe loading, turned-off TLS checks). Patterns cannot prove "
                        "code is safe; they only find the common mistakes.", "evidence": []}
    lines, rows, current = [f"{len(findings)} line{'s' if len(findings) != 1 else ''} to check, most serious kind first. "
                               "These are pattern matches, not proof: read each cited line."], [], None
    for kind, path, number, text in findings[:FINDINGS]:
        if kind != current:
            lines += ["", kind + ":"]
            current = kind
        lines.append(f"- {path}:{number}{' (test)' if is_test(path) else ''}: `{shown(text)}`")
        rows.append(evidence(path, number, text))
    if len(findings) > FINDINGS:
        lines += ["", f"... and {len(findings) - FINDINGS} more."]
    lines += ["", "Secrets are shown masked. A real secret in the code should be revoked and moved to an environment "
                  "variable or a secrets store, since it is in the project's history too."]
    return {"text": "\n".join(lines), "evidence": rows}


def impact(workspace, knowledge, name):
    """What a change to name would affect: its callers, the files that import it, and the tests that mention it."""
    every = trace.functions(knowledge) if knowledge is not None else []
    places = trace.locate(name, every)
    short = name.rsplit(".", 1)[-1]
    if not places:
        definitions = (getattr(knowledge, "symbols", {}) or {}).get(short, []) if knowledge is not None else []
        if not definitions:
            return {"text": f"No function, method, or class named {name} was found in the parsed code.", "evidence": []}
        places = [(path, {"name": short, "start": line, "end": line}) for path, line in definitions]
    target = places[0]
    path, span = target
    lines = [f"`{span['name']}` is defined at {path}:{span['start']}-{span['end']}."]
    rows = [evidence(path, span["start"], span["name"])]
    if len(places) > 1:
        lines.append("Other definitions with that name, not followed here: "
                     + ", ".join(f"{other}:{info['start']}" for other, info in places[1:4]) + ".")
    # Every indexed code file is looked through, not a capped search: in a large project the first matches of a name
    # are often documentation, which would crowd out the calls and the tests.
    texts = getattr(knowledge, "source_texts", {}) if knowledge is not None else {}
    call = re.compile(rf"(?<![\w$]){re.escape(short)}\s*\(")
    word = re.compile(rf"(?<![\w$]){re.escape(short)}(?![\w$])")
    calls, tests = [], []
    for source in sorted(texts):
        other = source.removeprefix("repo/")
        if Path(other).suffix.lower() not in CODE or trace.family(other) != trace.family(path):
            continue
        testing = is_test(other)
        for number, row in enumerate(texts[source].splitlines(), 1):
            if (other, number) == (path, span["start"]) or re.match(r"\s*(?:#|//|/?\*)", row):
                continue
            if testing and word.search(row):
                tests.append({"path": other, "line": number, "text": row})
            elif not testing and call.search(row) and not re.search(rf"\b(?:def|function|func|fn)\s+{re.escape(short)}\b", row):
                inside = next((info["name"] for where, info in every if where == other
                               and info["start"] <= number <= info["end"] and info["end"] - info["start"] < 400), "")
                calls.append((other, number, inside))
    edges = getattr(getattr(knowledge, "dependencies", None), "edges", None) or {}
    importers = sorted(other for other, targets in edges.items() if path in targets
                       and (short in targets[path] or "" in targets[path] or span["name"].split(".")[0] in targets[path]))
    lines.append("")
    if calls:
        lines.append(f"Called from ({len(calls)} place{'s' if len(calls) != 1 else ''} outside the tests):")
        for where, line, inside in calls[:PLACES]:
            lines.append(f"- {where}:{line}" + (f", in `{inside}`" if inside else ""))
            rows.append(evidence(where, line, short))
    else:
        lines.append("No call to it was found by name. It may be called through a variable, a decorator, or from outside "
                     "this project.")
    if importers:
        lines.append("")
        lines.append(f"Files that import {path} ({len(importers)}):")
        lines += [f"- {other}" for other in importers[:PLACES]]
    lines.append("")
    if tests:
        test_files = sorted({match["path"] for match in tests})
        lines.append(f"Tests that mention it ({len(test_files)} file{'s' if len(test_files) != 1 else ''}), to run after the change:")
        for match in sorted(tests, key=lambda row: (row["path"], row["line"]))[:PLACES]:
            lines.append(f"- {match['path']}:{match['line']}")
            rows.append(evidence(match["path"], match["line"], match.get("text", "")))
    else:
        lines.append("No test mentions it by name, so a change to it is not checked by a test that names it.")
    lines += ["", "Found by parsing and searching the code, not by running it: calls made through a variable or "
                  "inheritance are not seen."]
    return {"text": "\n".join(lines), "evidence": rows}
