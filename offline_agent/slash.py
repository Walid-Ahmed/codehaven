"""Slash commands: short forms of the requests CodeHaven has a purpose-built answer for, in the page and the terminal.

/trace checkout_total is the same as typing "Trace checkout_total". Each command only writes the request out in
full, so it is answered exactly as the typed request would be. /help lists them.
"""
COMMANDS = {
    "trace": ("Trace {}", "a function: what it calls and who calls it", True),
    "review": ("Review {}", "a file, one function at a time", True),
    "why": ("Why is {} here?", "a line, from Git history (give path:line)", True),
    "impact": ("What breaks if I change {}?", "a function: its callers, importers, and tests", True),
    "explain": ("Explain {}", "a file, function, or anything else", True),
    "tour": ("Give me a project tour", "a reading order for the project", False),
    "onboard": ("Write an onboarding guide for a new developer", "how to run, test, and read the project", False),
    "diagram": ("Draw an architecture diagram", "a Mermaid diagram of the imports (or /diagram name for calls)", False),
    "deps": ("Which libraries does this project use?", "the libraries it declares and imports", False),
    "scan": ("Run a security scan", "hard-coded secrets and risky code", False),
    "gaps": ("Which functions have no tests?", "functions no test mentions", False),
    "changes": ("Explain my uncommitted changes", "your Git changes", False),
    "review-changes": ("Review my changes", "only the functions your changes touched", False),
    "commit": ("Write a commit message for my changes", "a commit message, to copy", False),
    "pr": ("Write a PR description for my changes", "a pull request description, to copy", False),
}


def expand(text):
    """The full request for a slash command, the help text for /help, or text itself when it is not a command.

    Raises ValueError for a command that needs an argument and has none, or an unknown command."""
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith("//"):
        return text
    name, _, argument = stripped[1:].partition(" ")
    name, argument = name.lower(), argument.strip()
    if name == "help":
        return None
    if name not in COMMANDS:
        raise ValueError(f"Unknown command /{name}. Type /help to see the commands.")
    template, _, needs = COMMANDS[name]
    if name == "diagram" and argument:
        return f"Draw a Mermaid diagram of the calls of {argument}"
    if needs and not argument:
        raise ValueError(f"/{name} needs something to work on, such as /{name} {'store.py' if name == 'review' else 'checkout_total'}.")
    return template.format(argument) if needs else template


def help_text(extra=()):
    """One line per command, for /help."""
    lines = ["Commands (each is a short form of a request you could type in full):"]
    for name, (_, what, needs) in COMMANDS.items():
        lines.append(f"  /{name}{' <name>' if needs else ''}: {what}")
    lines += [f"  {line}" for line in extra]
    return "\n".join(lines)
