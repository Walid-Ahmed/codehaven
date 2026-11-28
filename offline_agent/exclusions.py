"""Root-scoped Git-style exclusion rules, with no Git subprocess or network."""

from pathspec import GitIgnoreSpec


class IgnoreConfigurationError(ValueError):
    """An ignore file or --exclude pattern is unusable. Raised, never skipped, so a bad rule cannot silently expose files."""


class Exclusions:
    """Decides whether a path is ignored by .gitignore files, .agentignore files, or --exclude patterns."""

    def __init__(self, root, patterns=()):
        """Validate the --exclude patterns and read the root rule files now, so a bad configuration fails at startup."""
        self.root = root
        self.patterns = tuple(patterns)
        # cache: (directory, filename) -> compiled rules, or None for no file.
        # decisions: (path parts, is a directory) -> whether that path is ignored.
        self.cache = {}
        self.decisions = {}
        for pattern in self.patterns:
            if not isinstance(pattern, str) or not pattern.strip() or '\n' in pattern or '\r' in pattern:
                raise IgnoreConfigurationError("--exclude needs a nonblank single-line Git-style pattern")
            if pattern.startswith('!'):
                raise IgnoreConfigurationError("--exclude is deny-only; use .agentignore for ordered negation rules")
        self.extra = self.compile(self.patterns, '--exclude')
        for filename in ('.gitignore', '.agentignore'):
            self.spec('', filename)

    @staticmethod
    def compile(lines, source):
        """Compile Git-style pattern lines; source only says where they came from if they are invalid."""
        try:
            return GitIgnoreSpec.from_lines(lines)
        except Exception as exc:
            raise IgnoreConfigurationError(f"Invalid exclusion rules in {source}: {exc}") from exc

    def spec(self, directory, filename):
        """The compiled rules of one rule file in a directory (a root-relative POSIX string, "" for the root), or None
        when there is none. Read once, then cached. Strings, not Paths, because this runs for every directory listed."""
        key = (directory, filename)
        if key not in self.cache:
            path = self.root / directory / filename if directory else self.root / filename
            if path.is_symlink():
                self.cache[key] = None  # Git does not follow symlinked .gitignore files.
            else:
                try:
                    if path.exists() and not path.is_file():
                        raise IgnoreConfigurationError(f"Exclusion configuration must be a regular file: {path}")
                    with path.open('rb') as handle:
                        # One byte past the 64 KB limit is enough to detect an oversized file.
                        data = handle.read(64_001)
                    if len(data) > 64_000:
                        raise IgnoreConfigurationError(f"Exclusion file exceeds 64 KB: {path}")
                    self.cache[key] = self.compile(data.decode('utf-8').splitlines(), path)
                except FileNotFoundError:
                    self.cache[key] = None
                except (OSError, UnicodeError) as exc:
                    raise IgnoreConfigurationError(f"Cannot read exclusion rules in {path}: {exc}") from exc
        return self.cache[key]

    def layer(self, parts, directory, filename):
        """Whether one kind of rule file ignores the path (given as its parts), applying each ancestor directory's file
        from the root down."""
        ignored = False
        for depth in range(len(parts)):
            spec = self.spec('/'.join(parts[:depth]), filename)
            if spec is not None:
                local = '/'.join(parts[depth:]) + ('/' if directory else '')
                # True: a rule ignores the path. False: a ! rule re-includes it. None: this file has no matching rule.
                # A deeper file's verdict replaces a shallower one, as in Git.
                decision = spec.check_file(local).include
                if decision is not None:
                    ignored = decision
        return ignored

    def ignored(self, relative, directory=False):
        """Whether the path, or any directory above it, is excluded by any of the three rule sources."""
        # Evaluate parents too: a child cannot re-include an excluded directory.
        # Rule files are cached once read, so each prefix decision can be cached as well.
        parts = relative.parts
        for depth in range(1, len(parts) + 1):
            is_dir = depth < len(parts) or directory
            key = (parts[:depth], is_dir)
            decision = self.decisions.get(key)
            if decision is None:
                prefix = parts[:depth]
                # Each source can only add exclusions: a ! rule in .agentignore cannot re-include what .gitignore ignores.
                decision = self.decisions[key] = bool(
                    self.layer(prefix, is_dir, '.gitignore')
                    or self.layer(prefix, is_dir, '.agentignore')
                    or self.extra.match_file('/'.join(prefix) + ('/' if is_dir else '')))
            if decision:
                return True
        return False
