"""Static Python import resolution over already indexed local files only."""
import ast
import sys
from collections import defaultdict
from pathlib import PurePosixPath


class PythonDependencies:
    """Imports, top-level definitions, and symbol spans of Python files, found with ast: nothing is imported or run."""
    def __init__(self, files, reuse=None):
        """Analyse files, a dict of repository path -> text.

        reuse maps a path to its (definitions, spans, edges, missing) from an earlier analysis of the same text and the same
        set of paths; those files are not parsed again.

        edges: importer -> {imported file: [names taken from it]}, where "" stands for the module as a whole.
        definitions: file -> {top-level name: line}. spans: file -> [{"name", "start", "end"}] for every function
        and class, nested ones too. A file that does not parse is counted in parse_failures and skipped.
        missing: file -> [{"module", "name", "line", "local"}] for each import of code that is not in the repository and
        not in Python's standard library: name is what the import binds in the file, and local marks a relative import.
        """
        self.missing = {}
        self.edges = {}
        self.definitions = {}
        self.spans = {}
        self.parse_failures = 0
        self.files = files
        reuse = reuse or {}
        for name, text in files.items():
            if name in reuse:
                self.definitions[name], self.spans[name], self.edges[name], self.missing[name] = reuse[name]
                continue
            try:
                tree = ast.parse(text, filename=name)
            except (SyntaxError, ValueError, RecursionError):
                self.parse_failures += 1
                continue
            definitions = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions[node.name] = node.lineno
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            definitions[target.id] = node.lineno
            self.definitions[name] = definitions
            # One walk of the tree collects both the spans and the imports: walking it twice doubled the indexing time.
            spans = self.spans[name] = []
            found = defaultdict(set)
            missing = self.missing[name] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # A span starts at its first decorator, so decorators are shown together with what they decorate.
                    spans.append({"name": node.name, "start": min([node.lineno] + [d.lineno for d in node.decorator_list]),
                                  "end": node.end_lineno})
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        target = self._absolute(alias.name, name)
                        self._add(found, target, None)
                        if not target:
                            # "import a.b" binds a; "import a.b as c" binds c.
                            self._missing(missing, alias.name, alias.asname or alias.name.split('.')[0], node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    # Relative import: one dot is the importing file's own folder, and each further dot climbs one folder.
                    if node.level:
                        parent = list(PurePosixPath(name).parent.parts)
                        if node.level > len(parent):
                            for alias in node.names:
                                self._missing(missing, '.' * node.level + (node.module or ''), alias.asname or alias.name,
                                              node.lineno, local=True)
                            continue
                        base = parent[:len(parent) - node.level + 1]
                        if node.module:
                            base += node.module.split('.')
                        stem = '/'.join(base)
                        target = self._resolve(stem)
                        for alias in node.names:
                            # In "from .pkg import name", name may be a submodule with its own file, or a symbol inside pkg.
                            child = self._resolve(stem + '/' + alias.name)
                            self._add(found, child or target, None if child else alias.name)
                            if not child and not target:
                                self._missing(missing, '.' * node.level + (node.module or ''), alias.asname or alias.name,
                                              node.lineno, local=True)
                    else:
                        module = node.module or ''
                        target = self._absolute(module, name)
                        for alias in node.names:
                            child = self._absolute(module + '.' + alias.name, name)
                            self._add(found, child or target, None if child else alias.name)
                            if not child and not target:
                                self._missing(missing, module, alias.asname or alias.name, node.lineno)
            # Package initializers can also define import-time behavior.
            for target in list(found):
                parts = PurePosixPath(target).parent.parts
                for length in range(1, len(parts) + 1):
                    initializer = '/'.join(parts[:length]) + '/__init__.py'
                    if initializer in files and initializer != name:
                        found[initializer].add('')
            self.edges[name] = {target: sorted(symbols) for target, symbols in sorted(found.items())
                                if target != name}

    def _resolve(self, stem):
        """The indexed file for a module path without suffix: a module, a package initializer, or a stub. None if not indexed."""
        for suffix in ('.py', '/__init__.py', '.pyi', '/__init__.pyi'):
            if stem + suffix in self.files:
                return stem + suffix
        return None

    def _absolute(self, module, importer):
        """The indexed file for a dotted module name, tried from the repository root, from src/, and beside the importer."""
        stem = module.replace('.', '/')
        for prefix in ('', 'src/', str(PurePosixPath(importer).parent) + '/'):
            result = self._resolve(prefix + stem)
            if result:
                return result
        return None

    @staticmethod
    def _missing(missing, module, bound, line, local=False):
        """Record an import of code that is not here. The standard library and __future__ are known code, so left out."""
        top = module.split('.')[0]
        if not local and (top in sys.stdlib_module_names or top == '__future__'):
            return
        if bound != '*':
            missing.append({"module": module, "name": bound, "line": line, "local": local})

    @staticmethod
    def _add(found, target, symbol):
        """Record that target is imported; a whole-module or star import is stored as the empty name."""
        if target:
            found[target].add('' if symbol in (None, '*') else symbol)
