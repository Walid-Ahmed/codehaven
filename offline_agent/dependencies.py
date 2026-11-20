"""Static Python import resolution over already indexed local files only."""
import ast
from collections import defaultdict
from pathlib import PurePosixPath


class PythonDependencies:
    """Imports, top-level definitions, and symbol spans of Python files, found with ast: nothing is imported or run."""
    def __init__(self, files):
        """Analyse files, a dict of repository path -> text.

        edges: importer -> {imported file: [names taken from it]}, where "" stands for the module as a whole.
        definitions: file -> {top-level name: line}. spans: file -> [{"name", "start", "end"}] for every function
        and class, nested ones too. A file that does not parse is counted in parse_failures and skipped.
        """
        self.edges = {}
        self.definitions = {}
        self.spans = {}
        self.parse_failures = 0
        self.files = files
        for name, text in files.items():
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
            # A span starts at its first decorator, so decorators are shown together with what they decorate.
            self.spans[name] = [
                {"name": node.name, "start": min([node.lineno] +
                 [d.lineno for d in node.decorator_list]), "end": node.end_lineno}
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            found = defaultdict(set)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add(found, self._absolute(alias.name, name), None)
                elif isinstance(node, ast.ImportFrom):
                    # Relative import: one dot is the importing file's own folder, and each further dot climbs one folder.
                    if node.level:
                        parent = list(PurePosixPath(name).parent.parts)
                        if node.level > len(parent):
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
                    else:
                        module = node.module or ''
                        target = self._absolute(module, name)
                        for alias in node.names:
                            child = self._absolute(module + '.' + alias.name, name)
                            self._add(found, child or target, None if child else alias.name)
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
    def _add(found, target, symbol):
        """Record that target is imported; a whole-module or star import is stored as the empty name."""
        if target:
            found[target].add('' if symbol in (None, '*') else symbol)
