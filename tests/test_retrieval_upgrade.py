import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.javascript import JavaScriptDependencies
from offline_agent.cli import main
from offline_agent.knowledge import KnowledgeBase


class JavaScriptTests(unittest.TestCase):
    def test_esm_alias_default_type_and_reexports(self):
        files = {'main.ts': 'import total, { price as p, type Item } from "./cost.js";',
                 'cost.ts': 'export function price(x: number) { return x; }\nexport default function total() { return 2; }',
                 'index.ts': 'export { price as cost } from "./cost";'}
        graph = JavaScriptDependencies(files)
        self.assertEqual(graph.edges['main.ts']['cost.ts'], ['Item', 'default', 'price'])
        self.assertEqual(graph.edges['index.ts']['cost.ts'], ['price'])
        self.assertEqual(graph.definitions['cost.ts']['default'], 2)

    def test_relative_directory_and_commonjs(self):
        files = {'src/app.js': 'const x = require("../lib"); import "./side.js";',
                 'lib/index.js': 'module.exports = {};', 'src/side.js': 'console.log(1);'}
        graph = JavaScriptDependencies(files)
        self.assertEqual(set(graph.edges['src/app.js']), {'lib/index.js', 'src/side.js'})

    def test_comments_strings_dynamic_packages_and_escapes_ignored(self):
        files = {'main.js': '''// import x from './fake';
const text = "require('./fake')";
import pkg from 'external'; import alias from '@/fake';
import('./fake'); require(variable); require('../../escape');
''', 'fake.js': 'export const x = 1;'}
        self.assertEqual(JavaScriptDependencies(files).edges['main.js'], {})

    def test_jsx_tsx_arrows_methods_and_anonymous_default(self):
        files = {'View.tsx': 'export const View = () => { return <div>Hello</div>; };',
                 'Anon.jsx': 'export default () => { return <span />; };',
                 'Class.ts': 'class A { calculate(x: number) { return x; } }'}
        graph = JavaScriptDependencies(files)
        self.assertEqual(graph.parse_failures, 0)
        self.assertEqual(graph.spans['View.tsx'][0]['name'], 'View')
        self.assertEqual(graph.spans['Anon.jsx'][0]['name'], 'default')
        self.assertIn('calculate', graph.definitions['Class.ts'])

    def test_invalid_syntax_retains_other_graphs(self):
        graph = JavaScriptDependencies({'bad.ts': 'function {', 'ok.js': 'import "./bad.ts";'})
        self.assertEqual(graph.parse_failures, 1)
        self.assertIn('bad.ts', graph.edges['ok.js'])

    def test_local_export_alias_is_resolved_to_original_definition(self):
        graph = JavaScriptDependencies({'lib.ts': 'function hidden() { return 1; }\nexport { hidden as visible };'})
        self.assertEqual(graph.definitions['lib.ts']['visible'], 1)


class RetrievalBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def test_complete_decorated_python_function_crosses_old_chunk_boundary(self):
        self.put('code.py', '@decorator\ndef uniqueworkflow():\n' + '    step = 1\n' * 55 + '    return step\n')
        result = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))
        passage = result['results'][0]
        self.assertTrue(passage['symbol_complete'])
        self.assertTrue(passage['text'].startswith('@decorator'))
        self.assertEqual(passage['end_line'], 58)
        self.assertIn('return step', passage['text'])

    def test_long_named_typescript_dependency_is_complete(self):
        self.put('app.ts', 'import { calculate } from "./price.js";\nexport function uniqueworkflow() { return calculate(10); }')
        self.put('price.ts', '// header\n' * 50 + 'export function calculate(x: number) {\n' + '  x += 1;\n' * 40 + '  return x;\n}')
        result = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))
        dep = next(r for r in result['results'] if r['source'] == 'repo/price.ts')
        self.assertEqual(dep['kind'], 'dependency')
        self.assertTrue(dep['symbol_complete'])
        self.assertEqual(dep['start_line'], 51)
        self.assertEqual(dep['end_line'], 93)
        self.assertIn('return x;', dep['text'])

    def test_two_imported_functions_from_one_module(self):
        self.put('main.py', 'from lib import first, second\ndef uniqueworkflow(): return first() + second()')
        self.put('lib.py', 'def first():\n    return 1\n' + '# gap\n' * 40 + 'def second():\n    return 2\n')
        results = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))['results']
        self.assertEqual({r['symbol'] for r in results if r['kind'] == 'dependency'}, {'first', 'second'})

    def test_symbols_carry_the_values_of_constants_they_use(self):
        self.put('rooms.py', 'MAX_GUESTS = 4\n' + '# gap\n' * 40 + 'def uniqueworkflow(guests):\n    return guests > MAX_GUESTS\n')
        row = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))['results'][0]
        self.assertEqual(row['constants'], [{'name': 'MAX_GUESTS', 'line': 1, 'text': 'MAX_GUESTS = 4'}])

    def test_javascript_constants_are_found_too(self):
        self.put('cart.js', 'const TAX = 0.1;\n\nexport function uniqueworkflow(p) {\n  return p * (1 + TAX);\n}\n')
        row = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))['results'][0]
        self.assertEqual(row['constants'], [{'name': 'TAX', 'line': 1, 'text': 'const TAX = 0.1;'}])

    def test_oversized_function_falls_back_and_preserves_budget(self):
        self.put('code.py', 'def uniqueworkflow():\n' + '    value = 123456789\n' * 900)
        kb = KnowledgeBase(workspace=self.root, context_budget=2000)
        raw = kb.retrieve('uniqueworkflow')
        result = json.loads(raw)
        self.assertTrue(result['results'])
        self.assertFalse(result['results'][0]['symbol_complete'])
        self.assertLessEqual(len(raw.encode()), 2000)
        self.assertEqual(result['context_bytes'], len(raw.encode()))
        kb.refresh()
        self.assertEqual(kb.context_budget, 2000)

    def test_skip_large_passage_and_continue_to_smaller_match(self):
        self.put('a.md', 'needle ' * 200 + '漢' * 400)
        self.put('b.md', 'needle\n')
        raw = KnowledgeBase(self.root, context_budget=2000).retrieve('needle')
        result = json.loads(raw)
        self.assertIn('user/b.md', [r['source'] for r in result['results']])
        self.assertLessEqual(len(raw.encode()), 2000)

    def test_budget_reserved_for_dependency_and_file_diversity(self):
        self.put('app.py', 'from lib import calculate\ndef uniqueworkflow():\n' + '    value = 1\n' * 80 + '    return calculate(value)\n')
        self.put('lib.py', 'def calculate(value):\n    return value + 7\n')
        self.put('guide.md', 'uniqueworkflow usage example')
        result = json.loads(KnowledgeBase(workspace=self.root, context_budget=4000).retrieve('uniqueworkflow'))
        self.assertIn('repo/lib.py', [r['source'] for r in result['results']])
        self.assertIn('repo/guide.md', [r['source'] for r in result['results']])

    def test_js_cycles_reexports_and_depth_are_bounded(self):
        self.put('main.ts', 'import { cost } from "./index";\nexport function uniqueworkflow() { return cost(); }')
        self.put('index.ts', 'export { price as cost } from "./price";')
        self.put('price.ts', 'import "./main";\nexport function price() { return 7; }')
        raw = KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow')
        result = json.loads(raw)
        self.assertIn('repo/price.ts', [r['source'] for r in result['results']])
        self.assertLessEqual(len(result['results']), 8)
        self.assertLessEqual(len(raw.encode()), 16000)

    def test_invalid_budgets_and_no_network(self):
        for budget in (100, 16001):
            with self.assertRaises(ValueError):
                KnowledgeBase(workspace=self.root, context_budget=budget)
        self.put('main.ts', 'import "./lib"; export function uniqueworkflow() {}')
        self.put('lib.ts', 'export default () => 7;')
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            self.assertIn('repo/lib.ts', KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))

    def test_cli_budget_is_passed_to_retrieval(self):
        with patch('offline_agent.cli.Ollama'), patch('offline_agent.cli.run', return_value=0) as agent:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--workspace', str(self.root), '--retrieval-budget', '4000', 'Explain']), 0)
        self.assertEqual(agent.call_args.kwargs['knowledge'].context_budget, 4000)

    def test_invalid_cli_budget_rejected_before_model_access(self):
        with patch('offline_agent.cli.Ollama') as model, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--retrieval-budget', '1', 'Explain'])
        model.assert_not_called()

    def test_complete_symbol_deduplicates_matching_chunks(self):
        self.put('code.py', 'def uniqueworkflow():\n' + '    uniqueworkflow_value = 1\n' * 60 + '    return uniqueworkflow_value\n')
        results = json.loads(KnowledgeBase(workspace=self.root).retrieve('uniqueworkflow'))['results']
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]['symbol_complete'])
