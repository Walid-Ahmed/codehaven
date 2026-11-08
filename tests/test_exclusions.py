import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.cli import main
from offline_agent.exclusions import IgnoreConfigurationError
from offline_agent.knowledge import KnowledgeBase
from offline_agent.workspace import Workspace


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, text='uniqueexcludedtoken'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def test_shared_policy_for_listing_search_index_and_direct_access(self):
        self.put('.gitignore', 'private/\n*.log\n')
        self.put('.agentignore', 'generated/\n')
        for name in ('private/secret.py', 'generated/code.py', 'skip.py', 'debug.log', 'keep.py'):
            self.put(name)
        ws = Workspace(self.root, excludes=['skip.py'])
        self.assertEqual(ws.files(), ['keep.py'])
        self.assertEqual([m['path'] for m in json.loads(ws.search('uniqueexcludedtoken'))['matches']], ['keep.py'])
        kb = KnowledgeBase(workspace=ws)
        self.assertEqual([r['source'] for r in json.loads(kb.retrieve('uniqueexcludedtoken'))['results']], ['repo/keep.py'])
        for name in ('private/secret.py', 'generated/code.py', 'skip.py', 'debug.log'):
            with self.subTest(name=name):
                for operation in (lambda: ws.read(name), lambda: ws.search('token', name)):
                    with self.assertRaises(ValueError):
                        operation()

    def test_nested_gitignore_overrides_parent_file_rule(self):
        self.put('.gitignore', '*.log\n')
        self.put('src/.gitignore', '!keep.log\n')
        self.put('src/keep.log'); self.put('src/drop.log'); self.put('keep.log')
        self.assertEqual(Workspace(self.root).files(), ['src/keep.log'])

    def test_cannot_reinclude_child_of_ignored_directory(self):
        self.put('.gitignore', 'private/\n!private/keep.py\n')
        self.put('private/.gitignore', '!keep.py\n')
        self.put('private/keep.py')
        ws = Workspace(self.root)
        self.assertEqual(ws.files(), [])
        with self.assertRaises(ValueError):
            ws.files('private')

    def test_directory_children_can_be_reincluded_when_parent_allowed(self):
        self.put('.gitignore', 'data/*\n!data/keep.py\n')
        self.put('data/keep.py'); self.put('data/drop.py')
        self.assertEqual(Workspace(self.root).files(), ['data/keep.py'])

    def test_root_anchoring_double_star_comments_and_escapes(self):
        self.put('.gitignore', '# comment\n/root.py\n**/temp?.py\n\\#literal.py\n\\!literal.py\n')
        for name in ('root.py', 'sub/root.py', 'temp1.py', 'sub/temp2.py', '#literal.py', '!literal.py', 'safe.py'):
            self.put(name)
        self.assertEqual(Workspace(self.root).files(), ['safe.py', 'sub/root.py'])

    def test_directory_only_pattern_does_not_exclude_same_named_file(self):
        self.put('.gitignore', 'cache/\n')
        self.put('cache')
        self.assertEqual(Workspace(self.root).files(), ['cache'])

    def test_agentignore_nested_rules_and_layer_priority(self):
        self.put('.gitignore', '*.log\n')
        self.put('.agentignore', '*.tmp\n!keep.tmp\n!keep.log\n')
        self.put('src/.agentignore', '!local.tmp\n')
        for name in ('keep.tmp', 'drop.tmp', 'keep.log', 'src/local.tmp', 'src/drop.tmp'):
            self.put(name)
        self.assertEqual(Workspace(self.root).files(), ['keep.tmp', 'src/local.tmp'])
        self.assertEqual(Workspace(self.root, excludes=['*.tmp']).files(), [])

    def test_negation_cannot_override_builtins_or_hidden_paths(self):
        self.put('.gitignore', '!node_modules/\n!node_modules/code.py\n!.env\n')
        self.put('node_modules/code.py'); self.put('.env')
        self.assertEqual(Workspace(self.root).files(), [])

    def test_cli_repeated_excludes_are_passed_to_all_tools(self):
        for name in ('a.py', 'b.py', 'keep.py'):
            self.put(name)
        with patch('offline_agent.cli.Ollama'), patch('offline_agent.cli.run', return_value=0) as agent:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--workspace', str(self.root), '--exclude', 'a.py', '--exclude', 'b.py', 'Explain']), 0)
        self.assertEqual(agent.call_args.args[1].files(), ['keep.py'])
        self.assertNotIn('repo/a.py', agent.call_args.kwargs['knowledge'].retrieve('uniqueexcludedtoken'))

    def test_excluded_import_target_not_in_dependency_context(self):
        self.put('.gitignore', 'secret.py\n')
        self.put('app.py', 'import secret\ndef uniqueentrypoint(): pass\n')
        self.put('secret.py', 'sensitive = 42')
        result = json.loads(KnowledgeBase(workspace=Workspace(self.root)).retrieve('uniqueentrypoint'))
        self.assertEqual([r['source'] for r in result['results']], ['repo/app.py'])

    def test_reference_root_cannot_bypass_repository_ignore(self):
        self.put('.gitignore', 'knowledge/\n')
        self.put('knowledge/team.md')
        kb = KnowledgeBase(self.root / 'knowledge', workspace=Workspace(self.root))
        self.assertEqual(json.loads(kb.retrieve('uniqueexcludedtoken'))['results'], [])
        self.assertEqual(kb.excluded_reference_roots, 1)

    def test_reference_files_in_nested_root_inherit_repository_rules(self):
        self.put('.gitignore', 'knowledge/private.md\n')
        self.put('knowledge/private.md'); self.put('knowledge/public.md', 'publicreference')
        kb = KnowledgeBase(self.root / 'knowledge', workspace=Workspace(self.root))
        self.assertEqual(json.loads(kb.retrieve('uniqueexcludedtoken'))['results'], [])
        self.assertIn('user/public.md', kb.retrieve('publicreference'))

    def test_external_library_has_own_rules_and_cli_exclusions(self):
        with tempfile.TemporaryDirectory() as external:
            Path(external, '.gitignore').write_text('private.md\n')
            Path(external, 'private.md').write_text('uniqueexcludedtoken')
            Path(external, 'skip.md').write_text('uniqueexcludedtoken')
            Path(external, 'keep.md').write_text('publicreference')
            kb = KnowledgeBase(external, workspace=Workspace(self.root, excludes=['skip.md']))
            self.assertEqual(json.loads(kb.retrieve('uniqueexcludedtoken'))['results'], [])
            self.assertIn('user/keep.md', kb.retrieve('publicreference'))

    def test_refresh_preserves_exclusions_and_reloads_ignore_files(self):
        self.put('a.py'); self.put('b.py'); self.put('c.py')
        kb = KnowledgeBase(workspace=Workspace(self.root, excludes=['a.py']))
        self.put('.gitignore', 'b.py\n')
        kb.refresh()
        self.assertEqual([r['source'] for r in json.loads(kb.retrieve('uniqueexcludedtoken'))['results']], ['repo/c.py'])

    def test_symlinked_ignore_file_not_followed(self):
        self.put('outside.txt', '*.py\n')
        (self.root / '.gitignore').symlink_to(self.root / 'outside.txt')
        self.put('keep.py')
        self.assertIn('keep.py', Workspace(self.root).files())

    def test_invalid_configuration_fails_closed(self):
        (self.root / '.gitignore').write_bytes(b'\xff')
        with self.assertRaises(IgnoreConfigurationError):
            Workspace(self.root)
        self.put('.gitignore', 'x' * 64001)
        with self.assertRaises(IgnoreConfigurationError):
            Workspace(self.root)
        (self.root / '.gitignore').unlink()
        self.put('src/.agentignore', 'x' * 64001)
        self.put('src/a.py')
        with self.assertRaises(IgnoreConfigurationError):
            Workspace(self.root).files()

    def test_invalid_cli_exclusion_patterns_rejected(self):
        for patterns in ([''], ['!safe.py'], ['a\nb']):
            with self.assertRaises(IgnoreConfigurationError):
                Workspace(self.root, excludes=patterns)

    def test_exclusions_do_not_use_network_or_git_subprocess(self):
        self.put('.gitignore', 'skip.py\n'); self.put('skip.py'); self.put('keep.py')
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
             patch('subprocess.Popen', side_effect=AssertionError('Subprocess forbidden')):
            self.assertEqual(Workspace(self.root).files(), ['keep.py'])

    @unittest.skipUnless(os.name == 'posix', 'POSIX FIFO fixture')
    def test_nonregular_config_rejected_without_reading(self):
        os.mkfifo(self.root / '.gitignore')
        with self.assertRaises(IgnoreConfigurationError):
            Workspace(self.root)

    @unittest.skipUnless(shutil.which('git'), 'Optional comparison against installed Git')
    def test_git_pattern_results_match_git_check_ignore(self):
        self.put('.gitignore', '*.log\n!keep.log\ncache/*\n!cache/keep.py\n/root.py\nsecret/\n!secret/keep.py\n')
        self.put('src/.gitignore', '!local.log\n')
        names = ['drop.log', 'keep.log', 'src/local.log', 'src/drop.log', 'cache/drop.py',
                 'cache/keep.py', 'root.py', 'src/root.py', 'secret/keep.py', 'safe.py']
        for name in names:
            self.put(name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        result = subprocess.run(['git', '-C', str(self.root), 'check-ignore', '--no-index', '-z', '--stdin'],
                                input=('\0'.join(names) + '\0').encode(), capture_output=True, check=False)
        self.assertIn(result.returncode, (0, 1))
        ignored = {name.decode() for name in result.stdout.split(b'\0') if name}
        self.assertEqual(set(Workspace(self.root).files()), set(names) - ignored)
