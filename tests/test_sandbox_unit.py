import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from offline_agent.test_runner import (command, container_command, docker_environment,
                                       inspect_image, local_socket, run_tests, snapshot)
from offline_agent.workspace import MAX_FILES, Workspace

IMAGE_ID = 'sha256:' + 'a' * 64


class SandboxUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tests').mkdir()
        (self.root / 'tests/test_ok.py').write_text('def test_ok(): assert True\n')
        self.ws = Workspace(self.root)

    def test_fixed_framework_commands(self):
        self.assertEqual(command()[1:3], ['-m', 'unittest'])
        self.assertEqual(command('pytest')[1:3], ['-m', 'pytest'])
        self.assertIn('addopts=', command('pytest'))
        with self.assertRaises(ValueError):
            command('shell')

    @patch('offline_agent.test_runner.shutil.which', return_value=None)
    @patch('offline_agent.test_runner.subprocess.Popen')
    def test_missing_docker_never_falls_back(self, launch, which):
        with self.assertRaisesRegex(ValueError, 'host execution is disabled'):
            run_tests(self.ws)
        launch.assert_not_called()

    def test_remote_socket_and_inherited_remote_context_rejected(self):
        for value in ('tcp://example.com:2375', 'relative.sock', 'https://example.com'):
            with self.assertRaises(ValueError):
                local_socket(value)
        with patch.dict(os.environ, {'DOCKER_HOST': 'tcp://example.com:2375',
                                    'DOCKER_CONTEXT': 'remote', 'HTTP_PROXY': 'secret-proxy'}):
            env = docker_environment()
        self.assertNotIn('DOCKER_HOST', env)
        self.assertNotIn('DOCKER_CONTEXT', env)
        self.assertNotIn('HTTP_PROXY', env)

    def test_security_flags_and_only_snapshot_mount(self):
        args = container_command(['docker', '--host', 'unix:///tmp/daemon.sock'], IMAGE_ID,
                                 'test-name', '/temporary/snapshot', 'pytest')
        for flag, value in [('--network', 'none'), ('--user', '65534:65534'),
                            ('--cap-drop', 'ALL'), ('--security-opt', 'no-new-privileges'),
                            ('--pull', 'never'), ('--log-driver', 'none')]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--read-only', args)
        self.assertEqual(args.count('--mount'), 1)
        self.assertEqual(args[args.index('--mount') + 1],
                         'type=bind,source=/temporary/snapshot,target=/workspace,readonly')
        self.assertNotIn(str(self.root), args)
        self.assertNotIn('--privileged', args)
        self.assertEqual(args[args.index('--entrypoint') + 1], 'python')

    @patch('offline_agent.test_runner.subprocess.run')
    def test_image_pinned_and_declared_volumes_refused(self, execute):
        execute.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({'Config': {}, 'Id': IMAGE_ID, 'Os': 'linux'}).encode())
        self.assertEqual(inspect_image(['docker'], 'local:tag', {}), IMAGE_ID)
        execute.return_value.stdout = json.dumps({'Config': {'Volumes': {'/data': {}}}, 'Id': IMAGE_ID, 'Os': 'linux'}).encode()
        with self.assertRaisesRegex(ValueError, 'VOLUME'):
            inspect_image(['docker'], 'local:tag', {})
        execute.return_value.returncode = 1
        with self.assertRaisesRegex(ValueError, 'never pulled'):
            inspect_image(['docker'], 'missing:tag', {})

    @unittest.skipIf(os.name == "nt", "isolated tests need macOS or Linux")
    def test_snapshot_filters_and_copies_binary_regular_files_only(self):
        (self.root / '.gitignore').write_text('private/\n')
        (self.root / 'private').mkdir()
        (self.root / 'private/secret.py').write_text('SECRET')
        (self.root / 'fixture.bin').write_bytes(b'\0binary')
        (self.root / 'linked.py').symlink_to(self.root / 'private/secret.py')
        with tempfile.TemporaryDirectory() as destination:
            count = snapshot(Workspace(self.root), Path(destination), 'pytest')
            self.assertEqual(count, 2)
            self.assertFalse(Path(destination, 'private').exists())
            self.assertFalse(Path(destination, 'linked.py').exists())
            self.assertEqual(Path(destination, 'fixture.bin').read_bytes(), b'\0binary')

    def test_excluded_or_missing_tests_never_launch(self):
        (self.root / '.agentignore').write_text('tests/\n')
        with tempfile.TemporaryDirectory() as destination:
            with self.assertRaises(ValueError):
                snapshot(Workspace(self.root), Path(destination), 'pytest')

    def test_snapshot_limit_refuses_partial_project(self):
        with patch.object(self.ws, 'files', return_value=['tests/test_ok.py'] * MAX_FILES):
            with tempfile.TemporaryDirectory() as destination, self.assertRaisesRegex(ValueError, 'file limit'):
                snapshot(self.ws, Path(destination), 'pytest')

    @patch('offline_agent.test_runner.shutil.which', return_value='/bin/docker')
    @patch('offline_agent.test_runner.local_socket', return_value=Path('/tmp/docker.sock'))
    @patch('offline_agent.test_runner.inspect_image', return_value=IMAGE_ID)
    @patch('offline_agent.test_runner.snapshot', return_value=1)
    @patch('offline_agent.test_runner.os.read', side_effect=[b'collected output', b''])
    @patch('offline_agent.test_runner.subprocess.Popen')
    @patch('offline_agent.test_runner.subprocess.run')
    def test_timeout_removes_container_and_client_configuration_is_clean(self, execute, launch, read,
                                                                          copy, inspect, socket, which):
        execute.return_value = SimpleNamespace(returncode=0)
        process = launch.return_value
        process.wait.side_effect = [subprocess.TimeoutExpired('docker start', 1), 137]
        process.poll.return_value = None
        process.returncode = 137
        def verify_config(base, image, env):
            config = Path(base[base.index('--config') + 1], 'config.json')
            self.assertEqual(json.loads(config.read_text()), {'proxies': {}})
            return IMAGE_ID
        inspect.side_effect = verify_config
        result = run_tests(self.ws, runner='pytest', timeout=1, output_limit=5)
        self.assertIn('Timed out', result)
        self.assertIn('Output truncated', result)
        self.assertIn('rm', execute.call_args.args[0])
        self.assertIn('-f', execute.call_args.args[0])
        self.assertIn('create', execute.call_args_list[0].args[0])
        self.assertIn('start', launch.call_args.args[0])
        process.kill.assert_called_once()

    @patch('offline_agent.test_runner.shutil.which', return_value='/bin/docker')
    @patch('offline_agent.test_runner.local_socket', return_value=Path('/tmp/docker.sock'))
    @patch('offline_agent.test_runner.inspect_image', side_effect=ValueError('image missing'))
    @patch('offline_agent.test_runner.subprocess.Popen')
    def test_failed_preflight_never_starts_tests(self, launch, inspect, socket, which):
        with self.assertRaisesRegex(ValueError, 'image missing'):
            run_tests(self.ws)
        launch.assert_not_called()

    def test_mount_separator_cannot_change_mount_options(self):
        with self.assertRaises(ValueError):
            container_command(['docker'], IMAGE_ID, 'test', '/tmp/name,with,commas', 'pytest')
