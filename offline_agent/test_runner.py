"""Fail-closed Docker test execution with filtered read-only project snapshots."""
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from .workspace import MAX_BYTES, MAX_FILES

TIMEOUT = 30
OUTPUT_LIMIT = 16_000
DEFAULT_IMAGE = 'offline-agent-tests:0.7.0'


def command(runner='unittest'):
    if runner == 'unittest':
        return ['python', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    if runner == 'pytest':
        return ['python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'addopts=', 'tests']
    raise ValueError('Test runner must be unittest or pytest')


def local_socket(configured=None):
    if os.name != 'posix':
        raise ValueError('Isolated tests require Docker Linux containers on macOS/Linux')
    candidates = [Path(configured).expanduser()] if configured else [
        Path('/var/run/docker.sock'), Path.home() / '.docker/run/docker.sock',
        Path.home() / '.colima/default/docker.sock', Path(f'/run/user/{os.getuid()}/docker.sock')]
    for path in candidates:
        if not path.is_absolute():
            raise ValueError('Docker socket must be an absolute local Unix socket path')
        if path.is_socket():
            return path
    raise ValueError('Local Docker socket unavailable. Start Docker; host test execution is disabled.')


def docker_environment():
    # Avoid remote contexts, proxy injection, TLS overrides, and client plugins/config.
    return {key: value for key, value in os.environ.items() if key in {'PATH', 'HOME', 'SYSTEMROOT'}}


def inspect_image(base, image, env):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,255}', image):
        raise ValueError('Invalid local test image name')
    try:
        result = subprocess.run(base + ['image', 'inspect', '--format', '{{json .}}', image],
                                env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Docker preflight timed out; tests were not run') from exc
    if result.returncode:
        raise ValueError('Docker or local test image unavailable. Start Docker and prepare the image. '
                         'Tests were not run; images are never pulled by the agent.')
    try:
        metadata = json.loads(result.stdout)
        if not isinstance(metadata, dict) or not isinstance(metadata.get('Config'), dict):
            raise ValueError('Invalid Docker image inspection response')
        identity = metadata.get('Id', '')
        if metadata['Config'].get('Volumes') or metadata.get('Os') != 'linux':
            raise ValueError('Test image must be Linux and declare no VOLUME mounts')
        if not re.fullmatch(r'sha256:[a-f0-9]{64}', identity):
            raise ValueError('Docker returned an invalid local image ID')
        return identity
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid Docker image inspection response') from exc


def snapshot(workspace, destination, runner):
    tests = workspace.path('tests')
    if not tests.is_dir():
        raise ValueError('A permitted tests/ directory is required')
    names = workspace.files()
    if len(names) >= MAX_FILES:
        raise ValueError('Snapshot file limit reached; narrow the workspace or add exclusions')
    def is_test(name):
        file = Path(name)
        return (name.startswith('tests/') and file.suffix == '.py' and
                (file.name.startswith('test') if runner == 'unittest' else
                 file.name.startswith('test_') or file.name.endswith('_test.py')))
    if not any(is_test(name) for name in names):
        raise ValueError('No permitted test files found for the selected runner')
    total = 0
    with_fd = os.open(workspace.root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names:
            workspace.path(name)  # Enforce the same ignore policy again before copying.
            parent = os.dup(with_fd)
            try:
                parts = Path(name).parts
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    os.close(parent)
                    parent = child
                file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(file_fd, 'rb') as handle:
                    info = os.fstat(handle.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError(f'Snapshot path is not a regular file: {name}')
                    data = handle.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError(f'Snapshot file changed or exceeds 64 KB: {name}')
            finally:
                os.close(parent)
            total += len(data)
            if total > 16_000_000:
                raise ValueError('Snapshot exceeds 16 MB; add exclusions or narrow the workspace')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            for directory in target.parents:
                if directory == destination.parent:
                    break
                directory.chmod(0o755)
            target.write_bytes(data)
            target.chmod(0o755 if info.st_mode & 0o111 else 0o644)
    finally:
        os.close(with_fd)
    destination.chmod(0o755)
    return len(names)


def container_command(base, image_id, name, source, runner):
    if ',' in str(source):
        raise ValueError('Snapshot path cannot contain commas in Docker mount syntax')
    return base + [
        'create', '--name', name, '--pull', 'never', '--network', 'none', '--no-healthcheck',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', '65534:65534', '--pids-limit', '128', '--memory', '512m', '--cpus', '1',
        '--ulimit', 'nofile=256:256', '--ulimit', 'core=0:0', '--log-driver', 'none',
        '--shm-size', '16m', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777',
        '--mount', f'type=bind,source={source},target=/workspace,readonly',
        '--workdir', '/workspace', '--env', 'HOME=/tmp', '--env', 'TMPDIR=/tmp',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONPATH=/workspace',
        '--env', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '--env', 'PYTEST_ADDOPTS=',
        '--env', 'PIP_NO_INDEX=1', '--entrypoint', 'python', image_id,
    ] + command(runner)[1:]


def run_tests(workspace, *, runner='unittest', image=DEFAULT_IMAGE, docker_socket=None,
              timeout=TIMEOUT, output_limit=OUTPUT_LIMIT):
    command(runner)
    if not 0 < timeout <= 120 or not 0 < output_limit <= OUTPUT_LIMIT:
        raise ValueError('Test timeout/output limit is out of range')
    executable = shutil.which('docker')
    if executable is None:
        raise ValueError('Docker is required for isolated tests; host execution is disabled')
    socket = local_socket(docker_socket)
    env = docker_environment()
    with tempfile.TemporaryDirectory(prefix='offline-agent-tests-') as temporary:
        root = Path(temporary)
        config = root / 'client'
        config.mkdir()
        (config / 'config.json').write_text('{"proxies": {}}')
        base = [executable, '--config', str(config), '--host', 'unix://' + str(socket)]
        identity = inspect_image(base, image, env)
        source = root / 'project'
        source.mkdir()
        count = snapshot(workspace, source, runner)
        # Docker Desktop may need to traverse the parent directory to share the snapshot.
        root.chmod(0o755)
        name = 'offline-agent-tests-' + uuid.uuid4().hex
        captured = bytearray()
        truncated = threading.Event()
        timed_out = False
        process = None
        reader = None
        try:
            try:
                created = subprocess.run(container_command(base, identity, name, source, runner),
                                         env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('Container creation timed out; no test process was started') from exc
            if created.returncode:
                raise ValueError('Cannot create the isolated container: ' + created.stderr.decode(errors='replace')[:1000])
            process = subprocess.Popen(base + ['start', '--attach', name],
                                       env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            def collect():
                while True:
                    chunk = os.read(process.stdout.fileno(), 8192)
                    if not chunk:
                        break
                    remaining = output_limit - len(captured)
                    captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated.set()
            reader = threading.Thread(target=collect, daemon=True)
            reader.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            # Kill the container itself, including detached test subprocesses, not only the CLI.
            try:
                cleanup = subprocess.run(base + ['rm', '-f', name], env=env, stdin=subprocess.DEVNULL,
                                         capture_output=True, timeout=10)
                if cleanup.returncode and b'No such container' not in cleanup.stderr:
                    raise ValueError(f'Container cleanup failed; inspect/remove {name} before continuing')
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f'Container cleanup timed out; inspect/remove {name}') from exc
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    if reader is not None and reader.ident is not None:
                        reader.join(timeout=2)
                    process.stdout.close()
        text = bytes(captured).decode('utf-8', errors='replace')
        status = f'Timed out after {timeout} seconds' if timed_out else f'Exit code: {process.returncode}'
        if truncated.is_set():
            text += f'\n[Output truncated to {output_limit} bytes]'
        return f'Isolation: Docker; runner: {runner}; snapshot files: {count}\n' + status + '\n' + text
