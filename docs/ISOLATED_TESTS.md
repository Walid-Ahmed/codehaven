# Isolated unittest and pytest execution

The run_tests tool now requires a local Docker Linux-container runtime. It never
runs project tests directly on your device. If the daemon, local image, or required
isolation settings are unavailable, the tool reports an error without falling back
to a host subprocess. The selected framework and image are user configuration;
the model cannot supply arbitrary commands, mounts, image names, or arguments.

## Prepare once

Install/start Docker Desktop on macOS, or a local Docker Engine on Linux. From
this repository, prepare the test image while online:

```sh
docker build -t offline-agent-tests:0.7.0 sandbox
```

This setup step downloads Python and pytest. The agent never builds or pulls images
while running tests; it uses an installed image ID with pulling disabled. The
prepared image supports standard-library unittest and pytest. Project dependencies
must already be installed in the image, not just in your host virtual environment.

## Run either framework

```sh
# Standard-library unittest discovery (default):
offline-agent --workspace /path/to/project 'Run tests and explain failures'

# pytest:
offline-agent --workspace /path/to/project --test-runner pytest \
  'Run the tests and explain failures'
```

Each run asks for approval. CodeHaven shows the
fixed command, selected image, and workspace first. The commands inside the
container are:

```sh
python -m unittest discover -s tests -v
python -m pytest -q -p no:cacheprovider -o addopts= tests
```

A permitted tests/ directory and discoverable test files are required. Pytest
plugin auto-discovery, environment addopts, config addopts, and disk cache writing
are disabled. conftest.py and pytest configuration can still affect tests inside
the sandbox. Explicit plugins in conftest.py must already exist in the image.
The tool returns the exit code and bounded output; agent exit code 0 means its
conversation finished, not that tests passed. Check the test count as well.

## What is isolated

- Docker networking is disabled. Container-local loopback remains; there is no
  external network interface/route or access to the host's Ollama server.
- The tool copies permitted project files into a fresh staging directory and
  mounts that copy read-only at /workspace. The original project, host home,
  credentials, Docker socket, and external reference libraries are not mounted.
- The container root filesystem is read-only. /tmp is a 64 MB temporary writable
  filesystem; Docker also provides small ephemeral device/shared-memory mounts.
  Tests should use tempfile/tmp_path instead of writing into the project.
- Execution uses UID/GID 65534, drops all Linux capabilities, enables
  no-new-privileges, and disables image healthchecks. The image entrypoint is
  overridden with Python. Images declaring VOLUME mounts are rejected.
- Limits: one CPU, 512 MB RAM, 128 processes, 256 open files, no core dumps, 16 MB
  shared memory, a 30-second test timeout, and 16,000 retained output bytes.
  Docker daemon logging is disabled for these containers; excess streamed output
  is drained rather than accumulated.
- Host environment variables and Docker proxy settings are not passed into the
  container. A clean temporary Docker client config and an explicit local Unix
  socket prevent inherited remote Docker contexts from selecting a remote daemon.
- The container is created before starting tests. Cleanup forcibly removes that
  unique container, including detached subprocesses, on completion, timeout, or
  cancellation. A cleanup failure reports the container name for manual removal;
  it does not return a successful test result.

The snapshot applies the same .gitignore, .agentignore, CLI exclusions, hidden-path,
symlink, and per-file size restrictions as the file tools. Small binary fixtures
are supported. Discovered files are read using non-following directory-relative
file descriptors. No symlinks or special files are copied. At most 999 permitted
files and 16 MB can be staged; reaching the listing cap refuses execution rather
than silently taking a partial 1000-file listing. Files above the existing 64 KB
listing limit are omitted. Missing fixtures/dependencies should produce failures,
not be interpreted as proof the whole project works.

Filtering and snapshot preparation happen before test execution; the 30-second
limit applies to the attached test process, with separate bounded Docker
preflight/create/cleanup operations. Run against a stable local project rather
than concurrently mutating its directory tree.

## Project dependencies and custom images

Prepare a separate minimal image containing the project's Python dependencies and
pytest, then select it explicitly:

```sh
offline-agent --workspace /path/to/project --test-runner pytest \
  --test-image my-project-tests:local 'Run tests'
```

For example, in a separate image-build directory, put the dependency list in
requirements-tests.txt and use this Dockerfile:

```dockerfile
FROM offline-agent-tests:0.7.0
COPY requirements-tests.txt /opt/test-dependencies.txt
RUN python -m pip install --no-cache-dir -r /opt/test-dependencies.txt
```

Build it once while online with `docker build -t my-project-tests:local .`.
Use a trusted image with no secrets or baked-in project copies: exclusions filter
the staged files, not data already present in an image. Image tags are resolved to
an immutable local image ID before execution. No automatic package installation
or test-framework detection occurs at runtime. Python/OS dependencies, native
extensions, and custom pytest plugins must be prepared in the image while online.

Only local Unix Docker sockets are supported. Common Docker Desktop, rootless
Docker, and Colima socket locations are detected. If needed:

```sh
offline-agent --workspace /path/to/project \
  --test-docker-socket /absolute/path/to/docker.sock 'Run tests'
```

TCP/SSH Docker endpoints and inherited DOCKER_HOST/DOCKER_CONTEXT are not used.
The isolated tool currently supports macOS/Linux with Linux containers. Other
agent tools remain available on Windows. Container isolation depends on the
trusted local Docker engine and its kernel; this is not a separate hardened VM
or a guarantee against container-runtime vulnerabilities.

## Verification

```sh
python -m unittest discover -s tests -v
OFFLINE_AGENT_SANDBOX_INTEGRATION=1 python -m unittest discover \
  -s tests -p test_sandbox_integration.py -v
```

The second command requires the daemon and prepared image and performs real
container checks for unittest/pytest pass/fail results, network/filesystem and
host-environment boundaries, writable temp files, output truncation, timeout,
and cleanup of detached children. Without the opt-in variable those tests are
explicitly skipped; no host substitute is used. CI has a dedicated image-build
and isolated integration job.

Docker settings follow the [official container CLI reference](https://docs.docker.com/reference/cli/docker/container/create/).
