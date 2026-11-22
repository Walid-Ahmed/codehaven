#!/usr/bin/env python3
"""Start a foreground, loopback-only Ollama server with cloud disabled."""
import argparse
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys


def existing_ollama():
    """Probe only numeric loopback; no proxies, redirects, or downloads."""
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=2)
    try:
        connection.request("GET", "/api/tags")
        response = connection.getresponse()
        raw = response.read(64_001)
        if response.status != 200 or len(raw) > 64_000:
            return False
        data = json.loads(raw)
        return isinstance(data, dict) and isinstance(data.get("models"), list)
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def main(argv=None):
    """Start `ollama serve` on loopback with cloud features off, unless a server already answers. Returns an exit code."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    executable = shutil.which("ollama")
    if executable is None:
        print("Error: Ollama is not installed or is missing from PATH. Install it from https://ollama.com/.",
              file=sys.stderr)
        return 1
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            occupied = True
    except ConnectionRefusedError:
        occupied = False
    except OSError as exc:
        print(f"Error: Cannot check the local Ollama port: {exc}", file=sys.stderr)
        return 1
    if occupied:
        if not existing_ollama():
            print("Error: Port 11434 is occupied, but the listener did not respond as Ollama. "
                  "No server was started or stopped.", file=sys.stderr)
            return 1
        print("Ollama is already running at http://127.0.0.1:11434. No second server is needed.\n"
              "Its cloud setting is unchanged and has not been verified. To enforce OLLAMA_NO_CLOUD=1, "
              "quit the existing Ollama app/service and run this script again.\n"
              "Next: run codehaven --doctor in your activated environment.")
        return 0
    env = os.environ.copy()
    env["OLLAMA_NO_CLOUD"] = "1"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    print("Starting Ollama with cloud disabled on 127.0.0.1:11434. Press Ctrl+C to stop.\n"
          "Check the server logs for: Ollama cloud disabled: true", flush=True)
    try:
        return subprocess.run([executable, "serve"], env=env, check=False).returncode
    except OSError as exc:
        print(f"Error: Could not start Ollama: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
