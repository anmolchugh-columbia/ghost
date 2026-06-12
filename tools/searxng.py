import os
import subprocess
import time
from pathlib import Path

import requests

_COLIMA = "/opt/homebrew/bin/colima"
_DOCKER = "/opt/homebrew/bin/docker"
_SETTINGS = str(Path(__file__).parent.parent / "searxng" / "settings.yml")
_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def start() -> None:
    # Start Colima VM if not running
    if "Running" not in _run([_COLIMA, "status"]).stdout:
        print("[searxng] starting Colima...", flush=True)
        subprocess.run([_COLIMA, "start"], check=True)

    # Start or create SearXNG container
    container_status = _run([
        _DOCKER, "ps", "-a", "--filter", "name=^searxng$", "--format", "{{.Status}}"
    ]).stdout.strip()

    if not container_status:
        print("[searxng] creating container...", flush=True)
        _run([_DOCKER, "run", "-d", "--name", "searxng", "-p", "8080:8080",
              "-v", f"{_SETTINGS}:/etc/searxng/settings.yml:ro", "searxng/searxng"])
    elif container_status.startswith("Up"):
        print("[searxng] already running", flush=True)
        return
    else:
        print("[searxng] starting container...", flush=True)
        _run([_DOCKER, "start", "searxng"])

    # Wait for SearXNG to accept requests
    for _ in range(20):
        try:
            if requests.get(f"{_URL}/search?q=ping&format=json", timeout=2).status_code == 200:
                print("[searxng] ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("SearXNG did not become ready within 20s")


def stop() -> None:
    print("[searxng] stopping container...", flush=True)
    _run([_DOCKER, "stop", "searxng"])
    print("[searxng] stopping Colima...", flush=True)
    _run([_COLIMA, "stop"])
