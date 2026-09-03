"""Container smoke test.

Builds the image, boots it, probes the same endpoints the deployment relies on,
and tears it down. Skipped — never failed — when Docker is unavailable, so a
laptop or a CI runner without a daemon still gets a green suite.

Run explicitly with `make docker-smoke`. It is excluded from the default run by
the `container` marker because a cold build takes minutes.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ml-service-blueprint:pytest"
CONTAINER = "ml-service-blueprint-pytest"
BOOT_TIMEOUT_SECONDS = 90

pytestmark = pytest.mark.container


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False)
    return result.returncode == 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _post(url: str, payload: dict, timeout: float = 15.0) -> tuple[int, dict]:
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


@pytest.fixture(scope="module")
def container_base_url() -> str:
    if not _docker_available():
        pytest.skip("Docker daemon is not available")
    if not (REPO_ROOT / "registry").is_dir():
        pytest.skip("no registry/ to bake into the image; run `make train-promote` first")

    build = subprocess.run(
        ["docker", "build", "-t", IMAGE, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"docker build failed:\n{build.stderr[-4000:]}")

    port = _free_port()
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER,
            "-p",
            f"127.0.0.1:{port}:8000",
            IMAGE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        pytest.fail(f"docker run failed:\n{run.stderr}")

    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                if _get(f"{base_url}/ready", timeout=2.0)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(1.0)
        else:
            logs = subprocess.run(
                ["docker", "logs", CONTAINER], capture_output=True, text=True, check=False
            )
            pytest.fail(f"container never became ready:\n{logs.stdout}\n{logs.stderr}")
        yield base_url
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)


def test_container_becomes_ready(container_base_url):
    status, body = _get(f"{container_base_url}/ready")
    assert status == 200
    assert body["model_loaded"] is True


def test_container_serves_predictions(container_base_url):
    _, info = _get(f"{container_base_url}/model-info")
    status, body = _post(f"{container_base_url}/predict", {"instances": [info["example_instance"]]})
    assert status == 200
    assert body["count"] == 1


def test_container_runs_as_a_non_root_user(container_base_url):
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "id", "-u"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() != "0"


def test_container_reports_a_healthy_healthcheck(container_base_url):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", CONTAINER],
            capture_output=True,
            text=True,
            check=True,
        )
        state = result.stdout.strip()
        if state == "healthy":
            return
        if state == "unhealthy":
            pytest.fail("HEALTHCHECK reported unhealthy")
        time.sleep(2)
    pytest.fail("HEALTHCHECK never reported healthy")


def test_container_logs_are_json(container_base_url):
    _get(f"{container_base_url}/health")
    logs = subprocess.run(
        ["docker", "logs", CONTAINER], capture_output=True, text=True, check=False
    ).stdout
    parsed = [
        json.loads(line)
        for line in logs.splitlines()
        if line.startswith("{") and line.rstrip().endswith("}")
    ]
    assert parsed, f"no JSON log lines found in container output:\n{logs[:2000]}"
    assert "level" in parsed[0] and "message" in parsed[0]
