"""Shared fixtures for BGR test suite."""
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning, message=".*doesn't match a supported version.*")
import pytest
import requests

TEST_PORT = 8001
BASE_URL = f"http://localhost:{TEST_PORT}"
# Small PDF used for ingest tests — 1-page, fast to process
TEST_PDF = "examples/3c-NT.pdf"
TEST_RULEBOOK_ID = "test-3cnt"

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
UVICORN_BIN = BACKEND_DIR / ".venv" / "bin" / "uvicorn"


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll GET url until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires backend at localhost:8001")
    config.addinivalue_line("markers", "ingest: uploads a PDF to the backend")
    config.addinivalue_line("markers", "frontend: requires Node.js + frontend dir")


@pytest.fixture(scope="session")
def backend(tmp_path_factory):
    """
    Start an isolated test backend on port 8001 with a temporary data directory.

    If port 8001 is already in use, assumes an existing test server and skips launching.
    The backend's CWD is set to a temp directory so that ``Path("./data")`` in
    storage.py and main.py resolves to an isolated location.
    """
    if _port_in_use(TEST_PORT):
        # Assume an existing test server is running
        yield BASE_URL
        return

    # Create a temp working directory for the server — data/ will be created here
    work_dir = tmp_path_factory.mktemp("bgr_test_server")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    # Override the .env path so main.py's load_dotenv still finds the API key.
    # main.py loads from Path(__file__).parent.parent / ".env" which resolves
    # to the real project root — this works since we don't move the source files.

    proc = subprocess.Popen(
        [
            str(UVICORN_BIN),
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(TEST_PORT),
            "--log-level", "warning",
        ],
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not _wait_for_server(f"{BASE_URL}/rulebooks", timeout=30):
            # Grab stderr for diagnostics
            proc.terminate()
            _, stderr = proc.communicate(timeout=5)
            pytest.skip(
                f"Test backend failed to start on port {TEST_PORT}.\n"
                f"stderr: {stderr.decode()[-500:]}"
            )

        yield BASE_URL
    finally:
        # Graceful shutdown
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        # Clean up test data
        data_dir = work_dir / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def ingested_rulebook(backend, project_root):
    """
    Ingest the test PDF once per session and return its rulebook_id.
    Uses a stable test ID so it won't collide with real rulebooks.
    """
    pdf_path = project_root / TEST_PDF
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{backend}/rulebooks/{TEST_RULEBOOK_ID}",
            params={"source_type": "core", "game_name": "Test 3C NT"},
            files={"file": ("3c-NT.pdf", f, "application/pdf")},
            stream=True,
            timeout=300,
        )
    resp.raise_for_status()

    # Consume the SSE stream and capture the final done event
    done_event = None
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
        if line.startswith("data:"):
            payload = json.loads(line[len("data:"):].strip())
            if payload.get("done"):
                done_event = payload
                break

    assert done_event is not None, "SSE stream never sent a done event"
    return TEST_RULEBOOK_ID, done_event


def parse_ask_sse(resp):
    """
    Parse an SSE response from GET /ask.
    Returns (results, answer_text) where:
      - results: list from the first 'results' event
      - answer_text: concatenated token strings
    """
    results = None
    tokens = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload = json.loads(line[len("data:"):].strip())
        if payload.get("type") == "results":
            results = payload.get("results", [])
        elif payload.get("type") == "token":
            tokens.append(payload.get("text", ""))
        elif payload.get("type") == "done":
            break
    return results, "".join(tokens)


@pytest.fixture(scope="session")
def project_root():
    # tests/ lives one level inside the project root
    return PROJECT_ROOT
