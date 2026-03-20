"""Shared fixtures for BGR test suite."""
import json
import os
import shutil
import signal
import socket
import subprocess
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

# Rulebook registry: (rulebook_id, pdf_path_relative_to_project_root, game_name)
RULEBOOKS = {
    "test-3cnt": ("examples/3c-NT.pdf", "Test 3C NT"),
    "ACBoV":     ("examples/AC.pdf", "Assassin's Creed Brotherhood of Venice"),
    "f":         ("examples/f_rulebook.pdf", "Fromage"),
    "LOG":       ("examples/log.pdf", "Lands of Galzyr"),
}


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


def _ingest_rulebook(base_url: str, rulebook_id: str, source_type: str = "core"):
    """
    Ingest a PDF into the test backend. Skips if already ingested or PDF missing.
    Returns the done event dict.
    """
    pdf_rel, game_name = RULEBOOKS[rulebook_id]
    pdf_path = PROJECT_ROOT / pdf_rel
    if not pdf_path.exists():
        pytest.skip(f"PDF not found: {pdf_path}")

    # Check if already ingested (avoids re-ingest on repeated runs against a persistent server)
    resp = requests.get(f"{base_url}/rulebooks", timeout=5)
    existing_ids = [rb["id"] for rb in resp.json().get("rulebooks", [])]
    if rulebook_id in existing_ids:
        return {"done": True, "rulebook_id": rulebook_id, "game_name": game_name,
                "pages_processed": 0, "elements_found": 0, "cached": True}

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/rulebooks/{rulebook_id}",
            params={"source_type": source_type, "game_name": game_name},
            files={"file": (pdf_path.name, f, "application/pdf")},
            stream=True,
            timeout=1200,  # large PDFs can take several minutes
        )
    resp.raise_for_status()

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

    assert done_event is not None, f"SSE stream for {rulebook_id} never sent a done event"
    return done_event


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires backend at localhost:8001")
    config.addinivalue_line("markers", "ingest: uploads a PDF to the backend")
    config.addinivalue_line("markers", "frontend: requires Node.js + frontend dir")


@pytest.fixture(scope="session")
def backend(tmp_path_factory):
    """
    Start an isolated test backend on port 8001 with a temporary data directory.

    If port 8001 is already in use, assumes an existing test server and skips launching.
    Sets BGR_DATA_DIR to a temp directory so storage.py and main.py use isolated data.
    """
    if _port_in_use(TEST_PORT):
        # Assume an existing test server is running
        yield BASE_URL
        return

    # Create an isolated data directory for the test server
    data_dir = tmp_path_factory.mktemp("bgr_test_data")

    env = os.environ.copy()
    env["BGR_DATA_DIR"] = str(data_dir)
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    proc = subprocess.Popen(
        [
            str(UVICORN_BIN),
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(TEST_PORT),
            "--log-level", "warning",
        ],
        cwd=str(BACKEND_DIR),
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
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Rulebook ingestion fixtures (session-scoped, ingest once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ingested_rulebook(backend):
    """Ingest the small test PDF (3c-NT) once per session."""
    done = _ingest_rulebook(backend, TEST_RULEBOOK_ID)
    return TEST_RULEBOOK_ID, done


@pytest.fixture(scope="session")
def ingested_acbov(backend):
    """Ingest ACBoV (AC.pdf, 32 pages) once per session."""
    done = _ingest_rulebook(backend, "ACBoV")
    return "ACBoV", done


@pytest.fixture(scope="session")
def ingested_fromage(backend):
    """Ingest Fromage (f_rulebook.pdf) once per session."""
    done = _ingest_rulebook(backend, "f")
    return "f", done


@pytest.fixture(scope="session")
def ingested_log(backend):
    """Ingest Lands of Galzyr (log.pdf) once per session."""
    done = _ingest_rulebook(backend, "LOG")
    return "LOG", done


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
