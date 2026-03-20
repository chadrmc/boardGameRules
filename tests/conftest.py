"""Shared fixtures for BGR test suite."""
import json
import warnings
warnings.filterwarnings("ignore", category=Warning, message=".*doesn't match a supported version.*")
import pytest
import requests

BASE_URL = "http://localhost:8000"
# Small PDF used for ingest tests — 1-page, fast to process
TEST_PDF = "examples/3c-NT.pdf"
TEST_RULEBOOK_ID = "test-3cnt"


def backend_is_up():
    try:
        requests.get(f"{BASE_URL}/rulebooks", timeout=2)
        return True
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires backend at localhost:8000")
    config.addinivalue_line("markers", "ingest: uploads a PDF to the backend")
    config.addinivalue_line("markers", "frontend: requires Node.js + frontend dir")


@pytest.fixture(scope="session")
def backend():
    """Skip entire session if backend is not running."""
    if not backend_is_up():
        pytest.skip("Backend not running at localhost:8000 — start it first")
    return BASE_URL


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


@pytest.fixture(scope="session")
def project_root():
    from pathlib import Path
    # tests/ lives one level inside the project root
    return Path(__file__).parent.parent
