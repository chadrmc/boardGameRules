"""
Frontend build test — verifies the Next.js app compiles without TypeScript errors.
Requires Node.js and npm to be available in PATH.
"""
import subprocess
import pytest
from pathlib import Path


@pytest.mark.frontend
def test_frontend_builds_without_errors(project_root):
    frontend_dir = project_root / "frontend"
    if not frontend_dir.exists():
        pytest.skip("frontend/ directory not found")

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"npm run build failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )
