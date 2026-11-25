import pytest


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Every test gets its own empty runs/ directory."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("MINESWEEPML_RUNS", str(runs))
    return runs
