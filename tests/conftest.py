"""Pytest fixtures for OS-INT tests."""
import os
import tempfile
from pathlib import Path

import pytest


def _has_run(run_id: str) -> bool:
    """Check if a run exists in the store."""
    from osi import store
    try:
        runs = store.list_runs()
        return any(run_id == run[0] for run in runs)
    except Exception:
        return False


@pytest.fixture
def temp_store(monkeypatch):
    """Point the store at a temporary file for the duration of the test.
    
    Loads the example graph as 'simple-v1' so tests that depend on 
    pre-saved runs can use it without requiring global state.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "store.db")
        monkeypatch.setenv("OSI_STORE", db_path)
        
        # Force a fresh store by clearing any cached connections
        from osi import store
        # Re-import to pick up the new env var
        import importlib
        importlib.reload(store)
        
        # Load the example graph
        from osi.loader import load_edge_list
        G = load_edge_list("examples/simple.csv")
        store.create_run("file", {"source": "file", "path": "examples/simple.csv"}, "simple-v1", run_id="simple-v1")
        store.save_graph("simple-v1", "file", G)
        
        yield "simple-v1"
