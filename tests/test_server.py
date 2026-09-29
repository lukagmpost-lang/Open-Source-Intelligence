import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from osi.server import app

client = TestClient(app)


def test_get_runs_returns_a_list():
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = response.json()
    assert isinstance(runs, list)
    assert all(isinstance(run_id, str) for run_id in runs)
    assert "simple-v1" in runs


def test_post_ask_returns_an_answer():
    response = client.post(
        "/api/ask",
        json={"run": "simple-v1", "question": "top 3 by pagerank", "use_llm": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert "answer" in body
    assert "alice" in body["answer"]
    assert "How sure:" in body["answer"]
    assert body["intent"] == "rank_nodes"
    assert body["method"] == "exact"
    assert body["trust"] == "stable"
    assert isinstance(body["runtime_ms"], int)


def test_post_ask_missing_run_is_404():
    response = client.post(
        "/api/ask",
        json={"run": "no-such-run-server", "question": "top 3 by pagerank", "use_llm": False},
    )
    assert response.status_code == 404


def test_post_ask_unsupported_question_returns_the_helpful_message():
    response = client.post(
        "/api/ask",
        json={"run": "simple-v1", "question": "   ", "use_llm": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "unsupported"
    assert "I don't know how to answer that" in body["answer"]
    assert "top 10 by pagerank" in body["answer"]


def test_post_ask_open_question_describes_the_graph():
    response = client.post(
        "/api/ask",
        json={"run": "simple-v1", "question": "why is the graph shaped this way", "use_llm": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "discuss"
    assert "alice" in body["answer"]
    assert "assortativity" not in body["answer"]
    assert "clustering" not in body["answer"]


def test_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<select" in response.text
    assert "Ask" in response.text
