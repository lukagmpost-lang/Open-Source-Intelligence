import json

import pytest

from osi.answer import call_llm, write_answer
from osi.result import ResultObject


class _Body:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _result() -> ResultObject:
    return ResultObject(
        intent="rank_nodes",
        params={"run": "simple-v1", "metric": "pagerank", "top": 3},
        values={"alice": 0.415481, "carol": 0.274590, "bob": 0.213570},
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=["The graph has only 4 nodes, so one added or removed edge can change the result."],
        runtime_ms=1,
    )


def _capture(monkeypatch, content: str) -> dict:
    captured: dict = {}

    def fake_urlopen(request, timeout=60):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Body({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("osi.answer.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    return captured


def monkeypatch_missing():
    from pathlib import Path

    return Path("/tmp/osi-answer-no-dotenv")


def _clear_llm_env(monkeypatch) -> None:
    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_ollama_request_has_no_authorization_header(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "phi4-mini")
    captured = _capture(monkeypatch, "Alice leads with 0.415481.")
    text = call_llm("say hi")
    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert text == "Alice leads with 0.415481."
    assert request.full_url == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "phi4-mini"
    assert body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "say hi"}]
    assert request.has_header("Authorization") is False


def test_groq_request_sends_authorization_header(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    captured = _capture(monkeypatch, "Alice leads with 0.415481.")
    text = call_llm("say hi")
    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert text == "Alice leads with 0.415481."
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["messages"] == [{"role": "user", "content": "say hi"}]
    assert body["temperature"] == 0
    assert request.get_header("Authorization") == "Bearer test-key"


def test_groq_without_a_key_raises(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        call_llm("say hi")


def test_write_answer_keeps_the_template_when_the_model_invents_a_number(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _capture(monkeypatch, "Alice's score is 9.9.")
    text = write_answer(_result(), use_llm=True)
    assert "9.9" not in text
    assert "0.415481" in text


def test_write_answer_uses_the_model_when_the_numbers_match(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _capture(monkeypatch, "Alice leads PageRank at 0.415481.")
    text = write_answer(_result(), use_llm=True)
    assert text.startswith("Alice leads PageRank at 0.415481.")
    assert "only 4 nodes" in text
