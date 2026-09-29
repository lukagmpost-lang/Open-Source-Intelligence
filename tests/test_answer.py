import json

import pytest

from osi.answer import SYSTEM_PROMPT, call_llm, verify_numbers, verify_style, write_answer
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
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "say hi"},
    ]
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
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "say hi"},
    ]
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


_SCORES = {"alice": 0.415481, "carol": 0.274590, "bob": 0.213570}


def test_verify_numbers_rejects_a_score_that_is_not_stored():
    assert verify_numbers("0.999999", _SCORES) is False


def test_verify_numbers_accepts_three_decimal_places():
    assert verify_numbers("0.415", _SCORES) is True


def test_verify_numbers_accepts_four_decimal_places():
    assert verify_numbers("0.4155", _SCORES) is True


def test_verify_numbers_accepts_the_stored_score_in_a_sentence():
    assert verify_numbers("alice at 0.415481", _SCORES) is True


def _communities() -> ResultObject:
    return ResultObject(
        intent="list_communities",
        params={"run": "simple-v1", "algorithm": "louvain"},
        values=ResultObject.community_values({0: 4}, 0.25),
        method="exact",
        sample_size=None,
        trust="unstable",
        caveats=["The graph has only 4 nodes, so one added or removed edge can change the result."],
        runtime_ms=1,
    )


def test_write_answer_names_community_fields():
    text = write_answer(_communities(), use_llm=False)
    assert text.startswith("This network has 1 communities.")
    assert "The largest contains 4 nodes." in text
    assert "How sure: unstable" in text


def test_write_answer_rejects_zero_communities_when_there_is_one(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _capture(monkeypatch, "There are 0 communities and the largest size is 4.")
    text = write_answer(_communities(), use_llm=True)
    assert "This network has 1 communities." in text
    assert "0 communities" not in text


def test_write_answer_keeps_a_community_sentence_that_names_the_zero_id(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    prose = "There is 1 community. largest_community is 0 and largest_size is 4. modularity is 0.250000."
    _capture(monkeypatch, prose)
    text = write_answer(_communities(), use_llm=True)
    assert text.startswith(prose)
    assert "only 4 nodes" in text


def _discuss_result() -> ResultObject:
    return ResultObject(
        intent="discuss",
        params={"run": "simple-v1", "question": "why is the graph shaped this way"},
        values={
            "nodes": 4,
            "edges": 4,
            "components": 1,
            "triangles": 1,
            "structure": [
                {
                    "node": "alice",
                    "degree": 3,
                    "neighbors": [
                        {"node": "bob", "weight": 2.0},
                        {"node": "carol", "weight": 3.0},
                        {"node": "dave", "weight": 1.0},
                    ],
                }
            ],
        },
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=["The graph has only 4 nodes, so one added or removed edge can change the result."],
        runtime_ms=1,
    )


def test_discuss_keeps_a_plain_explanation(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt, **kwargs: "Alice, Bob, and Carol form a triangle, and Dave is only tied to Alice.",
    )
    text = write_answer(_discuss_result(), use_llm=True, question="why is the graph shaped this way")
    assert "triangle" in text
    assert "Dave" in text


def test_discuss_drops_an_invented_count(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt, **kwargs: "The graph contains 99 separate cliques.",
    )
    text = write_answer(_discuss_result(), use_llm=True, question="why is the graph shaped this way")
    assert "99" not in text
    assert "4 nodes" in text


def test_write_answer_uses_the_model_when_the_numbers_match(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _capture(monkeypatch, "Alice leads PageRank at 0.415481.")
    text = write_answer(_result(), use_llm=True)
    assert text.startswith("Alice leads PageRank at 0.415481.")
    assert "only 4 nodes" in text


def _cached_llm(monkeypatch, replies: list[str]) -> list[str]:
    """Replace the model with a queue of sentences and record each call."""
    sent: list[str] = []

    def fake(prompt: str, **kwargs) -> str:
        sent.append(replies[len(sent)])
        return sent[-1]

    monkeypatch.setattr("osi.answer.call_llm", fake)
    return sent


def test_second_call_returns_the_cached_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    calls = _cached_llm(monkeypatch, ["Alice leads PageRank at 0.415481."])
    result = _result()
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    second = write_answer(result, use_llm=True, question="top 3 by pagerank")
    assert second == first
    assert second.startswith("Alice leads PageRank at 0.415481.")
    assert len(calls) == 1


def test_different_question_misses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    calls = _cached_llm(
        monkeypatch,
        ["Alice leads PageRank at 0.415481.", "Carol follows at 0.274590."],
    )
    result = _result()
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    second = write_answer(result, use_llm=True, question="who leads")
    assert first.startswith("Alice leads PageRank at 0.415481.")
    assert second.startswith("Carol follows at 0.274590.")
    assert len(calls) == 2


def test_updating_a_run_invalidates_its_cache(tmp_path, monkeypatch):
    import networkx as nx

    from osi.store import create_run, save_graph

    database = tmp_path / "store.db"
    monkeypatch.setenv("OSI_STORE", str(database))
    create_run("file", {"layer": "file"}, run_id="cache-v1", path=database)
    calls = _cached_llm(monkeypatch, ["Alice leads PageRank at 0.415481.", "Alice leads PageRank at 0.415481."])
    result = _result()
    result.params = {**result.params, "run": "cache-v1"}
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    graph = nx.Graph()
    graph.add_edge("alice", "bob", weight=1.0)
    save_graph("cache-v1", "file", graph, path=database)
    second = write_answer(result, use_llm=True, question="top 3 by pagerank")
    assert first.startswith("Alice leads PageRank at 0.415481.")
    assert second.startswith("Alice leads PageRank at 0.415481.")
    assert len(calls) == 2


def test_system_prompt_includes_plain_language_rules():
    assert "does not know network science" in SYSTEM_PROMPT
    assert "The first sentence must directly answer the question." in SYSTEM_PROMPT
    assert "Do not start with" in SYSTEM_PROMPT
    assert "How sure: " in SYSTEM_PROMPT
    assert "modularity, betweenness, assortativity" in SYSTEM_PROMPT


def test_write_answer_includes_run_context_in_the_prompt(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    captured = _capture(monkeypatch, "Alice leads PageRank at 0.415481.")
    result = _result()
    result.n_nodes = 4
    result.n_edges = 4
    write_answer(result, use_llm=True, question="top 3 by pagerank")
    body = json.loads(captured["request"].data.decode("utf-8"))
    user = body["messages"][1]["content"]
    assert (
        'Context: this result is from a network called "simple-v1" '
        'with 4 nodes and 4 edges. The question was: "top 3 by pagerank".'
    ) in user
    assert body["messages"][0]["content"] == SYSTEM_PROMPT


def test_verify_style_rejects_the_graph_is():
    assert verify_style("The graph is a triangle of three people.") is False


def test_verify_style_rejects_three_or_more_jargon_words():
    text = "Modularity, betweenness, and assortativity all look high."
    assert verify_style(text) is False
