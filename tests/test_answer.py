import json
import re

import networkx as nx
import pytest

from conftest import has_run
from osi.answer import (
    FINDINGS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    call_llm,
    verify_numbers,
    verify_style,
    write_answer,
)
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


def test_groq_request_sends_authorization_header(monkeypatch):
    monkeypatch.setattr("osi.answer._SEED_SUPPORTED", True, raising=False)
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
    assert body["seed"] == 42
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("User-agent") == "osi/0.1"


def test_a_provider_that_rejects_seed_retries_without_it(monkeypatch):
    import io
    import urllib.error

    from osi.answer import NONDETERMINISTIC_CAVEAT, note_determinism

    monkeypatch.setattr("osi.answer._SEED_SUPPORTED", True, raising=False)
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    bodies: list[dict] = []

    def fake_urlopen(request, timeout=60):
        body = json.loads(request.data.decode("utf-8"))
        bodies.append(body)
        if "seed" in body:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"error":{"message":"seed is not supported"}}'),
            )
        return _Body({"choices": [{"message": {"content": "The same sentence."}}]})

    monkeypatch.setattr("osi.answer.urllib.request.urlopen", fake_urlopen)
    try:
        assert call_llm("say hi", temperature=0, seed=42) == "The same sentence."
        assert bodies[0]["temperature"] == 0
        assert bodies[0]["seed"] == 42
        assert "seed" not in bodies[1]
        assert bodies[1]["temperature"] == 0
        noted = _result()
        note_determinism(noted)
        assert NONDETERMINISTIC_CAVEAT in noted.caveats
    finally:
        monkeypatch.setattr("osi.answer._SEED_SUPPORTED", True, raising=False)


def test_groq_without_a_key_raises(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        call_llm("say hi")


def test_call_llm_raises_when_the_api_key_is_empty(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    with pytest.raises(RuntimeError, match="https://console.groq.com/keys"):
        call_llm("say hi")


def test_default_base_url_is_groq(monkeypatch):
    from osi.answer import llm_settings

    _clear_llm_env(monkeypatch)
    monkeypatch.setattr("osi.answer._dotenv_path", lambda: monkeypatch_missing())
    assert llm_settings()["base_url"] == "https://api.groq.com/openai/v1"


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
    prose = "There is 1 community. The largest community id is 0 and its size is 4."
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
    assert "findings" not in text.lower()


def test_write_answer_uses_the_model_when_the_numbers_match(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _capture(monkeypatch, "Alice is more central than the other accounts. Rank 0.415481.")
    text = write_answer(_result(), use_llm=True)
    assert text.startswith("Alice is more central than the other accounts.")
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
    calls = _cached_llm(monkeypatch, ["Alice is more central than the other accounts. Rank 0.415481."])
    result = _result()
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    second = write_answer(result, use_llm=True, question="top 3 by pagerank")
    assert second == first
    assert second.startswith("Alice is more central than the other accounts.")
    assert len(calls) == 1


def test_different_question_misses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    calls = _cached_llm(
        monkeypatch,
        [
            "Alice is more central than the other accounts. Rank 0.415481.",
            "Carol is more central than a typical account. Rank 0.274590.",
        ],
    )
    result = _result()
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    second = write_answer(result, use_llm=True, question="who leads")
    assert first.startswith("Alice is more central than the other accounts.")
    assert second.startswith("Carol is more central than a typical account.")
    assert len(calls) == 2


def test_updating_a_run_invalidates_its_cache(tmp_path, monkeypatch):
    import networkx as nx

    from osi.store import create_run, save_graph

    database = tmp_path / "store.db"
    monkeypatch.setenv("OSI_STORE", str(database))
    create_run("file", {"layer": "file"}, run_id="cache-v1", path=database)
    calls = _cached_llm(
        monkeypatch,
        [
            "Alice is more central than the other accounts. Rank 0.415481.",
            "Alice is more central than the other accounts. Rank 0.415481.",
        ],
    )
    result = _result()
    result.params = {**result.params, "run": "cache-v1"}
    first = write_answer(result, use_llm=True, question="top 3 by pagerank")
    graph = nx.Graph()
    graph.add_edge("alice", "bob", weight=1.0)
    save_graph("cache-v1", "file", graph, path=database)
    second = write_answer(result, use_llm=True, question="top 3 by pagerank")
    assert first.startswith("Alice is more central than the other accounts.")
    assert second.startswith("Alice is more central than the other accounts.")
    assert len(calls) == 2


def test_system_prompt_includes_plain_language_rules():
    assert "Lead with what happened, not what the numbers are." in SYSTEM_PROMPT
    assert "Use one analogy per answer." in SYSTEM_PROMPT
    assert "if the top accounts leave, most people lose their connection to each other." in SYSTEM_PROMPT
    assert 'No "modularity", "clustering", "assortativity", "components", "density".' in SYSTEM_PROMPT
    assert "Every number must appear in a tool result." in SYSTEM_PROMPT
    assert "Never quote a raw centrality score" in SYSTEM_PROMPT
    assert "36x more central than typical" in SYSTEM_PROMPT
    assert "a small town with distinct neighborhoods" in SYSTEM_PROMPT
    assert "an office where different teams work on different floors" in SYSTEM_PROMPT
    assert "a neighborhood where people know each other by face" in SYSTEM_PROMPT
    assert "a network of safe houses connected by trusted couriers" in SYSTEM_PROMPT
    assert "a group of people connected by shared interests" in SYSTEM_PROMPT
    assert "Reddit in 2012 was a completely different animal" in SYSTEM_PROMPT
    assert "Write in this style." in SYSTEM_PROMPT
    assert "This is a general network." in SYSTEM_PROMPT


def test_write_answer_includes_run_context_in_the_prompt(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    captured = _capture(monkeypatch, "Alice leads PageRank at 0.415481.")
    result = _result()
    result.n_nodes = 4
    result.n_edges = 4
    write_answer(result, use_llm=True, question="top 3 by pagerank", use_cache=False)
    body = json.loads(captured["request"].data.decode("utf-8"))
    user = body["messages"][1]["content"]
    assert (
        'Context: this result is from a network called "simple-v1" '
        'with 4 nodes and 4 edges. The question was: "top 3 by pagerank".'
    ) in user
    assert body["messages"][0]["content"] == SYSTEM_PROMPT


def test_network_health_sends_only_findings(monkeypatch):
    seen: dict = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["system"] = kwargs.get("system")
        return "This is a dense network — most accounts are connected to dozens of others."

    monkeypatch.setattr("osi.answer.call_llm", fake)
    result = ResultObject(
        intent="network_health",
        params={"run": "simple-v1"},
        values={
            "avg_degree": 34.2,
            "modularity": 0.32,
            "findings": [
                "This is a dense network — most accounts are connected to dozens of others.",
                "The groups are blurry — they overlap heavily.",
            ],
        },
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=1,
    )
    text = write_answer(result, use_llm=True, question="how healthy is this network", use_cache=False)
    assert seen["system"] == FINDINGS_SYSTEM_PROMPT
    assert "34.2" not in seen["prompt"]
    assert "modularity" not in seen["prompt"]
    assert "dense network" in seen["prompt"]
    assert "dense network" in text
    assert "How sure: stable." in text


def test_metric_recital_falls_back_to_the_findings(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt, **kwargs: "The modularity is 0.32 and the clustering is high.",
    )
    result = ResultObject(
        intent="network_health",
        params={"run": "simple-v1"},
        values={
            "modularity": 0.32,
            "findings": ["The groups are blurry — they overlap heavily."],
        },
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=1,
    )
    text = write_answer(result, use_llm=True, question="how healthy", use_cache=False)
    assert "modularity" not in text.lower()
    assert text.startswith("The groups are blurry")
    assert "How sure: stable." in text


def test_agent_fallback_includes_findings_from_every_tool():
    from osi.answer import templated_fallback

    critical = ResultObject(
        intent="critical_nodes",
        params={},
        values={"findings": ["akdas is connected to 759 other accounts."]},
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=0,
    )
    structural = ResultObject(
        intent="structural_criticality",
        params={},
        values={"findings": ["Removing 30% of the top accounts by degree halves the network."]},
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=0,
    )
    text = templated_fallback(
        structural,
        {"critical_nodes": critical, "structural_criticality": structural},
    )
    assert "759" in text
    assert "halves the network" in text
    assert text.endswith("How sure: stable.")


def test_agent_fallback_dedupes_findings():
    from osi.answer import templated_fallback

    shared = "akdas is connected to 759 other accounts."
    first = ResultObject(
        intent="critical_nodes",
        params={},
        values={"findings": [shared, "These five are the hubs of the network."]},
        method="exact",
        sample_size=None,
        trust="moderate",
        caveats=[],
        runtime_ms=0,
    )
    second = ResultObject(
        intent="structural_criticality",
        params={},
        values={"findings": [shared, "Removing 30% halves the network."]},
        method="exact",
        sample_size=None,
        trust="moderate",
        caveats=[],
        runtime_ms=0,
    )
    text = templated_fallback(second, {"critical_nodes": first, "structural_criticality": second})
    assert text.count(shared) == 1
    assert "These five are the hubs" in text
    assert "halves the network" in text


def test_fallback_with_an_empty_scratchpad_uses_the_result_findings():
    from osi.answer import templated_fallback

    result = ResultObject(
        intent="network_health",
        params={"run": "simple-v1"},
        values={"findings": ["This is a sparse network — most accounts have only a few connections."]},
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=0,
    )
    text = templated_fallback(result, {})
    assert text == templated_fallback(result)
    assert text.startswith("This is a sparse network")
    assert "How sure: stable." in text


def _pagerank_result() -> ResultObject:
    return ResultObject(
        intent="rank_nodes",
        params={"metric": "pagerank", "run": "r2008-v2"},
        values={
            "akdas": 0.02494,
            "findings": ["akdas is 36x more central than the typical account."],
        },
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=0,
    )


def test_raw_centrality_score_is_rejected():
    from osi.answer import raw_centrality_problem

    problem = raw_centrality_problem("akdas has a centrality score of 0.02494.", _pagerank_result())
    assert problem is not None
    assert "0.02494" in problem
    assert "akdas is 36x more central than the typical account." in problem
    assert "Do not quote raw metric values." in problem


def test_comparative_centrality_ratio_passes():
    from osi.answer import raw_centrality_problem

    assert raw_centrality_problem("akdas is 36x more central than typical.", _pagerank_result()) is None


def test_answer_without_centrality_is_rejected():
    from osi.answer import raw_centrality_problem

    problem = raw_centrality_problem("akdas has 759 connections.", _pagerank_result())
    assert problem is not None
    assert "missing centrality" in problem.lower()


def test_verify_style_rejects_the_graph_is():
    assert verify_style("The graph is a triangle of three people.") is False


def test_verify_style_rejects_three_or_more_jargon_words():
    text = "Modularity, betweenness, and assortativity all look high."
    assert verify_style(text) is False


_CONSEQUENCE = re.compile(r"\b(because|means)\b", re.IGNORECASE)
_METRIC_WORDS = ("modularity", "assortativity", "components", "density")


def _last_sentence(text: str) -> str:
    body = re.sub(r"\s*How sure:.*\Z", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    return parts[-1]


@pytest.mark.skipif(not has_run("r2008-v2"), reason="Reddit 2008 run not in the store")
def test_rank_nodes_answer_for_reddit_2008_uses_an_analogy():
    from osi.executors import rank_nodes

    result = rank_nodes("r2008-v2", metric="pagerank", top=5)
    text = write_answer(result, use_llm=False, question="who matters here")
    assert re.search(r"\b(like|as|imagine)\b", text, re.IGNORECASE)


@pytest.mark.skipif(not has_run("r2012-v2"), reason="Reddit 2012 run not in the store")
def test_network_health_answer_for_reddit_2012_skips_metric_names():
    from osi.executors import network_health

    result = network_health("r2012-v2")
    text = write_answer(result, use_llm=False, question="how healthy is this network").lower()
    for word in _METRIC_WORDS:
        assert word not in text


@pytest.mark.skipif(not has_run("r2008-v2"), reason="Reddit 2008 run not in the store")
def test_answer_names_an_account_when_the_tools_name_one():
    from osi.executors import rank_nodes

    result = rank_nodes("r2008-v2", metric="pagerank", top=5)
    text = write_answer(result, use_llm=False, question="who matters here")
    names = [str(key) for key in result.values if key != "findings"]
    assert names
    assert any(name in text for name in names)


@pytest.mark.skipif(not has_run("r2008-v2"), reason="Reddit 2008 run not in the store")
def test_answer_ends_with_a_consequence_not_a_metric():
    from osi.executors import rank_nodes

    result = rank_nodes("r2008-v2", metric="pagerank", top=5)
    text = write_answer(result, use_llm=False, question="who matters here")
    last = _last_sentence(text).lower()
    assert _CONSEQUENCE.search(last) or "if the" in last or "most people" in last
    for word in _METRIC_WORDS:
        assert word not in last


def test_every_finding_states_a_consequence():
    from osi.executors import _criticality_findings, interpret_communities, interpret_health, interpret_rank
    from osi.findings import generate_findings, load_baselines

    texts: list[str] = []
    for graph in (nx.complete_graph(30), nx.path_graph(6), nx.empty_graph(120), nx.star_graph(10)):
        texts.extend(item["text"] for item in generate_findings({}, graph))
    texts.extend(item["text"] for item in generate_findings(load_baselines()["reddit_2012"]))
    texts.extend(
        interpret_health(
            {
                "avg_degree": 40,
                "max_degree": 800,
                "modularity": 0.2,
                "components": 40,
                "assortativity": -0.8,
            },
            nx.Graph(),
        )
    )
    star = nx.relabel_nodes(nx.star_graph(10), {0: "akdas", **{leaf: f"leaf{leaf}" for leaf in range(1, 11)}})
    ranked = [("akdas", 0.5), ("leaf1", 0.1), ("leaf2", 0.1), ("leaf3", 0.1), ("leaf4", 0.1)]
    texts.extend(interpret_rank(ranked, star, "pagerank"))
    texts.extend(interpret_communities(4, 12, 0.2))
    texts.extend(_criticality_findings({0.0: 1.0, 0.05: 0.4, 0.30: 0.05}, {}, {0.0: 1.0, 0.30: 0.9}, 40))
    assert texts
    for text in texts:
        assert _CONSEQUENCE.search(text), text
