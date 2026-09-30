import json

from osi.intent import classify_intent, required_tools


def _reply(monkeypatch, payload: dict):
    def fake(prompt: str, **kwargs) -> str:
        fake.prompt = prompt
        return json.dumps(payload)

    monkeypatch.setattr("osi.intent.call_llm", fake)
    return fake


def test_a_worry_question_is_diagnose_and_target(monkeypatch):
    _reply(
        monkeypatch,
        {"intents": ["diagnose", "target"], "scope": "whole", "scope_target": None},
    )
    classified = classify_intent("what should I be worried about")
    assert classified["intents"] == ["diagnose", "target"]
    assert classified["scope"] == "whole"
    assert required_tools(classified["intents"], classified["scope"], "general") == [
        "critical_nodes",
        "structural_criticality",
        "rank_nodes",
        "explain_node",
    ]


def test_an_importance_question_includes_rank_nodes(monkeypatch):
    _reply(monkeypatch, {"intents": ["rank", "target"], "scope": "whole", "scope_target": None})
    classified = classify_intent("who are the most important people")
    assert classified["intents"] == ["rank", "target"]
    tools = required_tools(classified["intents"], classified["scope"], "general")
    assert "rank_nodes" in tools


def test_a_connection_question_is_a_trace_on_an_edge(monkeypatch):
    _reply(
        monkeypatch,
        {"intents": ["trace"], "scope": "edge", "scope_target": ["alice", "bob"]},
    )
    classified = classify_intent("how is alice connected to bob")
    assert classified["intents"] == ["trace"]
    assert classified["scope"] == "edge"
    assert classified["scope_target"] == ["alice", "bob"]
    assert required_tools(classified["intents"], classified["scope"], "general") == ["connectivity"]


def test_a_fragmentation_question_describes_the_whole_graph(monkeypatch):
    _reply(monkeypatch, {"intents": ["describe"], "scope": "whole", "scope_target": None})
    classified = classify_intent("how fragmented is this network")
    assert classified["intents"] == ["describe"]
    assert classified["scope"] == "whole"
    assert required_tools(classified["intents"], classified["scope"], "general") == [
        "network_health",
        "list_communities",
    ]


def test_an_account_question_explains_and_locates_that_node(monkeypatch):
    _reply(
        monkeypatch,
        {"intents": ["explain", "locate"], "scope": "node", "scope_target": "alice"},
    )
    classified = classify_intent("tell me about alice")
    assert classified["intents"] == ["explain", "locate"]
    assert classified["scope"] == "node"
    assert classified["scope_target"] == "alice"
    assert required_tools(classified["intents"], classified["scope"], "general") == ["explain_node"]


def test_a_novel_failure_question_is_classified_by_the_model(monkeypatch):
    seen: dict[str, str] = {}

    def fake(prompt: str, **kwargs) -> str:
        seen["prompt"] = prompt
        return '{"intents": ["diagnose"], "scope": "whole", "scope_target": null}'

    monkeypatch.setattr("osi.intent.call_llm", fake)
    classified = classify_intent("where could this fail")
    assert "where could this fail" in seen["prompt"]
    assert classified["intents"] == ["diagnose"]
    assert classified["scope"] == "whole"
    assert required_tools(classified["intents"], classified["scope"], "general") == [
        "critical_nodes",
        "structural_criticality",
    ]
