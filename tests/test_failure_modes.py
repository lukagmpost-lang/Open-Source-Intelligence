import networkx as nx

from osi.agent import run_agent
from osi.failure_modes import apply_failure_modes, rank_by_question


def test_a_forum_hub_matches_hub_departure():
    graph = nx.star_graph(100)
    matched = apply_failure_modes({}, graph, "forum")
    names = [mode["name"] for mode in matched]
    assert "hub_departure" in names
    hub = next(mode for mode in matched if mode["name"] == "hub_departure")
    assert hub["severity"] == "high"
    assert hub["evidence"]["max_degree"] > 50 * hub["evidence"]["avg_degree"]


def test_a_company_bridge_matches_key_person_risk():
    graph = nx.barbell_graph(6, 1)
    matched = apply_failure_modes({}, graph, "company")
    names = [mode["name"] for mode in matched]
    assert "key_person_risk" in names


def test_fragmentation_is_expected_for_a_forum():
    graph = nx.Graph()
    graph.add_edges_from((index, index + 100) for index in range(20))
    matched = apply_failure_modes({}, graph, "forum")
    mode = next(item for item in matched if item["name"] == "growth_fragmentation")
    assert mode["is_expected"] is True
    assert mode["severity"] == "medium"


def test_a_worry_question_ranks_severe_modes_and_drops_expected_ones():
    matched = [
        {
            "name": "growth_fragmentation",
            "description": "Pieces multiply.",
            "severity": "medium",
            "is_expected": True,
        },
        {
            "name": "moderation_concentration",
            "description": "A few accounts moderate.",
            "severity": "medium",
            "is_expected": False,
        },
        {
            "name": "hub_departure",
            "description": "A top poster leaves.",
            "severity": "high",
            "is_expected": False,
        },
        {
            "name": "bridge_loss",
            "description": "A bridge stops posting.",
            "severity": "high",
            "is_expected": False,
        },
    ]
    ranked = rank_by_question(matched, "what should I be worried about")
    assert [mode["name"] for mode in ranked] == [
        "hub_departure",
        "bridge_loss",
        "moderation_concentration",
    ]
    assert all(not mode.get("is_expected") for mode in ranked)


def test_reddit_2012_worry_answer_names_a_failure_mode_and_an_account(monkeypatch):
    def answer_from_the_modes(prompt, **kwargs):
        assert "these failure modes apply" in prompt
        mode = next(
            name
            for name in (
                "hub_departure",
                "bridge_loss",
                "moderation_concentration",
                "key_person_risk",
            )
            if name in prompt
        )
        account = "CosmicBard" if "CosmicBard" in prompt else "incredible-ninja"
        assert account in prompt
        return (
            f"ANSWER: The failure mode {mode} is what matters. "
            f"{account} has 1890 connections and is more central than a typical account. "
            f"If {account} stopped posting, the forum would lose its anchor."
        )

    monkeypatch.setattr("osi.agent.call_llm", answer_from_the_modes)
    result = run_agent("r2012-v2", "what should I be worried about", use_llm=True)
    assert any(
        name in result.answer
        for name in ("hub_departure", "bridge_loss", "moderation_concentration", "key_person_risk")
    )
    assert "CosmicBard" in result.answer or "incredible-ninja" in result.answer
