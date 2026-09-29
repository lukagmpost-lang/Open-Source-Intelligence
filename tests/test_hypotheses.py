from itertools import combinations

import networkx as nx

from osi.hypotheses import find_unasked_observations, generate_hypotheses, infer_domain


def _cliques() -> tuple[nx.Graph, dict]:
    left = [f"a{index}" for index in range(6)]
    right = [f"b{index}" for index in range(6)]
    graph = nx.Graph()
    graph.add_edges_from(combinations(left, 2))
    graph.add_edges_from(combinations(right, 2))
    graph.add_edge("a0", "b0")
    return graph, {node: 0 for node in graph}


def test_two_weakly_linked_groups_suggest_a_schism():
    graph, assignment = _cliques()
    found = generate_hypotheses({"assignment": assignment}, graph, "forum")
    match = next(item for item in found if item["hypothesis"] == "There may be a schism forming.")
    assert match["check"]() is True
    assert 0 < match["confidence"] <= 1


def test_a_bridge_node_is_called_out():
    graph = nx.barbell_graph(6, 1)
    found = generate_hypotheses({}, graph, "social")
    match = next(item for item in found if "only bridge" in item["hypothesis"])
    assert match["check"]() is True


def test_fast_fragmentation_is_a_bus_factor_problem():
    found = generate_hypotheses({}, nx.star_graph(20), "social")
    match = next(item for item in found if item["hypothesis"] == "This network has a bus-factor problem.")
    assert match["check"]() is True


def test_an_isolated_company_team_is_flagged_only_for_that_domain():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graph.add_edges_from([(3, 4), (4, 5), (5, 3), (3, 6)])
    assignment = {0: "team", 1: "team", 2: "team", 3: "rest", 4: "rest", 5: "rest", 6: "rest"}
    found = generate_hypotheses({"assignment": assignment}, graph, "company")
    assert any(item["hypothesis"] == "This team may be isolated from the rest of the org." for item in found)
    other = generate_hypotheses({"assignment": assignment}, graph, "forum")
    assert all("isolated" not in item["hypothesis"] for item in other)


def test_a_falling_activity_series_names_the_periods():
    found = generate_hypotheses({"activity": [10, 8, 5, 1]}, None, "forum")
    assert found[0]["hypothesis"] == "Activity has declined in the last 3 periods."
    assert found[0]["check"]() is True


def test_at_most_four_hypotheses_are_returned():
    graph, assignment = _cliques()
    graph.add_node("solo")
    metrics = {
        "assignment": {**assignment, "solo": 1},
        "activity": [9, 4, 1],
        "robustness": {"degree": {0.05: 0.2}},
        "betweenness": {"a0": 0.8},
    }
    # a0 sits in a clique, so clustering stays high and the bridge rule stays quiet.
    found = generate_hypotheses(metrics, graph, "company")
    assert 1 <= len(found) <= 4
    assert found == sorted(found, key=lambda item: item["confidence"], reverse=True)


def test_unasked_observations_returns_two_or_three():
    notes = find_unasked_observations(
        nx.path_graph(12),
        {"nodes": 12, "avg_degree": 1.8, "max_degree": 2, "components": 1},
        "forum",
        "what should I be worried about",
    )
    assert 2 <= len(notes) <= 3
    assert all("worried" not in note.casefold() for note in notes)


def test_reddit_is_a_forum_domain():
    assert infer_domain("reddit", "reddit_user") == "forum"
    assert infer_domain("github", "github_co_contribution") == "company"
