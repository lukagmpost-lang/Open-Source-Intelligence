import networkx as nx

import pytest

from osi.analysis import (
    betweenness_centrality,
    compare_centralities,
    compare_communities,
    cross_reference,
    degree_centrality,
    degree_distribution,
    leiden_communities,
    louvain_communities,
    network_health,
    pagerank,
    rich_club,
)


def _barbell() -> nx.Graph:
    graph = nx.barbell_graph(4, 0)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def test_betweenness_and_closeness_match_networkx():
    graph = _barbell()
    graph.add_edge(100, 101, weight=2.0)
    bridges = betweenness_centrality(graph)
    expected_bridges = nx.betweenness_centrality(graph, weight="weight")
    closeness = nx.closeness_centrality(graph)
    from osi.analysis import closeness_centrality

    scored = closeness_centrality(graph)
    for node in graph:
        assert abs(bridges[node] - expected_bridges[node]) < 1e-8
        assert abs(scored[node] - closeness[node]) < 1e-8


def test_robustness_degree_removes_the_hub_first():
    from osi.analysis import giant_halved_at, robustness

    graph = nx.star_graph(10)
    nx.set_edge_attributes(graph, 1.0, "weight")
    intact = nx.global_efficiency(graph)
    results = robustness(graph, strategies=["degree", "betweenness", "random"], remove_ratio=[0.1], runs=5)
    assert abs(results["baseline"]["efficiency"] - intact) < 1e-9
    assert results["baseline"]["largest"] == 1.0
    # 10% of 11 nodes removes one node: the center. The leaves are then isolated.
    for name in ("degree", "betweenness"):
        stats = results[name][0.1]
        assert stats["remaining"] == 10
        assert abs(stats["largest"] - 1 / 11) < 1e-9
        assert stats["components"] == 10
        assert stats["efficiency"] == 0.0
    assert results["random"][0.1]["remaining"] == 10
    again = robustness(graph, strategies=["random"], remove_ratio=[0.1], runs=5)
    assert again["random"] == results["random"]
    halved = giant_halved_at(results)
    assert halved["degree"] == 0.1
    assert halved["betweenness"] == 0.1


def test_robustness_ranks_the_intact_graph_once(monkeypatch, capsys):
    from osi.analysis import betweenness_centrality as real_betweenness
    from osi.analysis import robustness

    calls = {"betweenness": 0}

    def counted(graph):
        calls["betweenness"] += 1
        return real_betweenness(graph)

    monkeypatch.setattr("osi.analysis.betweenness_centrality", counted)
    graph = nx.star_graph(10)
    nx.set_edge_attributes(graph, 1.0, "weight")
    results = robustness(graph)
    assert calls["betweenness"] == 1
    assert list(results["degree"]) == [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    # 0% is the intact graph for every strategy. The center is still connected to every leaf.
    assert results["degree"][0.0]["largest"] == 1.0
    assert results["betweenness"][0.0] == results["degree"][0.0]
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("robustness ")]
    assert "robustness 0/21" in lines
    assert "robustness 21/21" in lines
    # Passing the ranking skips the second computation.
    robustness(graph, betweenness_scores=real_betweenness(graph), degree_scores={node: graph.degree(node) for node in graph})
    assert calls["betweenness"] == 1


def test_centralities_are_sorted_dicts_without_printing(capsys):
    graph = _barbell()
    degree = degree_centrality(graph)
    ranks = pagerank(graph)
    bridges = betweenness_centrality(graph)
    assert list(degree) == sorted(degree, key=lambda node: (-degree[node], str(node)))
    assert list(ranks) == sorted(ranks, key=lambda node: (-ranks[node], str(node)))
    assert max(bridges, key=bridges.get) in (3, 4)
    assert capsys.readouterr().out == ""


def test_louvain_and_leiden_split_the_barbell():
    graph = _barbell()
    louvain = louvain_communities(graph)
    leiden = leiden_communities(graph)
    assert set(louvain.values()) == {0, 1}
    assert set(leiden.values()) == {0, 1}
    assert louvain[0] == louvain[1]
    assert leiden[0] == leiden[1]
    assert louvain[0] != louvain[5]


def test_compare_centralities_ranks_every_node():
    graph = _barbell()
    compared = compare_centralities(graph)
    assert set(compared["columns"]) == {"degree", "betweenness", "closeness", "pagerank", "eigenvector"}
    assert compared["rank"][compared["order"]["pagerank"][0]]["pagerank"] == 1
    assert len(compared["order"]["degree"]) == graph.number_of_nodes()


def test_compare_communities_runs_girvan_newman_on_a_small_graph():
    graph = _barbell()
    compared = compare_communities(graph)
    assert compared["louvain"]["count"] == 2
    assert compared["louvain"]["modularity"] > 0.3
    assert compared["girvan_newman"]["skipped"] is False
    assert compared["girvan_newman"]["count"] >= 2
    assert 0 in compared["girvan_newman"]["assignment"].values()


def test_girvan_newman_skips_graphs_over_1000_nodes():
    graph = nx.path_graph(1001)
    nx.set_edge_attributes(graph, 1.0, "weight")
    compared = compare_communities(graph)
    assert compared["girvan_newman"]["skipped"] is True
    assert "1001" in compared["girvan_newman"]["message"]
    assert compared["louvain"]["count"] >= 1


def test_cross_reference_prints_hub_communities(capsys):
    graph = _barbell()
    centralities = compare_centralities(graph)
    communities = compare_communities(graph)
    hubs = cross_reference(graph, communities, centralities, top_n=2)
    output = capsys.readouterr().out
    assert hubs
    assert "COMMUNITY ASSIGNMENT OF TOP HUBS:" in output
    assert "Top hubs" in output


def test_leiden_falls_back_to_louvain(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name in {"leidenalg", "igraph"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    graph = _barbell()
    assert leiden_communities(graph) == louvain_communities(graph)


def test_health_assortativity_uses_the_largest_component():
    graph = nx.star_graph(5)
    # A separate triangle would change the coefficient if the whole graph were used.
    graph.add_edges_from([(10, 11), (11, 12), (12, 10)])
    health = network_health(graph)
    largest = graph.subgraph(max(nx.connected_components(graph), key=len))
    assert health["assortativity"] == pytest.approx(nx.degree_assortativity_coefficient(largest))
    assert health["assortativity"] != pytest.approx(nx.degree_assortativity_coefficient(graph))
    assert health["num_components"] == 2
    assert health["max_degree"] == 5
    assert health["avg_degree"] == pytest.approx(sum(degree for _node, degree in graph.degree()) / graph.number_of_nodes())
    assert health["avg_clustering"] == pytest.approx(nx.average_clustering(graph))
    assert health["transitivity"] == pytest.approx(nx.transitivity(graph))


def test_degree_distribution_prefers_a_power_law_histogram():
    degrees = []
    for k in range(1, 12):
        degrees.extend([k] * max(1, round(400 / k**2)))
    if sum(degrees) % 2:
        degrees.append(1)
    graph = nx.configuration_model(degrees, seed=0)
    fit = degree_distribution(graph)
    assert set(fit) == {"best_fit", "power_law", "lognormal", "exponential"}
    assert fit["power_law"]["params"]["alpha"] > 0
    assert fit["power_law"]["r_squared"] > 0.95
    assert fit["power_law"]["r_squared"] > fit["exponential"]["r_squared"]
    assert fit["best_fit"] in {"power_law", "lognormal", "exponential"}


def test_rich_club_matches_networkx_below_the_default_thresholds():
    graph = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    expected = nx.rich_club_coefficient(graph, normalized=False)
    assert rich_club(graph, [0])[0] == pytest.approx(expected[0])
    assert rich_club(graph) == {10: None, 20: None, 50: None, 100: None}
    graph.add_edge(0, 0)
    assert rich_club(graph) == {10: None, 20: None, 50: None, 100: None}
