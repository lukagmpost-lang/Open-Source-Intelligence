import networkx as nx

from osi.analysis import (
    betweenness_centrality,
    compare_centralities,
    compare_communities,
    cross_reference,
    degree_centrality,
    leiden_communities,
    louvain_communities,
    pagerank,
)


def _barbell() -> nx.Graph:
    graph = nx.barbell_graph(4, 0)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


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
