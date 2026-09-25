import networkx as nx

from osi.analysis import (
    betweenness_centrality,
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
