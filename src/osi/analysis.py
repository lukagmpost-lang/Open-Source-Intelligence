"""Network measures on a public social graph. Nothing here prints."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities as _louvain_communities


def _by_score(scores: dict[Any, float]) -> dict[Any, float]:
    return dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))


def degree_centrality(G: nx.Graph) -> dict[Any, float]:
    return _by_score(nx.degree_centrality(G))


def pagerank(G: nx.Graph, weight: str = "weight") -> dict[Any, float]:
    return _by_score(nx.pagerank(G, weight=weight))


def betweenness_centrality(G: nx.Graph, weight: str = "weight") -> dict[Any, float]:
    # Slow. Exact betweenness inspects shortest paths for every pair, so it
    # gets expensive quickly as accounts are added. Keep it on small graphs.
    return _by_score(nx.betweenness_centrality(G, weight=weight))


def _community_index(groups: list[set[Any]]) -> dict[Any, int]:
    ordered = sorted(groups, key=lambda members: (-len(members), str(sorted(members, key=str))))
    scores: dict[Any, int] = {}
    for index, members in enumerate(ordered):
        for node in members:
            scores[node] = index
    return dict(sorted(scores.items(), key=lambda item: (item[1], str(item[0]))))


def louvain_communities(G: nx.Graph, weight: str = "weight") -> dict[Any, int]:
    groups = _louvain_communities(G, weight=weight, seed=0)
    return _community_index([set(group) for group in groups])


def leiden_communities(G: nx.Graph, weight: str = "weight") -> dict[Any, int]:
    # leidenalg needs python-igraph. If that import fails, Louvain is the stand-in.
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return louvain_communities(G, weight=weight)

    igraph = ig.Graph.from_networkx(G)
    edge_weight = weight if weight in igraph.es.attributes() else None
    partition = leidenalg.find_partition(
        igraph,
        leidenalg.ModularityVertexPartition,
        weights=edge_weight,
        seed=0,
    )
    names = igraph.vs["_nx_name"]
    groups: dict[int, set[Any]] = {}
    for index, community in enumerate(partition.membership):
        groups.setdefault(community, set()).add(names[index])
    return _community_index(list(groups.values()))
