"""Structural fingerprints as rank percentiles inside the node's own graph."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

# First call for a graph ranks every node. Later calls reuse that table.
_RANKED: dict[tuple[int, int], dict[Any, np.ndarray]] = {}


def _gini(degrees: list[float]) -> float:
    """Gini of a neighbor-degree list. Equal degrees score 0. No neighbors score 0."""
    if not degrees:
        return 0.0
    ordered = np.sort(np.array(degrees, dtype=float))
    total = float(ordered.sum())
    if total == 0.0:
        return 0.0
    count = len(ordered)
    index = np.arange(1, count + 1, dtype=float)
    # Sorted-sample Gini. The last term removes the bias from 1-based ranks.
    return float((2.0 * np.sum(index * ordered) / (count * total)) - (count + 1) / count)


def _rank_percentile(scores: dict[Any, float]) -> dict[Any, float]:
    """Mid-rank divided by n - 1. A lone minimum is 0 and a lone maximum is 1.

    Ties share one value, the average of the positions they occupy, so two nodes
    with the same score stay the same distance apart in every graph.
    """
    count = len(scores)
    # One observation has nobody below it. The feature stays 0 instead of dividing by zero.
    if count <= 1:
        return {key: 0.0 for key in scores}
    # Lowest score first. The name only orders a tie; the mid-rank does not use it.
    ordered = sorted(scores.items(), key=lambda item: (item[1], str(item[0])))
    percentile: dict[Any, float] = {}
    start = 0
    while start < count:
        end = start + 1
        while end < count and ordered[end][1] == ordered[start][1]:
            end += 1
        mid = (start + end - 1) / 2.0
        share = mid / (count - 1)
        for index in range(start, end):
            percentile[ordered[index][0]] = share
        start = end
    return percentile


def _rank_features(G: nx.Graph, metrics: dict) -> dict[Any, np.ndarray]:
    """Seven features, each already on [0, 1] inside this graph."""
    pagerank = metrics.get("pagerank") or {}
    betweenness = metrics.get("betweenness") or {}
    closeness = metrics.get("closeness") or {}
    community = metrics.get("community") or {}
    members: dict[Any, list] = {}
    for node in G.nodes:
        members.setdefault(community.get(node, node), []).append(node)
    sizes = {community_id: len(group) for community_id, group in members.items()}
    # Community size is ranked across communities, not across nodes.
    community_percentile = _rank_percentile({community_id: float(size) for community_id, size in sizes.items()})
    degree = dict(G.degree())
    degree_percentile = _rank_percentile({node: float(degree[node]) for node in G.nodes})
    pagerank_percentile = _rank_percentile({node: float(pagerank.get(node, 0.0)) for node in G.nodes})
    between_percentile = _rank_percentile({node: float(betweenness.get(node, 0.0)) for node in G.nodes})
    close_percentile = _rank_percentile({node: float(closeness.get(node, 0.0)) for node in G.nodes})
    mean_degree: dict[Any, float] = {}
    gini: dict[Any, float] = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        neighbor_degrees = [float(degree[other]) for other in neighbors]
        # No neighbors: mean degree 0, ranked with everyone else, not dropped from the denominator.
        mean_degree[node] = float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0
        gini[node] = _gini(neighbor_degrees)
    neighbor_percentile = _rank_percentile(mean_degree)
    rows: dict[Any, np.ndarray] = {}
    for node in G.nodes:
        community_id = community.get(node, node)
        rows[node] = np.array(
            [
                degree_percentile[node],
                pagerank_percentile[node],
                between_percentile[node],
                close_percentile[node],
                community_percentile[community_id],
                neighbor_percentile[node],
                gini[node],
            ],
            dtype=float,
        )
    return rows


def compute_fingerprint(G: nx.Graph, node, metrics: dict) -> np.ndarray:
    """Rank-percentile feature vector. No z-score: every feature is already in [0, 1]."""
    if node not in G:
        raise KeyError(node)
    key = (id(G), id(metrics))
    ranked = _RANKED.get(key)
    if ranked is None:
        ranked = _rank_features(G, metrics)
        _RANKED[key] = ranked
    vector = ranked.get(node)
    if vector is None:
        raise KeyError(node)
    # Callers compare copies. The cache must keep the ranked numbers.
    return vector.copy()


def fingerprint_similarity(fp1, fp2) -> float:
    """Cosine similarity. A zero vector is orthogonal to everything, including itself."""
    left = np.asarray(fp1, dtype=float)
    right = np.asarray(fp2, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)
