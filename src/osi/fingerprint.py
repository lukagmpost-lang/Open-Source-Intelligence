"""Structural fingerprints for one node, scaled so each feature has unit variance."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

# First call for a graph fits every node. Later calls reuse that scale.
_FITTED: dict[tuple[int, int], dict[Any, np.ndarray]] = {}

_FEATURE_COUNT = 7
# numpy.std of a constant column is a rounding residue (about 1e-15), not spread.
_STD_FLOOR = 1e-8


def _zscore(values: dict[Any, float]) -> dict[Any, float]:
    """Map each value to a z-score. A constant group stays at 0."""
    if not values:
        return {}
    series = np.array(list(values.values()), dtype=float)
    std = float(series.std())
    if std == 0.0:
        return {node: 0.0 for node in values}
    mean = float(series.mean())
    return {node: (float(value) - mean) / std for node, value in values.items()}


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


def _raw_features(G: nx.Graph, metrics: dict) -> dict[Any, np.ndarray]:
    pagerank = metrics.get("pagerank") or {}
    betweenness = metrics.get("betweenness") or {}
    closeness = metrics.get("closeness") or {}
    community = metrics.get("community") or {}
    # PageRank is compared inside the community. The other two scores are graph-wide.
    pagerank_z: dict[Any, float] = {}
    members: dict[Any, list] = {}
    for node in G.nodes:
        members.setdefault(community.get(node, node), []).append(node)
    for group in members.values():
        local = {node: float(pagerank.get(node, 0.0)) for node in group}
        pagerank_z.update(_zscore(local))
    between_z = _zscore({node: float(betweenness.get(node, 0.0)) for node in G.nodes})
    close_z = _zscore({node: float(closeness.get(node, 0.0)) for node in G.nodes})
    sizes = {community_id: len(group) for community_id, group in members.items()}
    degree = dict(G.degree())
    rows: dict[Any, np.ndarray] = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        neighbor_degrees = [float(degree[other]) for other in neighbors]
        mean_neighbor = float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0
        community_id = community.get(node, node)
        rows[node] = np.array(
            [
                math.log(degree[node] + 1),
                pagerank_z.get(node, 0.0),
                between_z.get(node, 0.0),
                close_z.get(node, 0.0),
                math.log(sizes.get(community_id, 1)),
                mean_neighbor,
                _gini(neighbor_degrees),
            ],
            dtype=float,
        )
    return rows


def _fit(G: nx.Graph, metrics: dict) -> dict[Any, np.ndarray]:
    raw = _raw_features(G, metrics)
    if not raw:
        return {}
    matrix = np.vstack([raw[node] for node in G.nodes])
    # Unit variance, not a z-score: the mean stays. A constant column cannot set the scale.
    spread = matrix.std(axis=0)
    constant = spread <= _STD_FLOOR
    scale = spread.copy()
    scale[constant] = 1.0
    scaled = matrix / scale
    # Zero the dead column. Dividing by the residue would make it the largest feature.
    scaled[:, constant] = 0.0
    return {node: scaled[index] for index, node in enumerate(G.nodes)}


def compute_fingerprint(G: nx.Graph, node, metrics: dict) -> np.ndarray:
    """Feature vector for one node, with each feature at unit variance across the graph."""
    if node not in G:
        raise KeyError(node)
    key = (id(G), id(metrics))
    fitted = _FITTED.get(key)
    if fitted is None:
        fitted = _fit(G, metrics)
        _FITTED[key] = fitted
    vector = fitted.get(node)
    if vector is None:
        raise KeyError(node)
    return vector


def fingerprint_similarity(fp1, fp2) -> float:
    """Cosine similarity. A zero vector is orthogonal to everything, including itself."""
    left = np.asarray(fp1, dtype=float)
    right = np.asarray(fp2, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)
