"""Structural fingerprints. Scaling happens across both graphs, not inside one."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

# First call for a graph builds every raw vector. Later calls reuse that table.
_RAW: dict[tuple[int, int], dict[Any, np.ndarray]] = {}

# numpy.std of a constant column is a rounding residue (about 1e-15), not spread.
_STD_FLOOR = 1e-8


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
    """Seven absolute features. Nothing here is scaled to this graph."""
    pagerank = metrics.get("pagerank") or {}
    betweenness = metrics.get("betweenness") or {}
    closeness = metrics.get("closeness") or {}
    community = metrics.get("community") or {}
    members: dict[Any, list] = {}
    for node in G.nodes:
        members.setdefault(community.get(node, node), []).append(node)
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
                float(pagerank.get(node, 0.0)),
                float(betweenness.get(node, 0.0)),
                float(closeness.get(node, 0.0)),
                math.log(sizes.get(community_id, 1)),
                mean_neighbor,
                _gini(neighbor_degrees),
            ],
            dtype=float,
        )
    return rows


def compute_fingerprint(G: nx.Graph, node, metrics: dict) -> np.ndarray:
    """Raw feature vector for one node. No z-score and no per-graph scale."""
    if node not in G:
        raise KeyError(node)
    key = (id(G), id(metrics))
    raw = _RAW.get(key)
    if raw is None:
        raw = _raw_features(G, metrics)
        _RAW[key] = raw
    vector = raw.get(node)
    if vector is None:
        raise KeyError(node)
    # Callers compare and scale copies. The cache must keep the unscaled numbers.
    return vector.copy()


def normalize_fingerprints(fps_a, fps_b):
    """Z-score both sets with the mean and std of the two sets pooled.

    A hub's z-score is then "high among these nodes", not "high inside its own graph".
    Returns the two sets in the same order, with the same keys.
    """
    keys_a = list(fps_a)
    keys_b = list(fps_b)
    rows_a = [np.asarray(fps_a[key], dtype=float) for key in keys_a]
    rows_b = [np.asarray(fps_b[key], dtype=float) for key in keys_b]
    if not rows_a and not rows_b:
        return {}, {}
    pooled = np.vstack(rows_a + rows_b)
    mean = pooled.mean(axis=0)
    spread = pooled.std(axis=0)
    # No cross-graph spread: the column carries nothing, so it must not dominate cosine.
    dead = spread <= _STD_FLOOR
    scale = spread.copy()
    scale[dead] = 1.0

    def _apply(rows: list[np.ndarray], keys: list) -> dict[Any, np.ndarray]:
        if not rows:
            return {}
        scaled = (np.vstack(rows) - mean) / scale
        scaled[:, dead] = 0.0
        return {key: scaled[index] for index, key in enumerate(keys)}

    return _apply(rows_a, keys_a), _apply(rows_b, keys_b)


def fingerprint_similarity(fp1, fp2) -> float:
    """Cosine similarity. A zero vector is orthogonal to everything, including itself."""
    left = np.asarray(fp1, dtype=float)
    right = np.asarray(fp2, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)
