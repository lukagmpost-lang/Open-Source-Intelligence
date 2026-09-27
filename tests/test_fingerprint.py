import math

import networkx as nx
import numpy as np
import pytest

from osi.fingerprint import compute_fingerprint, fingerprint_similarity, normalize_fingerprints


def _metrics(graph: nx.Graph) -> dict:
    nodes = list(graph.nodes)
    return {
        "pagerank": {node: float(index + 1) for index, node in enumerate(nodes)},
        "betweenness": {node: float(index) for index, node in enumerate(nodes)},
        "closeness": {node: 0.1 * (index + 1) for index, node in enumerate(nodes)},
        "community": {node: 0 if index < len(nodes) / 2 else 1 for index, node in enumerate(nodes)},
    }


def test_fingerprint_is_raw_and_pooled_columns_have_unit_variance():
    graph = nx.path_graph(6)
    metrics = _metrics(graph)
    rows = np.vstack([compute_fingerprint(graph, node, metrics) for node in graph.nodes])
    assert rows.shape == (6, 7)
    # An endpoint of a path has degree 1. The raw feature is log(degree + 1), not a z-score.
    assert rows[0, 0] == pytest.approx(math.log(2))
    nodes = list(graph.nodes)
    left = {node: compute_fingerprint(graph, node, metrics) for node in nodes[:3]}
    right = {node: compute_fingerprint(graph, node, metrics) for node in nodes[3:]}
    norm_left, norm_right = normalize_fingerprints(left, right)
    pooled = np.vstack(list(norm_left.values()) + list(norm_right.values()))
    assert pooled.mean(axis=0) == pytest.approx(0.0, abs=1e-8)
    for column in pooled.T:
        std = float(column.std())
        assert std == pytest.approx(1.0) or std == pytest.approx(0.0)


def test_constant_column_is_zero_when_the_pool_has_no_spread():
    # One community makes log(community size) constant. numpy.std of that column is ~1e-15.
    graph = nx.star_graph(30)
    metrics = _metrics(graph)
    metrics["community"] = {node: 0 for node in graph.nodes}
    raw = {node: compute_fingerprint(graph, node, metrics) for node in graph.nodes}
    # The unscaled column stays at log(size). Zeroing happens only when the pool has no spread.
    assert raw[0][4] == pytest.approx(math.log(graph.number_of_nodes()))
    left, _right = normalize_fingerprints(raw, raw)
    stacked = np.vstack(list(left.values()))
    assert stacked[:, 4] == pytest.approx(0.0)
    assert np.isfinite(stacked).all()
    assert float(np.max(np.abs(stacked))) < 1e3


def test_pooled_zscore_keeps_an_absolute_gap_between_hubs():
    # Each value would be "the hub" inside its own one-node set. Pooled, they are opposites.
    small = {"hub": np.array([1.0, 0.0])}
    large = {"hub": np.array([100.0, 0.0])}
    norm_small, norm_large = normalize_fingerprints(small, large)
    assert norm_small["hub"][0] < 0.0
    assert norm_large["hub"][0] > 0.0
    assert fingerprint_similarity(norm_small["hub"], norm_large["hub"]) < 0.0


def test_cosine_of_the_same_vector_is_one_and_a_zero_vector_is_zero():
    assert fingerprint_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert fingerprint_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert fingerprint_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
