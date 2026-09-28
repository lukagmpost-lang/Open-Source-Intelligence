import networkx as nx
import numpy as np
import pytest

from osi.fingerprint import compute_fingerprint, fingerprint_similarity


def _metrics(graph: nx.Graph) -> dict:
    nodes = list(graph.nodes)
    return {
        "pagerank": {node: float(index + 1) for index, node in enumerate(nodes)},
        "betweenness": {node: float(index) for index, node in enumerate(nodes)},
        "closeness": {node: 0.1 * (index + 1) for index, node in enumerate(nodes)},
        "community": {node: 0 if index < len(nodes) / 2 else 1 for index, node in enumerate(nodes)},
    }


def test_fingerprint_is_seven_rank_percentiles():
    graph = nx.path_graph(6)
    metrics = _metrics(graph)
    rows = np.vstack([compute_fingerprint(graph, node, metrics) for node in graph.nodes])
    assert rows.shape == (6, 7)
    assert np.all(rows >= 0.0)
    assert np.all(rows <= 1.0)
    # The last node has the unique highest PageRank, betweenness, and closeness.
    last = list(graph.nodes)[-1]
    leader = compute_fingerprint(graph, last, metrics)
    assert leader[1] == pytest.approx(1.0)
    assert leader[2] == pytest.approx(1.0)
    assert leader[3] == pytest.approx(1.0)
    # Both ends of the path have degree 1, so they share one degree percentile below the middle.
    ends = [compute_fingerprint(graph, node, metrics)[0] for node in (0, 5)]
    assert ends[0] == pytest.approx(ends[1])
    assert ends[0] < compute_fingerprint(graph, 1, metrics)[0]


def test_one_community_has_no_size_rank():
    graph = nx.star_graph(30)
    metrics = _metrics(graph)
    metrics["community"] = {node: 0 for node in graph.nodes}
    rows = np.vstack([compute_fingerprint(graph, node, metrics) for node in graph.nodes])
    # A single community cannot be ranked against another, so the feature is 0.
    assert rows[:, 4] == pytest.approx(0.0)
    assert np.isfinite(rows).all()


def test_degree_leader_scores_one_in_a_small_graph_and_a_large_one():
    for leaf_count in (4, 40):
        graph = nx.star_graph(leaf_count)
        metrics = _metrics(graph)
        center = compute_fingerprint(graph, 0, metrics)
        leaf = compute_fingerprint(graph, 1, metrics)
        assert center[0] == pytest.approx(1.0)
        assert leaf[0] < 1.0
        # Every leaf has the same degree, so every leaf shares that percentile.
        assert leaf[0] == pytest.approx(compute_fingerprint(graph, 2, metrics)[0])


def test_cosine_of_the_same_vector_is_one_and_a_zero_vector_is_zero():
    assert fingerprint_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert fingerprint_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert fingerprint_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
