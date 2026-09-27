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


def test_fingerprint_has_seven_features_at_unit_variance():
    graph = nx.path_graph(6)
    metrics = _metrics(graph)
    rows = np.vstack([compute_fingerprint(graph, node, metrics) for node in graph.nodes])
    assert rows.shape == (6, 7)
    for column in rows.T:
        std = float(column.std())
        assert std == pytest.approx(1.0) or std == pytest.approx(0.0)


def test_constant_column_is_zero_when_numpy_std_is_only_noise():
    # One community makes log(community size) constant. numpy.std of that column is ~1e-15.
    graph = nx.star_graph(30)
    metrics = _metrics(graph)
    metrics["community"] = {node: 0 for node in graph.nodes}
    rows = np.vstack([compute_fingerprint(graph, node, metrics) for node in graph.nodes])
    assert rows[:, 4] == pytest.approx(0.0)
    assert np.isfinite(rows).all()
    assert float(np.max(np.abs(rows))) < 1e3


def test_cosine_of_the_same_vector_is_one_and_a_zero_vector_is_zero():
    assert fingerprint_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert fingerprint_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert fingerprint_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
