import networkx as nx

from osi.answer import write_answer
from osi.executors import network_health
from osi.findings import generate_findings, load_baselines, pick_top_findings


def _texts(metrics: dict, graph=None) -> list[str]:
    return [item["text"] for item in generate_findings(metrics, graph)]


def test_dense_graph_fires_the_dense_network_rule():
    texts = _texts({}, nx.complete_graph(30))
    assert any("dense network" in text for text in texts)


def test_sparse_graph_fires_the_sparse_network_rule():
    texts = _texts({}, nx.path_graph(6))
    assert any("sparse network" in text for text in texts)


def test_fragmented_graph_fires_the_fragmentation_rule():
    texts = _texts({}, nx.empty_graph(120))
    assert any("highly fragmented" in text for text in texts)


def test_star_graph_fires_the_broadcast_pattern_rule():
    texts = _texts({}, nx.star_graph(10))
    assert any("broadcast pattern" in text for text in texts)


def test_reddit_2012_metrics_produce_at_least_three_findings():
    findings = generate_findings(load_baselines()["reddit_2012"])
    assert len(findings) >= 3
    assert findings == sorted(findings, key=lambda item: item["score"], reverse=True)


def test_pagerank_far_above_the_median_names_the_top_account():
    scores = {"akdas": 0.5, **{f"n{i}": 0.01 for i in range(20)}}
    texts = _texts({"pagerank": scores})
    assert "akdas is 50x more central than the typical account." in texts


def test_pagerank_within_ten_times_the_median_adds_no_finding():
    texts = _texts({"pagerank": {"akdas": 0.1, **{f"n{i}": 0.01 for i in range(20)}}})
    assert not any("more central" in text for text in texts)


def test_pick_top_findings_keeps_the_higher_score_for_one_metric():
    findings = [
        {"text": "lower", "score": 4, "metric": "components"},
        {"text": "higher", "score": 9, "metric": "components"},
        {"text": "other", "score": 5, "metric": "clustering"},
    ]
    assert pick_top_findings(findings, n=4) == ["higher", "other"]


def test_how_healthy_on_reddit_2012_returns_plain_findings():
    result = network_health("r2012-v2")
    findings = result.values["findings"]
    assert len(findings) >= 3
    blob = " ".join(findings).lower()
    assert "modularity" not in blob
    assert "density" not in blob
    text = write_answer(result, use_llm=False, question="how healthy")
    assert "modularity" not in text.lower()
    assert "density" not in text.lower()
