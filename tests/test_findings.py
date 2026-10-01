import networkx as nx
import pytest

from conftest import has_run
from osi.answer import write_answer
from osi.executors import network_health
from osi.executors import _selected_findings, network_health, rank_nodes
from osi.findings import HEALTH_SOURCES, RANK_SOURCES, generate_findings, load_baselines, pick_top_findings


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
    assert any("akdas is 50x more central than the typical account." in text for text in texts)


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


def test_each_finding_names_the_tool_category_that_owns_it():
    found = generate_findings(dict(load_baselines()["reddit_2012"]))
    assert found
    allowed = HEALTH_SOURCES | RANK_SOURCES | {"fragility", "criticality"}
    for item in found:
        assert {"text", "source", "score"} <= set(item)
        assert item["source"] in allowed


def test_network_health_findings_do_not_include_hub_findings():
    metrics = dict(load_baselines()["reddit_2012"])
    health = _selected_findings(metrics, sources=HEALTH_SOURCES, n=8)
    hubs = _selected_findings(metrics, sources=RANK_SOURCES, n=8)
    assert health
    assert hubs
    assert not (set(health) & set(hubs))
    blob = " ".join(health).lower()
    assert "1,890" not in blob
    assert "top hub" not in blob
    result = network_health("simple-v1")
    for line in result.values["findings"]:
        assert "1,890" not in line
        assert "top hub is twice" not in line.lower()
        assert not line.lower().startswith("one account")


def test_rank_nodes_findings_include_hub_findings():
    metrics = dict(load_baselines()["reddit_2012"])
    hubs = _selected_findings(metrics, sources=RANK_SOURCES, n=8)
    assert any(item["source"] in RANK_SOURCES for item in generate_findings(metrics))
    assert any("1,890" in line or "hub" in line.lower() for line in hubs)
    result = rank_nodes("simple-v1", metric="degree", top=5)
    blob = " ".join(result.values["findings"]).lower()
    assert "hub" in blob or "central" in blob


@pytest.mark.skipif(not has_run("r2012-v2"), reason="Reddit 2012 run not in the store")
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
