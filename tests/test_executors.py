import pytest

from osi.analysis import degree_centrality
from osi.executors import (
    anomaly_scan,
    baseline_compare,
    connectivity,
    explain_node,
    interpret_health,
    interpret_rank,
    list_communities,
    network_health,
    rank_nodes,
    structural_criticality,
)
from osi.sample import sample_graph
from osi.store import create_run, save_graph


@pytest.fixture
def sample_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    graph = sample_graph()
    # An isolate makes the disconnected-path case something the sample can answer.
    graph.add_node("solo")
    create_run("file", {"layer": "sample", "source": "file"}, run_id="sample-v1")
    save_graph("sample-v1", "sample", graph)
    return "sample-v1", graph


def _populated(result, intent: str) -> None:
    assert result.intent == intent
    assert result.method in {"exact", "sampled", "approximate"}
    assert result.trust in {"stable", "moderate", "unstable", "random"}
    assert isinstance(result.params, dict)
    assert isinstance(result.values, dict)
    assert isinstance(result.caveats, list)
    assert isinstance(result.runtime_ms, int)
    assert result.runtime_ms >= 0
    if result.method == "exact":
        assert result.sample_size is None
        assert result.trust == "stable" or intent == "list_communities"


def test_each_executor_fills_a_result(sample_run):
    run, _graph = sample_run
    ranked = rank_nodes(run, metric="pagerank", top=5)
    communities = list_communities(run)
    health = network_health(run)
    critical = structural_criticality(run)
    path = connectivity(run, "alpha-0", "gamma-11")
    explained = explain_node(run, "hub")
    _populated(ranked, "rank_nodes")
    _populated(communities, "list_communities")
    _populated(health, "network_health")
    _populated(critical, "structural_criticality")
    _populated(path, "connectivity")
    _populated(explained, "explain_node")
    assert set(health.values) == {
        "assortativity",
        "clustering",
        "transitivity",
        "components",
        "avg_degree",
        "max_degree",
        "power_law",
        "findings",
    }
    assert isinstance(health.values["findings"], list)
    assert health.values["findings"]
    assert all("modularity" not in item.lower() for item in health.values["findings"])
    assert {"random", "degree", "betweenness", "findings"} <= set(critical.values)
    assert any("%" in item for item in critical.values["findings"])
    assert communities.trust in {"stable", "moderate", "unstable"}
    assert sum(communities.values["sizes"]) == _graph.number_of_nodes()


def test_list_communities_uses_named_fields(sample_run):
    run, graph = sample_run
    result = list_communities(run)
    values = result.values
    assert set(values) == {
        "n_communities",
        "modularity",
        "sizes",
        "largest_community",
        "largest_size",
        "findings",
    }
    assert values["n_communities"] == len(values["sizes"])
    assert values["n_communities"] >= 1
    assert sum(values["sizes"]) == graph.number_of_nodes()
    assert values["sizes"] == sorted(values["sizes"], reverse=True)
    assert values["largest_size"] == values["sizes"][0]
    assert isinstance(values["largest_community"], int)
    assert isinstance(values["modularity"], float)


def test_rank_nodes_puts_the_hub_first(sample_run):
    run, graph = sample_run
    result = rank_nodes(run, metric="degree", top=3)
    expected = next(iter(degree_centrality(graph)))
    assert expected == "hub"
    assert next(iter(result.values)) == "hub"
    scores = {key: value for key, value in result.values.items() if key != "findings"}
    assert len(scores) == 3
    assert result.values["findings"]
    assert "hub" in result.values["findings"][0]


def test_connectivity_path_and_disconnected(sample_run):
    run, _graph = sample_run
    connected = connectivity(run, "alpha-0", "gamma-11")
    assert connected.values["connected"] is True
    assert connected.values["path"][0] == "alpha-0"
    assert connected.values["path"][-1] == "gamma-11"
    assert connected.values["distance"] == len(connected.values["path"]) - 1
    assert connected.values["weight"] > 0
    disconnected = connectivity(run, "hub", "solo")
    assert disconnected.values == {"connected": False}


def test_large_betweenness_is_sampled(sample_run, monkeypatch):
    run, graph = sample_run
    monkeypatch.setattr("osi.executors.BETWEENNESS_SAMPLE_NODES", 1)
    result = rank_nodes(run, metric="betweenness", top=1)
    assert result.method == "sampled"
    assert result.sample_size == min(500, graph.number_of_nodes())
    assert result.trust == "moderate"
    assert result.caveats


def test_large_criticality_uses_ten_trials(sample_run, monkeypatch):
    run, _graph = sample_run
    monkeypatch.setattr("osi.executors.CRITICALITY_SAMPLE_NODES", 1)
    result = structural_criticality(run)
    assert result.method == "sampled"
    assert result.sample_size == 10
    assert result.params["runs"] == 10
    assert result.trust == "moderate"


def test_structural_criticality_findings_name_the_halving_percentage():
    from osi.executors import _criticality_findings

    degree = {0.0: 1.0, 0.01: 0.8, 0.05: 0.4, 0.10: 0.2, 0.30: 0.05}
    betweenness = {0.0: 1.0, 0.20: 0.4}
    random = {0.0: 1.0, 0.30: 0.9}
    findings = _criticality_findings(degree, betweenness, random, n_components=40)
    assert findings[0] == (
        "The network is extremely fragile: removing just 5% of the top accounts by degree halves it."
    )
    assert "Removing random accounts has much less effect — this is a targeted-vulnerability pattern." in findings
    assert "Removing 30% of the top accounts shatters the network into 40 disconnected pieces." in findings

    fragile = _criticality_findings({0.0: 1.0, 0.10: 0.4, 0.30: 0.2}, {}, {0.0: 1.0, 0.20: 0.4}, None)
    assert fragile[0] == "The network is fragile: removing 10% of the top accounts halves it."

    moderate = _criticality_findings({0.0: 1.0, 0.30: 0.4}, {}, {0.0: 1.0, 0.30: 0.4}, None)
    assert moderate == ["The network is moderately fragile: removing 30% halves it."]

    resilient = _criticality_findings({0.0: 1.0, 0.30: 0.8}, {}, {0.0: 1.0, 0.30: 0.7}, None)
    assert resilient == [
        "The network is resilient: it survives removing 30% of the top accounts without halving."
    ]


def test_interpret_health_keeps_the_three_strongest_findings():
    import networkx as nx

    graph = nx.Graph()
    metrics = {
        "avg_degree": 40,
        "max_degree": 800,
        "modularity": 0.2,
        "n_communities": 96,
        "components": 40,
        "assortativity": -0.8,
    }
    findings = interpret_health(metrics, graph)
    assert findings == [
        "This is a dense network — most accounts are connected to dozens of others.",
        "The groups are blurry — they overlap heavily.",
        "Hubs connect to leaves, not to each other. This is a broadcast network, not a community.",
    ]


def test_interpret_health_names_a_clean_split_and_a_sparse_network():
    import networkx as nx

    findings = interpret_health(
        {"avg_degree": 1.5, "max_degree": 2, "modularity": 0.8, "n_communities": 4, "components": 1, "assortativity": 0.1},
        nx.Graph(),
    )
    assert findings == [
        "This is a sparse network — most accounts have only a few connections.",
        "This network splits cleanly into groups.",
    ]


def test_interpret_health_flags_a_dominant_hub_and_fragmentation():
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(1000))
    findings = interpret_health(
        {
            "avg_degree": 2,
            "max_degree": 250,
            "modularity": 0.45,
            "components": 20,
            "assortativity": 0.0,
        },
        graph,
    )
    assert findings[0] == "A few hubs dominate — the top node has 250 connections vs an average of 2."
    assert "It's fragmented — most nodes are in small disconnected pieces." in findings


def test_interpret_rank_names_a_hub_far_above_the_average():
    import networkx as nx

    graph = nx.star_graph(10)
    graph = nx.relabel_nodes(graph, {0: "akdas", **{leaf: f"leaf{leaf}" for leaf in range(1, 11)}})
    ranked = [("akdas", 0.5), ("leaf1", 0.1), ("leaf2", 0.1), ("leaf3", 0.1), ("leaf4", 0.1)]
    findings = interpret_rank(ranked, graph, "pagerank")
    assert findings[0] == "The most central accounts are akdas, leaf1, leaf2, leaf3, and leaf4."
    assert findings[1] == "akdas is connected to 10 other accounts — more than 5x the average."
    assert findings[2] == "These five are the hubs of the network."
    wide = [("akdas", 0.5), *[(f"n{i}", 0.01) for i in range(20)]]
    assert "akdas is 50x more central than the typical account." in interpret_rank(wide, graph, "pagerank")
    assert "more central" not in " ".join(interpret_rank(wide, graph, "betweenness"))


def test_baseline_compare_names_a_ratio(sample_run):
    run, _graph = sample_run
    result = baseline_compare(run, baseline="snap_facebook")
    assert result.intent == "baseline_compare"
    assert result.values["ratios"]
    assert any("x" in line for line in result.values["findings"])
    assert "hub" in " ".join(result.values["findings"])


def test_anomaly_scan_returns_three_departures(sample_run):
    run, _graph = sample_run
    result = anomaly_scan(run)
    assert result.intent == "anomaly_scan"
    assert len(result.values["deviations"]) == 3
    assert result.values["findings"]


def test_explain_node_has_six_fields(sample_run):
    run, _graph = sample_run
    result = explain_node(run, "hub")
    assert set(result.values) == {"degree", "pagerank", "betweenness", "closeness", "community", "neighbors"}
    assert len(result.values["neighbors"]) == 5
    assert all("node" in item and "weight" in item for item in result.values["neighbors"])
