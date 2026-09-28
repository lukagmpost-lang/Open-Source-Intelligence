import pytest

from osi.analysis import degree_centrality
from osi.executors import (
    connectivity,
    explain_node,
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
    }
    assert set(critical.values) == {"random", "degree", "betweenness"}
    assert communities.trust in {"stable", "moderate", "unstable"}
    assert sum(communities.values.values()) == _graph.number_of_nodes()


def test_rank_nodes_puts_the_hub_first(sample_run):
    run, graph = sample_run
    result = rank_nodes(run, metric="degree", top=3)
    expected = next(iter(degree_centrality(graph)))
    assert expected == "hub"
    assert next(iter(result.values)) == "hub"
    assert len(result.values) == 3


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


def test_explain_node_has_six_fields(sample_run):
    run, _graph = sample_run
    result = explain_node(run, "hub")
    assert set(result.values) == {"degree", "pagerank", "betweenness", "closeness", "community", "neighbors"}
    assert len(result.values["neighbors"]) == 5
    assert all("node" in item and "weight" in item for item in result.values["neighbors"])
