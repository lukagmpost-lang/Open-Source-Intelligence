"""Cached betweenness, critical nodes, and removal curves."""

import time

import networkx as nx

from osi.agent import critical_nodes
from osi.analysis import _betweenness_uncached, betweenness_centrality
from osi.executors import robustness, structural_criticality
from osi.store import create_run, delete_run, get_metric, save_graph


def _graph() -> nx.Graph:
    graph = nx.star_graph(12)
    nx.set_edge_attributes(graph, 1.0, "weight")
    graph = nx.relabel_nodes(graph, {node: f"n{node}" for node in graph})
    return graph


def _save(tmp_path, monkeypatch, run_id="cache-v1"):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    graph = _graph()
    create_run("file", {"layer": "g"}, run_id=run_id)
    save_graph(run_id, "g", graph)
    return run_id, graph


def test_first_call_computes_and_caches(tmp_path, monkeypatch):
    run, graph = _save(tmp_path, monkeypatch)
    calls = {"between": 0, "critical": 0, "curve": 0}
    real_between = _betweenness_uncached
    real_rank = __import__("osi.agent", fromlist=["rank_nodes"]).rank_nodes
    real_curve = robustness

    def count_between(*args, **kwargs):
        calls["between"] += 1
        return real_between(*args, **kwargs)

    def count_rank(*args, **kwargs):
        calls["critical"] += 1
        return real_rank(*args, **kwargs)

    def count_curve(*args, **kwargs):
        calls["curve"] += 1
        return real_curve(*args, **kwargs)

    monkeypatch.setattr("osi.analysis._betweenness_uncached", count_between)
    monkeypatch.setattr("osi.agent.rank_nodes", count_rank)
    monkeypatch.setattr("osi.executors.robustness", count_curve)

    scores = betweenness_centrality(graph, run_id=run)
    ranked = critical_nodes(run, top_n=3)
    curve = structural_criticality(run)

    assert calls == {"between": 1, "critical": 1, "curve": 1}
    assert scores
    assert ranked.intent == "critical_nodes"
    assert curve.intent == "structural_criticality"
    assert get_metric(run, "betweenness_exact")
    assert get_metric(run, "critical_nodes_v2_3")
    assert get_metric(run, "structural_criticality")
    sampled = betweenness_centrality(graph, run_id=run, k=4)
    assert sampled
    assert get_metric(run, "betweenness_sampled_4")
    assert calls["between"] == 2


def test_second_call_returns_in_under_100ms(tmp_path, monkeypatch):
    run, graph = _save(tmp_path, monkeypatch)
    real_between = _betweenness_uncached
    real_rank = __import__("osi.agent", fromlist=["rank_nodes"]).rank_nodes
    real_curve = robustness

    def slow_between(*args, **kwargs):
        time.sleep(0.2)
        return real_between(*args, **kwargs)

    def slow_rank(*args, **kwargs):
        time.sleep(0.2)
        return real_rank(*args, **kwargs)

    def slow_curve(*args, **kwargs):
        time.sleep(0.2)
        return real_curve(*args, **kwargs)

    monkeypatch.setattr("osi.analysis._betweenness_uncached", slow_between)
    monkeypatch.setattr("osi.agent.rank_nodes", slow_rank)
    monkeypatch.setattr("osi.executors.robustness", slow_curve)

    betweenness_centrality(graph, run_id=run)
    critical_nodes(run, top_n=3)
    structural_criticality(run)

    started = time.perf_counter()
    betweenness_centrality(graph, run_id=run)
    critical_nodes(run, top_n=3)
    structural_criticality(run)
    assert time.perf_counter() - started < 0.1


def test_resaving_the_run_clears_the_cache(tmp_path, monkeypatch):
    run, graph = _save(tmp_path, monkeypatch)
    betweenness_centrality(graph, run_id=run)
    critical_nodes(run, top_n=3)
    structural_criticality(run)
    assert get_metric(run, "betweenness_exact")
    assert get_metric(run, "critical_nodes_v2_3")
    assert get_metric(run, "structural_criticality")

    save_graph(run, "g", graph)
    assert get_metric(run, "betweenness_exact") is None
    assert get_metric(run, "critical_nodes_v2_3") is None
    assert get_metric(run, "structural_criticality") is None

    betweenness_centrality(graph, run_id=run)
    create_run("file", {"layer": "g"}, run_id=run)
    assert get_metric(run, "betweenness_exact") is None

    betweenness_centrality(graph, run_id=run)
    delete_run(run)
    assert get_metric(run, "betweenness_exact") is None


def test_cached_result_matches_fresh_computation(tmp_path, monkeypatch):
    run, graph = _save(tmp_path, monkeypatch)
    scores = betweenness_centrality(graph, run_id=run)
    ranked = critical_nodes(run, top_n=3)
    curve = structural_criticality(run)

    again_scores = betweenness_centrality(graph, run_id=run)
    again_ranked = critical_nodes(run, top_n=3)
    again_curve = structural_criticality(run)
    assert again_scores == scores
    assert again_ranked.values == ranked.values
    assert again_curve.values == curve.values

    save_graph(run, "g", graph)
    fresh_scores = betweenness_centrality(graph, run_id=run)
    fresh_ranked = critical_nodes(run, top_n=3)
    fresh_curve = structural_criticality(run)
    assert fresh_scores == scores
    assert fresh_ranked.values == ranked.values
    assert fresh_curve.values == curve.values
