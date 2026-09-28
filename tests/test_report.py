import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import report
from osi.store import create_run, save_communities, save_graph, save_metrics


def _write_run(path, run_id, source, graph, metrics, communities):
    create_run(source, {"layer": "layer"}, "", run_id=run_id, path=path)
    save_graph(run_id, "layer", graph, path=path)
    for metric, scores in metrics.items():
        save_metrics(run_id, scores, metric=metric, path=path)
    for algorithm, membership in communities.items():
        save_communities(run_id, algorithm, membership, path=path)


def _two_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    early = nx.Graph()
    early.add_edge("a", "b", weight=3)
    early_metrics = {
        "pagerank": {"a": 0.5, "b": 0.3, "c": 0.2},
        "degree": {"a": 0.4, "b": 0.4, "c": 0.2},
        "betweenness": {"a": 0.0, "b": 0.1, "c": 0.0},
        "closeness": {"a": 0.9, "b": 0.1, "c": 0.05},
    }
    early_communities = {
        "louvain": {"a": 0, "b": 0, "c": 1},
        "leiden": {"a": 4, "b": 4, "c": 2},
    }
    late = nx.Graph()
    late.add_edge("a", "b", weight=9)
    late.add_edge("a", "d", weight=2)
    late_metrics = {
        "pagerank": {"a": 0.2, "d": 0.8},
        "degree": {"a": 0.1, "d": 0.9},
        "betweenness": {"a": 0.3, "d": 0.0},
        "closeness": {"a": 0.4, "d": 0.6},
    }
    late_communities = {
        "louvain": {"a": 7, "b": 7, "d": 1},
        "leiden": {"a": 8, "b": 8, "d": 2},
    }
    solo = nx.Graph()
    solo.add_node("zzzz")
    _write_run(path, "solo", "reddit", solo, {"pagerank": {"zzzz": 1.0}, "degree": {"zzzz": 1.0}, "betweenness": {"zzzz": 0.0}, "closeness": {"zzzz": 0.0}}, {"louvain": {"zzzz": 0}, "leiden": {"zzzz": 0}})
    _write_run(path, "early", "reddit", early, early_metrics, early_communities)
    _write_run(path, "late", "reddit_2012", late, late_metrics, late_communities)
    return path


def test_user_report_ranks_changes_and_neighbors(tmp_path, monkeypatch, capsys):
    _two_runs(tmp_path, monkeypatch)
    loaded = []
    real_load = report.load_graph

    def tracking(run_id, layer, path=None):
        loaded.append(run_id)
        return real_load(run_id, layer, path=path)

    monkeypatch.setattr(report, "load_graph", tracking)
    assert report.main(["--user", "a"]) == 0
    text = capsys.readouterr().out
    assert "solo" not in text
    assert "early (reddit," in text
    assert "late (reddit_2012," in text
    assert "PageRank:     0.500000  (rank 1 / 3)" in text
    assert "Degree:       0.400000  (rank 1)" in text
    assert "Betweenness:  0.000000  (rank 2)" in text
    assert "Louvain:      0" in text
    assert "Leiden:       4" in text
    assert "early  pagerank 1/3  degree 1  betweenness 2  closeness 1  top 100: yes  top 1%: no  top 10%: no" in text
    assert "late  pagerank 2/2  degree 2  betweenness 1  closeness 2  top 100: yes  top 1%: no  top 10%: no" in text
    assert "PageRank:  early -> late  0.500000 -> 0.200000 (delta -0.300000)" in text
    assert "Betweenness:  early -> late  0.000000 -> 0.300000 (delta +0.300000)" in text
    assert "Neighbors in late" in text
    assert "  b  9.000000  louvain 7  leiden 8" in text
    assert "  d  2.000000  louvain 1  leiden 2" in text
    # The run that does not contain the user stays in SQLite. Its graph is not parsed.
    assert loaded == ["late"]


def test_runs_flag_keeps_time_order(tmp_path, monkeypatch, capsys):
    _two_runs(tmp_path, monkeypatch)
    assert report.main(["--user", "a", "--runs", "late,early"]) == 0
    text = capsys.readouterr().out
    assert text.index("early (reddit,") < text.index("late (reddit_2012,")
    assert "Neighbors in late" in text
    assert "solo" not in text


def test_single_run_has_no_delta(tmp_path, monkeypatch, capsys):
    _two_runs(tmp_path, monkeypatch)
    assert report.main(["--user", "a", "--runs", "early"]) == 0
    text = capsys.readouterr().out
    assert "late" not in text
    assert "delta" not in text
    assert "Neighbors in early" in text
    assert "  b  3.000000  louvain 0  leiden 4" in text


def test_unknown_run_lists_saved_runs(tmp_path, monkeypatch, capsys):
    _two_runs(tmp_path, monkeypatch)
    assert report.main(["--user", "a", "--runs", "missing"]) == 1
    listed = capsys.readouterr().out
    assert "early reddit" in listed
    assert "PageRank" not in listed


def test_missing_user_prints_ten_closest_names(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    names = [
        "baaa", "caaa", "daaa",
        "abaa", "acaa", "adaa",
        "aaba", "aaca", "aada",
        "aaab", "aaac", "aaad",
        "aabb",
        "zzzz",
    ]
    graph = nx.Graph()
    graph.add_nodes_from(names)
    scores = {name: 1.0 / (index + 1) for index, name in enumerate(names)}
    _write_run(path, "names", "reddit", graph, {"pagerank": scores}, {"louvain": {name: 0 for name in names}, "leiden": {name: 0 for name in names}})
    assert report.main(["--user", "aaaa"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "similar names:"
    assert lines[1:] == ["  aaab", "  aaac", "  aaad", "  aaba", "  aaca", "  aada", "  abaa", "  acaa", "  adaa", "  baaa"]


def test_top_neighbor_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    graph = nx.Graph()
    nodes = [f"n{index:02d}" for index in range(1, 13)]
    for index, node in enumerate(nodes, start=1):
        # n01 is the strongest tie. n12 must fall off the ten-name list.
        graph.add_edge("user", node, weight=13 - index)
    metrics = {"pagerank": {"user": 1.0}, "degree": {"user": 1.0}, "betweenness": {"user": 0.0}, "closeness": {"user": 0.0}}
    membership = {"user": 0, **{node: 0 for node in nodes}}
    _write_run(path, "cap", "reddit", graph, metrics, {"louvain": membership, "leiden": membership})
    assert report.main(["--user", "user"]) == 0
    text = capsys.readouterr().out
    assert "  n01  12.000000  louvain 0  leiden 0" in text
    assert "n10" in text
    assert "n11" not in text
    assert "n12" not in text


def test_missing_metric_uses_query_scores(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    _write_run(
        path,
        "partial",
        "reddit",
        graph,
        {"pagerank": {"a": 0.7, "b": 0.3}},
        {"louvain": {"a": 1, "b": 1}, "leiden": {"a": 2, "b": 2}},
    )
    calls = []

    def fake(run_id, graph, metric):
        calls.append(metric)
        return {"a": 0.2, "b": 0.1}

    monkeypatch.setattr(report, "_metric_scores", fake)
    assert report.main(["--user", "a"]) == 0
    assert calls == ["degree", "betweenness", "closeness"]
    assert "Degree:       0.200000  (rank 1)" in capsys.readouterr().out


def test_percent_bands_use_rank_over_population():
    assert report._in_band(1, 100, 0.01)
    assert not report._in_band(2, 100, 0.01)
    assert report._in_band(10, 100, 0.10)
    assert not report._in_band(11, 100, 0.10)
    assert report._in_band(51, 5110, 0.01)
    assert not report._in_band(52, 5110, 0.01)
    assert report.levenshtein("akdas", "akdat") == 1
    assert report.levenshtein("kitten", "sitting") is None
    assert report.levenshtein("kitten", "sitting", limit=3) == 3
