import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main


def _tiny_graph() -> nx.Graph:
    graph = nx.barbell_graph(3, 0)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def test_github_all_prints_pagerank_and_communities(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    out = tmp_path / "graph.json"
    code = main.main(["--source", "github", "--username", "octocat", "--analyze", "all", "--out", str(out)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "PageRank top 20:" in captured
    assert "Louvain communities:" in captured
    assert "Leiden communities:" in captured
    payload = json.loads(out.read_text())
    assert "nodes" in payload and "links" in payload
    assert len(payload["nodes"]) == 6


def test_robustness_flag_prints_removal_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    code = main.main(
        [
            "--source",
            "github",
            "--username",
            "octocat",
            "--analyze",
            "communities",
            "--robustness",
            "--out",
            str(tmp_path / "g.json"),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "strategy  ratio  remaining  largest  components  efficiency" in output
    assert "degree" in output and "betweenness" in output and "random" in output


def test_identity_flag_prints_layer_membership(monkeypatch, tmp_path, capsys):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"person_a": {"github": "ada", "reddit": "ada_r"}}), encoding="utf-8")
    github = nx.Graph()
    github.add_edge("github:ada", "github:grace", weight=2.0)
    reddit = nx.Graph()
    reddit.add_edge("reddit:ada_r", "reddit:linus", weight=1.0)

    def fake_layer(name, _args):
        return {"github": github, "reddit": reddit}.get(name)

    monkeypatch.setattr(main, "_load_named_layer", fake_layer)
    code = main.main(
        [
            "--identity",
            str(path),
            "--layers",
            "github,reddit,missing",
            "--analyze",
            "centrality",
            "--out",
            str(tmp_path / "g.json"),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "PageRank top 20 layer membership:" in output
    assert "person_a|github" in output and "layer github" in output
    assert "person_a|reddit" in output and "layer reddit" in output
    assert "cross-platform persons 1" in output
    assert "interlayer edges 1" in output
    code = main.main(
        [
            "--identity",
            str(path),
            "--layers",
            "github,reddit",
            "--normalize-layers",
            "--analyze",
            "centrality",
            "--out",
            str(tmp_path / "norm.json"),
        ]
    )
    normalized = capsys.readouterr().out
    assert code == 0
    assert "before normalization" in normalized
    assert "after normalization" in normalized
    after = normalized.split("after normalization", 1)[1]
    assert after.count("total_weight 1.000000") == 2


def test_communities_mode_skips_pagerank(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    called = {"pagerank": False}
    monkeypatch.setattr(main, "pagerank", lambda *args, **kwargs: called.__setitem__("pagerank", True))
    main.main(["--username", "octocat", "--analyze", "communities", "--out", str(tmp_path / "g.json")])
    output = capsys.readouterr().out
    assert called["pagerank"] is False
    assert "communities:" in output
    assert "PageRank" not in output


def test_snap_source_uses_the_loader(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "load_snap_facebook", _tiny_graph)
    main.main(["--source", "snap_facebook", "--analyze", "communities", "--out", str(tmp_path / "g.json")])
    assert "Louvain communities:" in capsys.readouterr().out


def test_compare_flag_prints_the_centrality_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    main.main(
        ["--username", "octocat", "--analyze", "communities", "--compare", "centrality", "--out", str(tmp_path / "g.json")]
    )
    output = capsys.readouterr().out
    assert "TOP NODES BY EACH CENTRALITY MEASURE:" in output
    assert "Rank | Degree | Betweenness | Closeness | PageRank" in output


def test_link_predict_flag_prints_shared_community(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    main.main(
        ["--username", "octocat", "--analyze", "communities", "--link-predict", "top=2", "--out", str(tmp_path / "g.json")]
    )
    output = capsys.readouterr().out
    assert "adamic_adar" in output
    assert "shared=" in output


def test_reddit_2012_source_uses_that_month(monkeypatch, tmp_path, capsys):
    def fake_layers(*args, **kwargs):
        assert kwargs.get("user_layer") == "reddit_2012_user"
        assert args[0] == "2012-08"
        return {"reddit_2012_user": _tiny_graph()}

    monkeypatch.setattr(main, "load_reddit_layers", fake_layers)
    main.main(["--source", "reddit_2012", "--analyze", "communities", "--out", str(tmp_path / "g.json")])
    assert "Louvain communities:" in capsys.readouterr().out


def test_reddit_source_uses_the_user_layer(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "load_reddit_layers", lambda: {"reddit_user": _tiny_graph(), "reddit_subreddit": _tiny_graph()})
    main.main(["--source", "reddit", "--analyze", "communities", "--out", str(tmp_path / "g.json")])
    assert "Louvain communities:" in capsys.readouterr().out


def test_plot_flag_writes_graph_png(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    monkeypatch.chdir(tmp_path)
    main.main(["--username", "octocat", "--analyze", "communities", "--plot", "--out", str(tmp_path / "g.json")])
    output = capsys.readouterr().out
    assert output.strip().endswith("Saved graph.png")
    image = tmp_path / "graph.png"
    assert image.is_file()
    assert image.stat().st_size > 1000


def test_interactive_uses_node_and_weight_limits(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_html(graph, output="graph.html", max_nodes=1000, min_edge_weight=2):
        seen["max_nodes"] = max_nodes
        seen["min_edge_weight"] = min_edge_weight
        return output

    monkeypatch.setattr(main, "to_interactive_html", fake_html)
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    main.main(
        [
            "--username",
            "octocat",
            "--analyze",
            "communities",
            "--interactive",
            "--max-nodes",
            "1000",
            "--min-edge-weight",
            "2",
            "--out",
            str(tmp_path / "g.json"),
        ]
    )
    assert seen == {"max_nodes": 1000, "min_edge_weight": 2.0}


def test_save_and_load_run_print_the_same_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    calls = {"n": 0}

    def fetch(username):
        calls["n"] += 1
        return _tiny_graph()

    monkeypatch.setattr(main, "fetch_github_graph", fetch)
    first_out = tmp_path / "a.json"
    second_out = tmp_path / "b.json"
    main.main(["--username", "octocat", "--save-run", "tiny", "--out", str(first_out)])
    first = capsys.readouterr().out
    main.main(["--load-run", "tiny", "--out", str(second_out)])
    second = capsys.readouterr().out
    assert calls["n"] == 1
    assert first == second
    assert first_out.read_text() == second_out.read_text()


def test_missing_run_warns_and_computes(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    code = main.main(
        ["--load-run", "missing", "--username", "octocat", "--analyze", "communities", "--out", str(tmp_path / "g.json")]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "warning:" in captured.err
    assert "computing fresh" in captured.err
    assert "Louvain communities:" in captured.out


def test_no_cache_rebuilds(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    calls = {"n": 0}

    def fetch(username):
        calls["n"] += 1
        return _tiny_graph()

    monkeypatch.setattr(main, "fetch_github_graph", fetch)
    main.main(["--username", "octocat", "--save-run", "tiny", "--analyze", "communities", "--out", str(tmp_path / "a.json")])
    capsys.readouterr()
    main.main(
        ["--load-run", "tiny", "--no-cache", "--username", "octocat", "--analyze", "communities", "--out", str(tmp_path / "b.json")]
    )
    assert calls["n"] == 2


def test_list_runs_prints_saved_ids(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: _tiny_graph())
    main.main(["--username", "octocat", "--save-run", "tiny", "--analyze", "communities", "--out", str(tmp_path / "a.json")])
    capsys.readouterr()
    code = main.main(["--list-runs"])
    assert code == 0
    assert "tiny github" in capsys.readouterr().out


def test_save_run_stores_standard_metrics(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    # Two components, so eigenvector must be skipped rather than aborting the save.
    disconnected = nx.Graph([(0, 1), (2, 3)])
    nx.set_edge_attributes(disconnected, 1.0, "weight")
    monkeypatch.setattr(main, "fetch_github_graph", lambda username: disconnected)
    code = main.main(
        ["--username", "octocat", "--analyze", "communities", "--save-run", "disc", "--out", str(tmp_path / "g.json")]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "eigenvector skipped" in captured.err
    from osi.store import load_metrics

    for metric in ("degree", "pagerank", "betweenness", "closeness"):
        assert set(load_metrics("disc", metric)) == {0, 1, 2, 3}
    assert load_metrics("disc", "eigenvector") == {}


def test_robustness_without_source_uses_the_newest_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    from osi.store import create_run, save_graph

    older = nx.Graph()
    older.add_node("old-node")
    newer = nx.Graph()
    newer.add_node("new-node")
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "old"}, "old", run_id="older", path=path)
    save_graph("older", "old", older, path=path)
    create_run("reddit_2012", {"layer": "new"}, "new", run_id="newer", path=path)
    save_graph("newer", "new", newer, path=path)
    # Same-second timestamps would tie. Pin the order so "newer" is unambiguously latest.
    import sqlite3

    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE runs SET created_at = ? WHERE id = ?", ("2020-01-01T00:00:00+00:00", "older"))
        conn.execute("UPDATE runs SET created_at = ? WHERE id = ?", ("2021-01-01T00:00:00+00:00", "newer"))
    seen = {}

    def fake_robustness(graph, **_kwargs):
        seen["nodes"] = set(graph.nodes)
        return {"baseline": {"remaining": 1, "largest": 1.0, "components": 1.0, "efficiency": 0.0}}

    monkeypatch.setattr(main, "robustness", fake_robustness)
    code = main.main(["--robustness", "--analyze", "communities", "--out", str(tmp_path / "g.json")])
    assert code == 0
    assert "using saved run newer" in capsys.readouterr().out
    assert seen["nodes"] == {"new-node"}


def test_robustness_without_source_or_saved_runs_explains_why(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "empty.db"))
    try:
        main.main(["--robustness", "--out", str(tmp_path / "g.json")])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("expected an empty-store error")
    assert "store is empty" in capsys.readouterr().err


def test_run_overrides_source(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    from osi.store import create_run, save_graph

    older = nx.Graph()
    older.add_node("old-node")
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "old"}, "old", run_id="older", path=path)
    save_graph("older", "old", older, path=path)

    def fetch(_username):
        raise AssertionError("source should not be built when --run is set")

    monkeypatch.setattr(main, "fetch_github_graph", fetch)
    code = main.main(
        [
            "--run",
            "older",
            "--source",
            "github",
            "--username",
            "octocat",
            "--analyze",
            "communities",
            "--out",
            str(tmp_path / "g.json"),
        ]
    )
    assert code == 0
    capsys.readouterr()
    payload = json.loads((tmp_path / "g.json").read_text())
    assert {node["id"] for node in payload["nodes"]} == {"old-node"}


def test_missing_run_does_not_rebuild(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "empty.db"))
    code = main.main(["--run", "missing", "--source", "github", "--username", "octocat", "--out", str(tmp_path / "g.json")])
    assert code == 1
    assert "run missing was not found" in capsys.readouterr().err


def test_health_reuses_the_store_and_compares_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    from osi.store import create_run, save_graph

    path = tmp_path / "store.db"
    for run_id, graph in (("left", nx.star_graph(4)), ("right", nx.path_graph(5))):
        nx.set_edge_attributes(graph, 1.0, "weight")
        create_run("reddit", {"layer": "users"}, run_id, run_id=run_id, path=path)
        save_graph(run_id, "users", graph, path=path)
    calls = {"n": 0}
    original = main.network_health

    def wrapped(graph):
        calls["n"] += 1
        return original(graph)

    monkeypatch.setattr(main, "network_health", wrapped)
    assert main.main(["--health", "--run", "left", "--out", str(tmp_path / "a.json")]) == 0
    first = capsys.readouterr().out
    assert "health left" in first
    assert "health compare" not in first
    assert "assortativity" in first
    assert calls["n"] == 1
    assert main.main(["--health", "--run", "left", "--out", str(tmp_path / "b.json")]) == 0
    second = capsys.readouterr().out
    assert calls["n"] == 1
    assert "health left" in second
    assert "health compare" not in second
    assert main.main(["--health", "--run", "right", "--out", str(tmp_path / "c.json")]) == 0
    third = capsys.readouterr().out
    assert calls["n"] == 2
    assert "health compare left right" in third


def test_bluesky_source_reads_the_saved_layer(monkeypatch, tmp_path, capsys):
    saved = _tiny_graph()

    def _refuse(_handles):
        raise AssertionError("bluesky source should not fetch when a layer is already stored")

    monkeypatch.setattr(main, "_saved_bluesky_layer", lambda: saved)
    monkeypatch.setattr(main, "build_bluesky_layer", _refuse)
    out = tmp_path / "graph.json"
    code = main.main(["--source", "bluesky", "--analyze", "communities", "--out", str(out)])
    assert code == 0
    assert "Louvain communities:" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert len(payload["nodes"]) == saved.number_of_nodes()


def test_github_multi_ego_source_reads_the_saved_layer(monkeypatch, tmp_path, capsys):
    saved = _tiny_graph()

    def _refuse(*_args, **_kwargs):
        raise AssertionError("github_multi_ego should not fetch")

    monkeypatch.setattr(main, "load_graph", lambda run_id, layer: saved)
    monkeypatch.setattr(main, "fetch_github_graph", _refuse)
    out = tmp_path / "graph.json"
    code = main.main(["--source", "github_multi_ego", "--analyze", "communities", "--out", str(out)])
    assert code == 0
    assert "Louvain communities:" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert len(payload["nodes"]) == saved.number_of_nodes()


def test_organic_sources_use_their_builders(monkeypatch, tmp_path, capsys):
    saved = _tiny_graph()

    def _github():
        return saved

    def _bluesky():
        return saved

    monkeypatch.setattr(main, "build_github_organic", _github)
    monkeypatch.setattr(main, "build_bluesky_organic", _bluesky)
    out = tmp_path / "graph.json"
    assert main.main(["--source", "github_organic", "--analyze", "communities", "--out", str(out)]) == 0
    assert main.main(["--source", "bluesky_organic", "--analyze", "communities", "--out", str(out)]) == 0
    assert "Louvain communities:" in capsys.readouterr().out


def test_github_requires_username():
    try:
        main.main(["--source", "github", "--analyze", "communities"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("expected a missing-username error")
