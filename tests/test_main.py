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


def test_github_requires_username():
    try:
        main.main(["--source", "github", "--analyze", "communities"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("expected a missing-username error")
