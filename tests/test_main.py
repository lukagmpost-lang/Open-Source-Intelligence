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


def test_github_requires_username():
    try:
        main.main(["--source", "github", "--analyze", "communities"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("expected a missing-username error")
