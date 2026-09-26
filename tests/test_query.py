import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import query
from osi.store import create_run, save_communities, save_graph, save_metrics


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    graph = nx.Graph()
    graph.add_edge("akdas", "yellowking", weight=3)
    graph.add_edge("akdas", "sjs", weight=1)
    create_run("reddit", {"layer": "reddit_user"}, "tiny", run_id="tiny", path=tmp_path / "store.db")
    save_graph("tiny", "reddit_user", graph, path=tmp_path / "store.db")
    save_metrics("tiny", {"akdas": 0.5, "yellowking": 0.3, "sjs": 0.2}, metric="pagerank", path=tmp_path / "store.db")
    save_communities("tiny", "louvain", {"akdas": 0, "yellowking": 0, "sjs": 1}, path=tmp_path / "store.db")


def test_queries_print_a_table(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    assert query.main(["--run", "tiny", "top 2 by pagerank"]) == 0
    assert capsys.readouterr().out.splitlines() == ["rank  node        pagerank", "1     akdas       0.500000", "2     yellowking  0.300000"]
    assert query.main(["--run", "tiny", "top 1 in 0 by pagerank"]) == 0
    assert "akdas" in capsys.readouterr().out
    assert query.main(["--run", "tiny", "neighbors of akdas"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[0] == "yellowking"
    assert query.main(["--run", "tiny", "community of yellowking"]) == 0
    assert capsys.readouterr().out.splitlines()[1] == "yellowking  0"
    assert query.main(["--run", "tiny", "communities by size"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split() == ["0", "2"]
    assert query.main(["--run", "tiny", "nodes where pagerank > 0.25"]) == 0
    text = capsys.readouterr().out
    assert "akdas" in text and "sjs" not in text


def _path_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    graph = nx.Graph()
    # The direct tie is stronger than the two-hop chain, and it is the one-hop path.
    graph.add_edge("akdas", "yellowking", weight=3)
    graph.add_edge("akdas", "sjs", weight=1)
    graph.add_edge("sjs", "yellowking", weight=1)
    graph.add_edge("akdas", "malcontent", weight=2)
    graph.add_node("solo")
    create_run("reddit", {"layer": "reddit_user"}, "paths", run_id="paths", path=tmp_path / "store.db")
    save_graph("paths", "reddit_user", graph, path=tmp_path / "store.db")


def test_path_queries(tmp_path, monkeypatch, capsys):
    _path_store(tmp_path, monkeypatch)
    assert query.main(["--run", "paths", "path from akdas to yellowking"]) == 0
    assert capsys.readouterr().out.splitlines() == ["akdas  yellowking  3.000000"]
    assert query.main(["--run", "paths", "paths from akdas to yellowking limit 2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "akdas yellowking"
    assert len(lines) == 2
    assert "sjs" in lines[1]
    assert query.main(["--run", "paths", "distance between akdas and yellowking"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert query.main(["--run", "paths", "neighborhood of akdas radius 1"]) == 0
    text = capsys.readouterr().out.splitlines()
    assert text[0] == "4"
    assert text[1:] == ["akdas  0", "malcontent  1", "sjs  1", "yellowking  1"]


def test_path_edges(tmp_path, monkeypatch, capsys):
    _path_store(tmp_path, monkeypatch)
    assert query.main(["--run", "paths", "path from akdas to solo"]) == 1
    assert capsys.readouterr().out.strip() == "no path (disconnected)"
    assert query.main(["--run", "paths", "distance between akdas and solo"]) == 1
    assert capsys.readouterr().out.strip() == "inf"
    assert query.main(["--run", "paths", "path from akdat to yellowking"]) == 1
    missing = capsys.readouterr().out
    assert "similar names:" in missing and "akdas" in missing
    assert query.main(["--run", "paths", "neighborhood of akdas radius 6"]) == 0
    assert "warning: radius above 5 may be slow" in capsys.readouterr().err


def test_bad_query_and_missing_run(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    assert query.main(["--run", "tiny", "show everything"]) == 1
    assert "top N by METRIC" in capsys.readouterr().out
    assert query.main(["--run", "missing", "top 1 by degree"]) == 1
    assert "tiny reddit" in capsys.readouterr().out
