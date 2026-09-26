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


def test_bad_query_and_missing_run(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    assert query.main(["--run", "tiny", "show everything"]) == 1
    assert "top N by METRIC" in capsys.readouterr().out
    assert query.main(["--run", "missing", "top 1 by degree"]) == 1
    assert "tiny reddit" in capsys.readouterr().out
