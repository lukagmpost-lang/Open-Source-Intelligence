import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import extract
from osi.store import create_run, save_communities, save_graph, save_metrics


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    graph = nx.Graph()
    graph.add_edge("akdas", "yellowking", weight=3)
    graph.add_edge("akdas", "sjs", weight=1)
    graph.add_node("solo")
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "reddit_user"}, "tiny", run_id="tiny", path=path)
    save_graph("tiny", "reddit_user", graph, path=path)
    save_metrics("tiny", {"akdas": 0.5, "yellowking": 0.3, "sjs": 0.2, "solo": 0.1}, metric="pagerank", path=path)
    save_communities("tiny", "louvain", {"akdas": 0, "yellowking": 0, "sjs": 1, "solo": 2}, path=path)


def test_community_extract_writes_induced_json(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    target = tmp_path / "c0.json"
    assert extract.main(["--run", "tiny", "--community", "0", "--format", "json", "--out", str(target)]) == 0
    payload = json.loads(target.read_text())
    assert {node["id"] for node in payload["nodes"]} == {"akdas", "yellowking"}
    assert len(payload["links"]) == 1
    text = capsys.readouterr().out
    assert "nodes 2" in text and "edges 1" in text and "communities 1" in text
    assert text.splitlines()[3] == "1 akdas 0.500000"


def test_radius_extract_and_errors(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    target = tmp_path / "near.json"
    assert extract.main(["--run", "tiny", "--node", "sjs", "--radius", "1", "--format", "json", "--out", str(target)]) == 0
    payload = json.loads(target.read_text())
    assert {node["id"] for node in payload["nodes"]} == {"sjs", "akdas"}
    assert "communities 2" in capsys.readouterr().out
    assert extract.main(["--run", "missing", "--community", "0", "--out", str(tmp_path / "x.graphml")]) == 1
    assert "tiny reddit" in capsys.readouterr().out
    assert extract.main(["--run", "tiny", "--community", "9", "--out", str(tmp_path / "x.graphml")]) == 1
    assert "0 2" in capsys.readouterr().out
    assert extract.main(["--run", "tiny", "--node", "akdat", "--radius", "1", "--out", str(tmp_path / "x.graphml")]) == 1
    assert "akdas" in capsys.readouterr().out
    assert extract.main(["--run", "tiny", "--node", "solo", "--radius", "6", "--format", "graphml", "--out", str(tmp_path / "solo.graphml")]) == 0
    err = capsys.readouterr().err
    assert "warning: radius above 5 may be slow" in err
    assert (tmp_path / "solo.graphml").exists()
    html = tmp_path / "c0.html"
    assert extract.main(["--run", "tiny", "--community", "0", "--format", "html", "--out", str(html)]) == 0
    page = html.read_text()
    # Search bar comes from PyVis select_menu; color title carries the stored Louvain id.
    assert "user: akdas | community: 0 | pagerank: 0.500000" in page
    assert "user: sjs |" not in page
