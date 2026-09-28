import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.loader import load_edge_list, load_graphml
from osi.store import load_graph
import main


def test_csv_keeps_quoted_fields_and_skips_a_bad_weight(tmp_path, capsys):
    path = tmp_path / "edges.csv"
    path.write_text(
        'source,target,weight\n"ada, lovelace",grace,2\nhub,grace,\nbad,row,nope\n',
        encoding="utf-8",
    )
    graph = load_edge_list(path)
    assert set(graph.nodes) == {"ada, lovelace", "grace", "hub"}
    assert graph["ada, lovelace"]["grace"]["weight"] == 2.0
    assert graph["hub"]["grace"]["weight"] == 1.0
    captured = capsys.readouterr()
    assert "3 nodes, 2 edges, 1 connected component" in captured.out
    assert "skipped row" in captured.err


def test_tsv_and_json_list_and_node_link(tmp_path, capsys):
    tsv = tmp_path / "edges.tsv"
    tsv.write_text("source\ttarget\na\tb\nb\tc\n", encoding="utf-8")
    graph = load_edge_list(tsv)
    assert graph.number_of_edges() == 2
    capsys.readouterr()

    listing = tmp_path / "edges.json"
    listing.write_text(
        json.dumps([{"source": "a", "target": "b", "weight": 1.5}, {"source": "b", "target": "c"}]),
        encoding="utf-8",
    )
    loaded = load_edge_list(listing)
    assert loaded["a"]["b"]["weight"] == 1.5
    assert loaded["b"]["c"]["weight"] == 1.0

    document = tmp_path / "nodelink.json"
    document.write_text(json.dumps(nx.node_link_data(loaded, edges="links")), encoding="utf-8")
    roundtrip = load_edge_list(document)
    assert set(roundtrip.edges) == {("a", "b"), ("b", "c")}


def test_graphml_keeps_attributes(tmp_path, capsys):
    path = tmp_path / "graph.graphml"
    graph = nx.Graph()
    graph.add_node("ada", role="hub")
    graph.add_edge("ada", "grace", relation="knows", weight=2.0)
    nx.write_graphml(graph, path)
    loaded = load_graphml(path)
    assert loaded.nodes["ada"]["role"] == "hub"
    assert loaded.edges["ada", "grace"]["relation"] == "knows"
    assert "2 nodes, 1 edge, 1 connected component" in capsys.readouterr().out


def test_main_saves_each_format(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("source,target\na,b\nb,c\n", encoding="utf-8")
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps([{"source": "a", "target": "b"}, {"source": "b", "target": "d"}]), encoding="utf-8")
    graphml_path = tmp_path / "test.graphml"
    sample = nx.Graph()
    sample.add_edge("a", "e")
    nx.write_graphml(sample, graphml_path)
    out = tmp_path / "out.json"

    for path, run_id, expected in (
        (csv_path, "corp-csv", {"a", "b", "c"}),
        (json_path, "corp-json", {"a", "b", "d"}),
        (graphml_path, "corp-graphml", {"a", "e"}),
    ):
        code = main.main(["--source", "file", "--path", str(path), "--save-run", run_id, "--out", str(out)])
        assert code == 0
        stored = load_graph(run_id, "file")
        assert stored is not None
        assert set(stored.nodes) == expected
    text = capsys.readouterr().out
    assert "saved run corp-csv" in text
    assert "saved run corp-json" in text
    assert "saved run corp-graphml" in text
