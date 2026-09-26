import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.analysis import adamic_adar, cpm_communities, jaccard, preferential_attachment, print_cpm_summary
from viz.interactive import filter_graph, to_interactive_html


def _path_with_a_chordless_gap() -> nx.Graph:
    graph = nx.Graph()
    # a and c share b, so that missing edge should score above d-c, which shares nobody.
    graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "d")])
    return graph


def test_link_scores_rank_a_shared_neighbor_first():
    graph = _path_with_a_chordless_gap()
    for scorer in (adamic_adar, jaccard, preferential_attachment):
        ranked = scorer(graph)
        assert ranked
        assert {tuple(sorted((left, right))) for left, right, _score in ranked} <= {
            tuple(sorted(pair)) for pair in nx.non_edges(graph)
        }
        assert ("a", "c") in {(left, right) for left, right, _score in ranked} or (
            "c",
            "a",
        ) in {(left, right) for left, right, _score in ranked}


def test_cpm_reports_overlap_counts(capsys):
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    membership = cpm_communities(graph, k=3)
    assert isinstance(membership[0], list)
    print_cpm_summary(membership)
    output = capsys.readouterr().out
    assert "nodes in 1 community:" in output
    assert "top 20 nodes by overlapping communities:" in output


def test_interactive_html_is_standalone(tmp_path):
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 2.0, "weight")
    target = tmp_path / "graph.html"
    to_interactive_html(graph, str(target), max_nodes=1000, min_edge_weight=2)
    text = target.read_text()
    assert "user: 0 | community:" in text
    assert "pagerank:" in text
    assert target.stat().st_size > 1000


def test_interactive_filter_keeps_top_nodes_and_heavy_edges(tmp_path, capsys):
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("b", "c", weight=5)
    graph.add_edge("a", "c", weight=5)
    graph.add_edge("a", "d", weight=1)
    graph.add_edge("d", "e", weight=4)
    target = tmp_path / "graph.html"
    to_interactive_html(graph, str(target), max_nodes=3, min_edge_weight=2)
    output = capsys.readouterr().out
    assert "filtered: 3 nodes, 3 edges" in output
    text = target.read_text()
    assert "user: a |" in text
    assert "user: e |" not in text


def test_filter_prunes_edges_before_the_node_cap():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("b", "c", weight=5)
    graph.add_edge("a", "c", weight=5)
    # This bridge is lighter than the cutoff, so the pair stays its own component.
    graph.add_edge("a", "p", weight=1)
    graph.add_edge("p", "q", weight=5)
    view = filter_graph(graph, max_nodes=2, min_edge_weight=2)
    assert set(view.nodes) == {"p", "q"}
    assert view.number_of_edges() == 1
    whole = filter_graph(graph, max_nodes=1000, min_edge_weight=2)
    assert nx.number_connected_components(whole) == 2
    assert whole.number_of_edges() == 4
