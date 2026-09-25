import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.analysis import adamic_adar, cpm_communities, jaccard, preferential_attachment, print_cpm_summary
from viz.interactive import to_interactive_html


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
    target = tmp_path / "graph.html"
    to_interactive_html(graph, str(target))
    text = target.read_text()
    assert "user: 0 | community:" in text
    assert "pagerank:" in text
    assert target.stat().st_size > 1000
