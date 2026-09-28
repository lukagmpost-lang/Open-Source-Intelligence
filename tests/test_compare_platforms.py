import networkx as nx

from viz.compare_platforms import (
    PANELS,
    cap_by_pagerank,
    caption_line,
    display_label,
    panel_records,
    render_html,
)
from viz.interactive import populate_pyvis, write_pyvis
from pyvis.network import Network


def test_caption_uses_the_full_graph_counts():
    line = caption_line("Reddit 2008", 5110, 86268, 0.324677, 0.640181, 0.30)
    assert line == (
        "Reddit 2008 | 5110 nodes | 86268 edges | "
        "modularity 0.324677 | clustering 0.640181 | halving 0.30"
    )


def test_cap_keeps_the_highest_pagerank_nodes():
    graph = nx.path_graph(4)
    scores = {0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
    view = cap_by_pagerank(graph, scores, 2)
    assert set(view.nodes) == {1, 2}
    # The two kept nodes are neighbors, so the induced edge stays.
    assert view.number_of_edges() == 1


def test_bluesky_follow_ids_display_as_handles():
    assert display_label("bluesky:jay.bsky.team") == "jay.bsky.team"
    assert display_label("hadley") == "hadley"


def test_page_has_one_search_six_panels_and_the_controls():
    graph = nx.path_graph(2)
    scores = {0: 0.2, 1: 0.1}
    communities = {0: 1, 1: 2}
    degrees = {0: 1, 1: 1}
    nodes, edges = panel_records(graph, scores, communities, degrees)
    panels = []
    for spec in PANELS:
        panels.append(
            {
                "name": spec["name"],
                "caption": caption_line(spec["name"], 10, 20, 0.5, 0.25, spec["halving"]),
                "metrics": {
                    "clustering": 0.25,
                    "assortativity": -0.1,
                    "modularity": 0.5,
                    "halving": spec["halving"],
                },
                "nodes": nodes,
                "edges": edges,
            }
        )
    html = render_html(panels, vis_js="/* vis */")
    assert "Edge Semantics Determines Network Structure" in html
    assert html.count('class="network"') == 6
    assert 'id="search"' in html
    assert 'id="physics-toggle"' in html
    assert '<option value="community">community</option>' in html
    assert '<option value="pagerank">pagerank</option>' in html
    assert '<option value="degree">degree</option>' in html
    assert "Follow graphs, plus the SNAP friendship graph" in html
    assert "<h2>Co-participation</h2>" in html
    # One library, then the page script that builds every panel.
    assert html.count("<script>") == 2
    assert "colorCommunity" in html
    assert "halving point" in html


def test_single_graph_writer_still_uses_pagerank_size(tmp_path):
    graph = nx.path_graph(2)
    write_pyvis(graph, str(tmp_path / "one.html"), {0: 0.01, 1: 0.0}, {0: 3, 1: 3})
    text = (tmp_path / "one.html").read_text()
    assert "user: 0 | community: 3 | pagerank: 0.010000" in text
    net = Network()
    populate_pyvis(net, graph, {0: 0.01, 1: 0.0}, {0: 3, 1: 3})
    # 0.01 * 20000 is the single-graph size. The comparison page uses a smaller scale.
    assert net.nodes[0]["size"] == 200
