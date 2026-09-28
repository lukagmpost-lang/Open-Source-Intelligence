from osi.app import graph_view
from osi.sample import sample_graph


def test_sample_is_a_small_graph_with_a_hub():
    graph = sample_graph()
    assert 30 <= graph.number_of_nodes() <= 50
    assert graph.number_of_edges() > graph.number_of_nodes()
    degrees = dict(graph.degree())
    hub = max(degrees, key=degrees.get)
    assert hub == "hub"
    assert degrees[hub] > max(degree for node, degree in degrees.items() if node != hub)
    view = graph_view(graph)
    assert {node["id"] for node in view["graph"]["nodes"]} == {str(node) for node in graph.nodes}
    assert len(view["communities"]) >= 2
    assert any(bridge["id"] == "hub" for bridge in view["bridges"])
    assert view["layers"] == ["sample"]
