import pytest

from osi.router import route

# question, intent, and the params besides run. Unsupported questions expect low confidence.
_CASES = (
    ("top 5 by pagerank", "rank_nodes", {"metric": "pagerank", "top": 5}),
    ("top 3 degree", "rank_nodes", {"metric": "degree", "top": 3}),
    ("bridges", "rank_nodes", {"metric": "betweenness", "top": 10}),
    ("closest to everyone", "rank_nodes", {"metric": "closeness", "top": 10}),
    ("most important", "rank_nodes", {"metric": "pagerank", "top": 10}),
    ("most connected", "rank_nodes", {"metric": "degree", "top": 10}),
    ("brokers", "rank_nodes", {"metric": "betweenness", "top": 10}),
    ("communities", "list_communities", {"algorithm": "louvain"}),
    ("how many communities", "list_communities", {"algorithm": "louvain"}),
    ("what communities exist", "list_communities", {"algorithm": "louvain"}),
    ("health", "network_health", {}),
    ("how healthy", "network_health", {}),
    ("assortativity", "network_health", {}),
    ("power law", "network_health", {}),
    ("how fragile", "structural_criticality", {}),
    ("what happens if we remove the top", "structural_criticality", {}),
    ("resilience", "structural_criticality", {}),
    ("path from alice to bob", "connectivity", {"source": "alice", "target": "bob"}),
    ("distance between alice and bob", "connectivity", {"source": "alice", "target": "bob"}),
    ("are alice and bob connected", "connectivity", {"source": "alice", "target": "bob"}),
    ("tell me about alice", "explain_node", {"node": "alice"}),
    ("who is jay.bsky.team", "explain_node", {"node": "jay.bsky.team"}),
    ("describe carol", "explain_node", {"node": "carol"}),
    ("what is the weather", "unsupported", {}),
    ("", "unsupported", {}),
    ("TOP 5 BY PAGERANK", "rank_nodes", {"metric": "pagerank", "top": 5}),
    ("tell me about alice?", "explain_node", {"node": "alice"}),
    ("  path  from  alice  to  bob  ", "connectivity", {"source": "alice", "target": "bob"}),
)


@pytest.mark.parametrize("question,intent,extra", _CASES)
def test_route_parses_the_question(question, intent, extra):
    parsed = route(question, "simple-v1")
    assert parsed["intent"] == intent
    assert parsed["confidence"] == ("low" if intent == "unsupported" else "high")
    assert parsed["params"]["run"] == "simple-v1"
    for key, value in extra.items():
        assert parsed["params"][key] == value
