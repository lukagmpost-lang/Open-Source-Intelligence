import json

from osi.analyze import bridges, communities
from osi.app import Handler, analyze_graph
from osi.connectors.telegram import gifts_to_graph
from osi.graph import Graph
from osi.sample import sample_graph


def test_sample_has_two_communities_and_a_bridge():
    graph = sample_graph()
    found = communities(graph)
    assert len(found) >= 2
    ranked = bridges(graph, found)
    labels = {item["label"] for item in ranked}
    assert "ada" in labels


def test_import_merges_same_layer_weight_and_ignores_self_loops():
    graph = Graph.from_dict(
        {
            "nodes": [
                {"platform": "github", "account_id": "a", "label": "a"},
                {"platform": "github", "account_id": "b", "label": "b"},
            ],
            "edges": [
                {"source": "github:a", "target": "github:b", "layer": "follow", "weight": 1},
                {"source": "github:b", "target": "github:a", "layer": "follow", "weight": 2},
                {"source": "github:a", "target": "github:a", "layer": "follow", "weight": 5},
            ],
        }
    )
    assert len(graph.edges) == 1
    assert graph.edges[0].weight == 3


def test_telegram_keeps_only_displayed_gifts_with_a_visible_sender():
    graph = gifts_to_graph(
        "100",
        "ada",
        [
            {"is_saved": True, "sender_user": {"id": 200, "username": "grace"}},
            {"is_saved": False, "sender_user": {"id": 300, "username": "hidden"}},
            {"is_saved": True, "name_hidden": True, "sender_user": {"id": 400, "username": "anon"}},
            {"sender_user": {"id": 500, "username": "unknown"}},
        ],
    )
    assert [edge.source for edge in graph.edges] == ["telegram:200"]
    assert graph.nodes["telegram:200"].label == "grace"


def test_analyze_payload_shape():
    payload = analyze_graph(sample_graph())
    assert payload["communities"]
    assert "gift" in payload["layers"]


class _FakeHandler(Handler):
    def __init__(self):
        self.status = None
        self.body = b""
        self.headers = {}
        self.rfile = None
        self.wfile = None
        self.path = "/api/sample"

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        return


def test_github_fetch_caps_pages_and_marks_mutuals(monkeypatch):
    import networkx as nx

    from osi.connectors import github
    from osi.github_graph import fetch_github_graph
    from osi.graph import fetch_github_graph as exposed

    calls = []

    def fake_get(url, headers=None):
        calls.append(url)
        if url.endswith("/octocat"):
            return {"login": "octocat", "type": "User"}
        if "/followers" in url:
            return [{"login": "ada"}, {"login": "grace"}] if "page=1" in url else []
        if "/following" in url:
            return [{"login": "grace"}] if "page=1" in url else []
        raise AssertionError(url)

    monkeypatch.setattr("osi.github_graph.get_json", fake_get)
    nx_graph = fetch_github_graph("octocat")
    assert isinstance(nx_graph, nx.Graph)
    assert exposed is fetch_github_graph
    assert nx_graph.edges["octocat", "grace"]["relation"] == "mutual"
    assert nx_graph.edges["octocat", "ada"]["direction"] == "follower"
    graph = github.fetch_github("octocat")
    layers = {(edge.source, edge.target, edge.layer) for edge in graph.edges}
    assert ("github:octocat", "github:grace", "mutual") in layers
    assert ("github:ada", "github:octocat", "follow") in layers
    assert all("page=3" not in url for url in calls)


def test_handler_sample_and_import():
    handler = _FakeHandler()
    chunks = []

    class Writer:
        def write(self, data):
            chunks.append(data)

    handler.wfile = Writer()
    handler.do_GET()
    payload = json.loads(b"".join(chunks))
    assert handler.status == 200
    assert payload["graph"]["nodes"]

    chunks.clear()
    body = json.dumps(
        {
            "nodes": [
                {"platform": "github", "account_id": "a"},
                {"platform": "github", "account_id": "b"},
            ],
            "edges": [{"source": "github:a", "target": "github:b", "layer": "follow"}],
        }
    ).encode()

    class Reader:
        def read(self, _n):
            return body

    handler.path = "/api/import"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = Reader()
    handler.do_POST()
    imported = json.loads(b"".join(chunks))
    assert imported["communities"]
