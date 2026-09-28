"""Local viewer. The sample is a NetworkX graph. Platform fetch is gone."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from osi.analysis import louvain_communities
from osi.sample import sample_graph

STATIC = Path(__file__).with_name("static")
MAX_BODY = 1_000_000
# Platform fetch left with the connectors. Import of the old multi-layer JSON left too.
_REMOVED = "platform fetch was removed"


def graph_view(graph) -> dict[str, Any]:
    """Shape a NetworkX graph the way the static page draws it."""
    membership = louvain_communities(graph)
    grouped: dict[int, list[Any]] = {}
    for node, community in membership.items():
        grouped.setdefault(int(community), []).append(node)
    communities = []
    for community_id, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        ordered = sorted(members, key=str)
        communities.append(
            {
                "id": community_id,
                "size": len(ordered),
                "members": [{"id": str(member), "label": str(member)} for member in ordered],
            }
        )
    member_of = {member["id"]: community["id"] for community in communities for member in community["members"]}
    bridges = []
    for node in graph.nodes:
        touched = {member_of.get(str(node))}
        for neighbor in graph.neighbors(node):
            touched.add(member_of.get(str(neighbor)))
        touched.discard(None)
        # A bridge is a node whose neighbors are not all in its own community.
        if len(touched) < 2:
            continue
        bridges.append(
            {
                "id": str(node),
                "label": str(node),
                "communities": sorted(touched),
                "span": len(touched),
            }
        )
    bridges.sort(key=lambda item: (-item["span"], item["label"].lower()))
    return {
        "graph": {
            "nodes": [{"id": str(node), "label": str(node)} for node in graph.nodes],
            "edges": [{"source": str(left), "target": str(right)} for left, right in graph.edges],
        },
        "layers": ["sample"],
        "communities": communities,
        "bridges": bridges,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = (STATIC / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/sample":
            self._json(200, graph_view(sample_graph()))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            self._read_json()
            if path == "/api/import" or path.startswith("/api/fetch/"):
                self._json(410, {"error": _REMOVED})
                return
            self._json(404, {"error": "not found"})
        except Exception as error:  # noqa: BLE001 - surface a single client error
            self._json(400, {"error": str(error)})


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"osi listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
