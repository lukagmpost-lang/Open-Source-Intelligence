"""Local viewer for a multi-layer public graph."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from osi.analyze import bridges, communities
from osi.connectors.github import fetch_github
from osi.connectors.reddit import fetch_reddit
from osi.connectors.spotify import fetch_spotify
from osi.connectors.steam import fetch_steam
from osi.connectors.telegram import fetch_telegram_gifts
from osi.graph import Graph
from osi.sample import sample_graph

STATIC = Path(__file__).with_name("static")
MAX_BODY = 1_000_000


def analyze_graph(graph: Graph, layers: list[str] | None = None) -> dict[str, Any]:
    selected = layers or None
    found = communities(graph, selected)
    return {
        "graph": graph.to_dict(),
        "layers": graph.layers(),
        "communities": found,
        "bridges": bridges(graph, found, selected),
    }


def _fetch(kind: str, query: dict[str, Any]) -> Graph:
    if kind == "github":
        return fetch_github(str(query.get("username") or ""))
    if kind == "reddit":
        return fetch_reddit(str(query.get("username") or ""))
    if kind == "steam":
        return fetch_steam(str(query.get("steam_id") or ""))
    if kind == "spotify":
        return fetch_spotify()
    if kind == "telegram":
        return fetch_telegram_gifts(str(query.get("user_id") or ""))
    raise ValueError(f"unknown source {kind}")


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
            self._json(200, analyze_graph(sample_graph()))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/import":
                graph = Graph.from_dict(payload)
                layers = payload.get("layers")
                selected = layers if isinstance(layers, list) else None
                self._json(200, analyze_graph(graph, selected))
                return
            if path.startswith("/api/fetch/"):
                graph = _fetch(path.removeprefix("/api/fetch/"), payload)
                self._json(200, analyze_graph(graph))
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
