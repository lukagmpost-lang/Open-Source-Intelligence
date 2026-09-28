"""Public GitHub followers and following via the REST API."""

from __future__ import annotations

import networkx as nx

from osi.github_graph import fetch_github_graph
from osi.graph import Graph


def _to_graph(nx_graph: nx.Graph) -> Graph:
    graph = Graph()
    center = next(node for node, data in nx_graph.nodes(data=True) if data.get("center"))
    for login, data in nx_graph.nodes(data=True):
        graph.add_node("github", login, data.get("label") or login, {"public": True})
    root = f"github:{center}"
    for _left, other, data in nx_graph.edges(center, data=True):
        direction = data.get("direction")
        if direction in ("follower", "mutual"):
            graph.add_edge(f"github:{other}", root, "follow", 1.0)
        if direction in ("following", "mutual"):
            graph.add_edge(root, f"github:{other}", "follow", 1.0)
        if direction == "mutual":
            graph.add_edge(root, f"github:{other}", "mutual", 1.0)
    return graph


def fetch_github(username: str) -> Graph:
    return _to_graph(fetch_github_graph(username))
