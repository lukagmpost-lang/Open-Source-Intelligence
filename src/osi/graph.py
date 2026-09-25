"""Weighted multi-layer graph of public social relations.

``fetch_github_graph`` is the importable NetworkX builder. The script
``python -m osi.github_graph USERNAME`` uses the same function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


def node_id(platform: str, account_id: str) -> str:
    platform = platform.strip().lower()
    account_id = str(account_id).strip()
    if not platform or not account_id or ":" in platform:
        raise ValueError("platform and account_id are required, and platform cannot contain ':'")
    return f"{platform}:{account_id}"


@dataclass
class Node:
    id: str
    platform: str
    account_id: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    layer: str
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Graph:
    """Undirected multi-layer graph. Parallel edges stay distinct by layer."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(
        self,
        platform: str,
        account_id: str,
        label: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Node:
        identifier = node_id(platform, account_id)
        existing = self.nodes.get(identifier)
        if existing is None:
            existing = Node(
                id=identifier,
                platform=platform.strip().lower(),
                account_id=str(account_id).strip(),
                label=label or str(account_id),
                meta=dict(meta or {}),
            )
            self.nodes[identifier] = existing
        else:
            if label:
                existing.label = label
            if meta:
                existing.meta.update(meta)
        return existing

    def add_edge(
        self,
        source: str,
        target: str,
        layer: str,
        weight: float = 1.0,
        meta: dict[str, Any] | None = None,
    ) -> Edge | None:
        layer = layer.strip().lower()
        if not layer:
            raise ValueError("layer is required")
        if source == target or source not in self.nodes or target not in self.nodes:
            return None
        weight = float(weight)
        if weight <= 0:
            return None
        for edge in self.edges:
            if edge.layer == layer and {edge.source, edge.target} == {source, target}:
                edge.weight += weight
                if meta:
                    edge.meta.update(meta)
                return edge
        edge = Edge(source=source, target=target, layer=layer, weight=weight, meta=dict(meta or {}))
        self.edges.append(edge)
        return edge

    def merge(self, other: Graph) -> None:
        for node in other.nodes.values():
            self.add_node(node.platform, node.account_id, node.label, node.meta)
        for edge in other.edges:
            self.add_edge(edge.source, edge.target, edge.layer, edge.weight, edge.meta)

    def layers(self) -> list[str]:
        return sorted({edge.layer for edge in self.edges})

    def neighbors(self, identifier: str, layers: Iterable[str] | None = None) -> dict[str, float]:
        allowed = None if layers is None else set(layers)
        weights: dict[str, float] = {}
        for edge in self.edges:
            if allowed is not None and edge.layer not in allowed:
                continue
            other = None
            if edge.source == identifier:
                other = edge.target
            elif edge.target == identifier:
                other = edge.source
            if other is not None:
                weights[other] = weights.get(other, 0.0) + edge.weight
        return weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Graph:
        graph = cls()
        for raw in payload.get("nodes", []):
            graph.add_node(raw["platform"], raw["account_id"], raw.get("label"), raw.get("meta"))
        for raw in payload.get("edges", []):
            graph.add_edge(
                raw["source"],
                raw["target"],
                raw["layer"],
                raw.get("weight", 1),
                raw.get("meta"),
            )
        for link in payload.get("same_as", []):
            left = graph.add_node(link["platform"], link["account_id"], link.get("label"))
            right = graph.add_node(link["same_platform"], link["same_account_id"], link.get("same_label"))
            graph.add_edge(left.id, right.id, "same_as", 1.0, {"asserted_by": "input"})
        return graph


from osi.github_graph import fetch_github_graph as fetch_github_graph  # noqa: E402
