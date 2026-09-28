"""Community detection and bridge ranking on a multi-layer graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from osi.graph import Graph


def communities(graph: Graph, layers: Iterable[str] | None = None, rounds: int = 25) -> list[dict[str, Any]]:
    """Weighted label propagation. Returns communities sorted by size."""
    nodes = list(graph.nodes)
    labels = {node: index for index, node in enumerate(nodes)}
    allowed = None if layers is None else list(layers)
    for _ in range(rounds):
        changed = False
        for node in nodes:
            scores: dict[int, float] = {}
            for neighbor, weight in graph.neighbors(node, allowed).items():
                label = labels[neighbor]
                scores[label] = scores.get(label, 0.0) + weight
            if not scores:
                continue
            best = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break

    grouped: dict[int, list[str]] = defaultdict(list)
    for node, label in labels.items():
        grouped[label].append(node)

    result = []
    for index, members in enumerate(sorted(grouped.values(), key=len, reverse=True)):
        members = sorted(members, key=lambda item: graph.nodes[item].label.lower())
        result.append(
            {
                "id": index,
                "size": len(members),
                "members": [
                    {"id": member, "label": graph.nodes[member].label, "platform": graph.nodes[member].platform}
                    for member in members
                ],
            }
        )
    return result


def bridges(graph: Graph, community_list: list[dict[str, Any]], layers: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Rank accounts whose neighbors sit in more than one community."""
    membership = {}
    for community in community_list:
        for member in community["members"]:
            membership[member["id"]] = community["id"]
    allowed = None if layers is None else list(layers)
    ranked = []
    for node in graph.nodes:
        touched = {membership[node]}
        for neighbor in graph.neighbors(node, allowed):
            touched.add(membership.get(neighbor, membership[node]))
        if len(touched) < 2:
            continue
        ranked.append(
            {
                "id": node,
                "label": graph.nodes[node].label,
                "platform": graph.nodes[node].platform,
                "communities": sorted(touched),
                "span": len(touched),
            }
        )
    ranked.sort(key=lambda item: (-item["span"], item["label"].lower()))
    return ranked
