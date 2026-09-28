"""Link the same person across platform layers, then build one supra-graph."""

from __future__ import annotations

import json
import sys
from typing import Any

import networkx as nx

from osi.github_graph import fetch_github_graph
from osi.graph import build_supra_graph


def load_identity_map(path: str) -> dict[str, dict[str, str]]:
    """Load JSON mapping canonical person id to platform handles:
    {"person_a": {"github": "handle", "reddit": "handle"}, ...}"""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for person, accounts in payload.items():
        # A person with no platform object cannot be matched to a layer.
        if not isinstance(accounts, dict):
            continue
        kept = {}
        for layer, handle_name in accounts.items():
            if handle_name is None or layer is None:
                continue
            kept[str(layer)] = str(handle_name)
        if kept:
            cleaned[str(person)] = kept
    return cleaned


def _handle_index(layer_name: str, identity_map: dict[str, dict[str, str]]) -> dict[str, str]:
    """Map a handle on this layer to its person. Other layers are ignored."""
    index: dict[str, str] = {}
    for person, accounts in identity_map.items():
        if not isinstance(accounts, dict):
            continue
        handle_name = accounts.get(layer_name)
        # This person has no account on the layer being rewritten.
        if handle_name is None or handle_name == "":
            continue
        index[str(handle_name)] = str(person)
    return index


def apply_identity_map(
    G: nx.Graph,
    layer_name: str,
    identity_map: dict[str, dict[str, str]],
) -> nx.Graph:
    """Rewrite node ids from "{layer}:{handle}" to "{person_id}|{layer}".
    Nodes not in identity_map keep their original id.
    Preserve all node attributes."""
    handle_to_person = _handle_index(layer_name, identity_map)
    prefix = f"{layer_name}:"
    mapping: dict[Any, str] = {}
    for node in G.nodes:
        text = str(node)
        if text.startswith(prefix):
            handle_name = text[len(prefix) :]
        elif text in handle_to_person:
            # Stored layers use the bare handle, not the layer prefix.
            handle_name = text
        else:
            handle_name = None
        person = handle_to_person.get(handle_name) if handle_name else None
        # Missing handle: leave the node id alone.
        if person is None:
            continue
        mapping[node] = f"{person}|{layer_name}"
    # copy=True keeps the caller's graph intact and carries node and edge attributes over.
    return nx.relabel_nodes(G, mapping, copy=True)


def _map_for_rewritten_ids(
    identity_map: dict[str, dict[str, str]],
    layers: dict[str, nx.Graph],
) -> dict[str, dict[str, str]]:
    """Point each person at the id apply_identity_map already wrote, when that node exists."""
    aligned: dict[str, dict[str, str]] = {}
    for person, accounts in identity_map.items():
        if not isinstance(accounts, dict):
            continue
        kept: dict[str, str] = {}
        for layer_name in accounts:
            graph = layers.get(layer_name)
            # Layer was not loaded, so this handle cannot be joined.
            if graph is None:
                continue
            rewritten = f"{person}|{layer_name}"
            if rewritten not in graph:
                continue
            kept[layer_name] = rewritten
        if kept:
            aligned[str(person)] = kept
    return aligned


def _ids_supra_expects(graph: nx.Graph, layer_name: str, rewritten_ids: set[str]) -> nx.Graph:
    """Strip a layer prefix from nodes that were not rewritten, so supra-ids are not doubled."""
    prefix = f"{layer_name}:"
    mapping: dict[Any, str] = {}
    for node in graph.nodes:
        text = str(node)
        if text in rewritten_ids:
            continue
        if text.startswith(prefix):
            mapping[node] = text[len(prefix) :]
    if not mapping:
        return graph
    return nx.relabel_nodes(graph, mapping, copy=True)


def fetch_github_identities(identity_map: dict[str, dict[str, str]]) -> nx.Graph:
    """Followers and following for every GitHub handle in the map.

    One missing or private account is skipped. The other handles are still fetched.
    """
    combined = nx.Graph()
    seen: set[str] = set()
    for person, accounts in identity_map.items():
        if not isinstance(accounts, dict):
            continue
        handle = accounts.get("github")
        if not handle or handle in seen:
            continue
        seen.add(str(handle))
        try:
            piece = fetch_github_graph(str(handle))
        except (OSError, RuntimeError, ValueError) as error:
            # The GitHub side of this person is absent. Reddit can still be linked.
            print(f"warning: github handle {handle} for {person} skipped: {error}", file=sys.stderr)
            continue
        # A login can be the center of one ego graph and a neighbor in another.
        combined = nx.compose(combined, piece)
    return combined


def warn_missing_handles(layers: dict[str, nx.Graph | None], identity_map: dict[str, dict[str, str]]) -> None:
    """Warn when a mapped handle is not a node on the layer that was loaded."""
    for person, accounts in identity_map.items():
        if not isinstance(accounts, dict):
            continue
        for layer_name, handle in accounts.items():
            graph = layers.get(layer_name)
            if graph is None:
                continue
            # Archive layers store the bare handle. A built layer may use "reddit:handle".
            if handle in graph or f"{layer_name}:{handle}" in graph:
                continue
            print(f"warning: {layer_name} handle {handle} for {person} was not in the layer", file=sys.stderr)


def annotate_coverage(graph: nx.Graph) -> dict[str, list[str]]:
    """Mark each node github-only, reddit-only, or both, and record the identity persons.

    A person is "both" only when the supra-graph actually contains both of their layer nodes.
    Other nodes take the label of the single layer they were loaded from.
    """
    by_person: dict[str, list[Any]] = {}
    for node, data in graph.nodes(data=True):
        person = data.get("person")
        if person:
            by_person.setdefault(str(person), []).append(node)
    buckets: dict[str, list[str]] = {"both": [], "github-only": [], "reddit-only": [], "bluesky-only": []}
    for person, nodes in by_person.items():
        layers = {graph.nodes[node].get("layer") for node in nodes}
        if "github" in layers and "reddit" in layers:
            label = "both"
        elif "github" in layers:
            label = "github-only"
        elif "reddit" in layers:
            label = "reddit-only"
        elif "bluesky" in layers:
            label = "bluesky-only"
        else:
            continue
        buckets[label].append(person)
        for node in nodes:
            graph.nodes[node]["coverage"] = label
    for node, data in graph.nodes(data=True):
        if data.get("coverage"):
            continue
        layer = data.get("layer")
        if layer == "github":
            data["coverage"] = "github-only"
        elif layer == "reddit":
            data["coverage"] = "reddit-only"
        elif layer == "bluesky":
            data["coverage"] = "bluesky-only"
    listed = {label: sorted(people) for label, people in buckets.items()}
    # Saved with the graph so a later load can see who was linked without recomputing.
    graph.graph["coverage"] = listed
    return listed


# Every layer is scaled to this same sum before the layers are coupled.
_LAYER_WEIGHT_TARGET = 1.0


def _total_edge_weight(graph: nx.Graph) -> float:
    # An edge with no weight counts as 1, matching the loaders and the neighbor table.
    return sum(float(data.get("weight", 1.0)) for _left, _right, data in graph.edges(data=True))


def normalize_layers(
    layers: dict[str, nx.Graph | None],
    target_total: float = _LAYER_WEIGHT_TARGET,
) -> dict[str, nx.Graph]:
    """Scale each layer so its edge weights sum to the same target.

    The copy leaves the caller's graph alone, so a saved run used as a layer is not rewritten.
    """
    present = {str(name): graph for name, graph in layers.items() if graph is not None and name is not None}
    print("before normalization")
    totals: dict[str, float] = {}
    for name, graph in present.items():
        totals[name] = _total_edge_weight(graph)
        print(
            f"{name} nodes {graph.number_of_nodes()} edges {graph.number_of_edges()} "
            f"total_weight {totals[name]:.6f}"
        )
    scaled: dict[str, nx.Graph] = {}
    print("after normalization")
    for name, graph in present.items():
        # copy() so the factor below does not change the graph that was loaded from the store.
        copy = graph.copy()
        total = totals[name]
        if total > 0:
            factor = target_total / total
            for _left, _right, data in copy.edges(data=True):
                data["weight"] = float(data.get("weight", 1.0)) * factor
        else:
            # No edges to scale. Dividing by zero would invent a weight.
            print(f"warning: layer {name} has no edge weight to scale", file=sys.stderr)
        scaled[name] = copy
        print(
            f"{name} nodes {copy.number_of_nodes()} edges {copy.number_of_edges()} "
            f"total_weight {_total_edge_weight(copy):.6f}"
        )
    return scaled


def merge_identity_layers(
    layers: dict[str, nx.Graph | None],
    identity_map: dict[str, dict[str, str]],
    interlayer_weight: float = 1.0,
) -> nx.Graph:
    """Apply identity across each layer, then call build_supra_graph.
    Returns one merged nx.Graph."""
    prepared: dict[str, nx.Graph] = {}
    for layer_name, graph in layers.items():
        # A missing layer is skipped. The other layers still merge.
        if graph is None or layer_name is None:
            continue
        prepared[str(layer_name)] = apply_identity_map(graph, str(layer_name), identity_map)
    if not prepared:
        return nx.Graph()
    aligned = _map_for_rewritten_ids(identity_map, prepared)
    rewritten_ids = {handle for accounts in aligned.values() for handle in accounts.values()}
    bare = {
        layer_name: _ids_supra_expects(graph, layer_name, rewritten_ids)
        for layer_name, graph in prepared.items()
    }
    return build_supra_graph(bare, aligned, interlayer_weight=interlayer_weight)
