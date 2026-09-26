"""Link the same person across platform layers, then build one supra-graph."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

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
