"""Multislice communities across ordered graph snapshots.

The quality function is Mucha et al. (2010). Within a snapshot the usual
modularity term applies. The same node in neighboring snapshots is tied by a
coupling of strength omega. networkx_temporal optimizes that quality with
Leiden. If that package cannot be imported, the same Leiden coupling is run
through leidenalg directly.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def _with_weights(graph: nx.Graph) -> nx.Graph:
    """Copy with a numeric weight on every edge. The caller's graph is left as it was."""
    copied = graph.copy()
    for _left, _right, data in copied.edges(data=True):
        # A missing weight is one shared event, matching the co-participation graphs.
        data["weight"] = float(data.get("weight", 1.0))
    return copied


def _assignments_from_membership(graphs: list[nx.Graph], membership: list[list]) -> list[dict[Any, int]]:
    """Zip each snapshot's node order with the membership vector from that slice."""
    assignments = []
    for graph, labels in zip(graphs, membership):
        nodes = list(graph.nodes())
        if len(nodes) != len(labels):
            raise RuntimeError(
                f"membership length {len(labels)} does not match {len(nodes)} nodes"
            )
        # One id space spans every snapshot, so the same integer is the same community.
        assignments.append({node: int(label) for node, label in zip(nodes, labels)})
    return assignments


def _via_networkx_temporal(graphs: list[nx.Graph], omega: float) -> list[dict[Any, int]]:
    import networkx_temporal as nxt

    temporal = nxt.from_snapshots(graphs)
    # max_iter=-1 runs Leiden until a pass stops improving. seed keeps the four omegas comparable.
    membership = nxt.leiden_communities(
        temporal,
        interslice_weight=float(omega),
        weight="weight",
        device="cpu",
        seed=0,
        max_iter=-1,
    )
    return _assignments_from_membership(list(temporal), membership)


def _via_leidenalg(graphs: list[nx.Graph], omega: float) -> list[dict[Any, int]]:
    """Mucha coupling without networkx_temporal: Leiden on each slice plus an interslice layer."""
    import igraph as ig
    import leidenalg

    layers = []
    for graph in graphs:
        layer = ig.Graph.from_networkx(graph)
        # from_networkx stores the original name on _nx_name. That is the cross-slice identity.
        layer.vs["id"] = layer.vs["_nx_name"]
        layers.append(layer)
    membership, _improvement = leidenalg.find_partition_temporal(
        layers,
        leidenalg.ModularityVertexPartition,
        interslice_weight=float(omega),
        vertex_id_attr="id",
        weight_attr="weight",
        n_iterations=-1,
        seed=0,
    )
    return _assignments_from_membership(graphs, membership)


def multislice_communities(graphs: list[nx.Graph], omega: float = 1.0) -> list[dict[Any, int]]:
    """Community id per node, one dict per snapshot.

    omega is the inter-slice coupling C_jsr in Mucha et al. (2010). A larger
    omega pays more for giving a node the same community in neighboring
    snapshots. Zero coupling lets each snapshot cluster on its own.
    """
    prepared = [_with_weights(graph) for graph in graphs]
    try:
        return _via_networkx_temporal(prepared, omega)
    except ImportError:
        return _via_leidenalg(prepared, omega)


def multislice_modularity(graphs: list[nx.Graph], assignments: list[dict[Any, int]], omega: float = 1.0) -> float:
    """Mucha multislice modularity of an assignment at this omega."""
    import networkx_temporal as nxt

    prepared = [_with_weights(graph) for graph in graphs]
    temporal = nxt.from_snapshots(prepared)
    # The scorer wants one label list per snapshot, in that snapshot's node order.
    labels = [[assignment[node] for node in graph.nodes()] for graph, assignment in zip(prepared, assignments)]
    return float(
        nxt.modularity_multislice(
            temporal,
            labels,
            interslice_weight=float(omega),
            weight="weight",
        )
    )


def community_persistence(earlier: dict[Any, int], later: dict[Any, int]) -> list[tuple[int, int, int, float]]:
    """For each earlier community, the fraction of shared members who keep that community id.

    Members absent from the later snapshot are not in the fraction. A community
    with no shared members is omitted. Each row is
    (community id, size in the earlier snapshot, shared members, fraction).
    """
    members: dict[int, list] = {}
    for node, community in earlier.items():
        members.setdefault(int(community), []).append(node)
    rows = []
    for community, nodes in members.items():
        shared = [node for node in nodes if node in later]
        if not shared:
            continue
        # Same integer means the multislice solution did not move them apart.
        stayed = sum(1 for node in shared if int(later[node]) == community)
        rows.append((community, len(nodes), len(shared), stayed / len(shared)))
    rows.sort(key=lambda row: (-row[2], -row[1], row[0]))
    return rows


def mean_persistence(rows: list[tuple[int, int, int, float]]) -> float:
    """Persistence averaged over shared members, so a large community counts more."""
    shared = sum(row[2] for row in rows)
    if shared == 0:
        return 0.0
    return sum(row[2] * row[3] for row in rows) / shared
