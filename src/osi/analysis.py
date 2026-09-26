"""Network measures on a public social graph.

The score functions do not print. ``cross_reference`` is the exception:
it prints which community the top hubs fall in.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities as _louvain_communities


def _by_score(scores: dict[Any, float]) -> dict[Any, float]:
    return dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))


def degree_centrality(G: nx.Graph) -> dict[Any, float]:
    return _by_score(nx.degree_centrality(G))


def pagerank(G: nx.Graph, weight: str = "weight") -> dict[Any, float]:
    return _by_score(nx.pagerank(G, weight=weight))


def _as_igraph(G: nx.Graph, weight: str | None):
    """Copy an undirected NetworkX graph into igraph, optionally with edge weights."""
    import igraph as ig

    names = list(G.nodes)
    index = {node: position for position, node in enumerate(names)}
    edges = []
    weights: list[float] = []
    for left, right, data in G.edges(data=True):
        edges.append((index[left], index[right]))
        if weight is not None:
            weights.append(float(data.get(weight, 1.0)))
    graph = ig.Graph(n=len(names), edges=edges, directed=False)
    if weight is not None and weights:
        graph.es["weight"] = weights
    return names, graph


def betweenness_centrality(G: nx.Graph, weight: str = "weight") -> dict[Any, float]:
    # Exact betweenness. igraph implements the same sum as NetworkX, in C++,
    # which is what makes the August 2012 graph finish.
    node_count = G.number_of_nodes()
    if node_count < 3:
        return _by_score({node: 0.0 for node in G.nodes})
    try:
        names, graph = _as_igraph(G, weight)
        raw = graph.betweenness(directed=False, weights="weight" if graph.ecount() else None)
    except ImportError:
        return _by_score(nx.betweenness_centrality(G, weight=weight))
    # Undirected NetworkX divides by the number of unordered node pairs.
    scale = 2.0 / ((node_count - 1) * (node_count - 2))
    return _by_score({names[index]: float(raw[index]) * scale for index in range(node_count)})


def closeness_centrality(G: nx.Graph) -> dict[Any, float]:
    # Unweighted. Edge weight is tie strength, and closeness would treat it as distance.
    node_count = G.number_of_nodes()
    if node_count <= 1:
        return _by_score({node: 0.0 for node in G.nodes})
    try:
        names, graph = _as_igraph(G, None)
        raw = graph.closeness(mode="all", normalized=True)
        membership = graph.connected_components(mode="weak").membership
    except ImportError:
        return _by_score(nx.closeness_centrality(G))
    component_sizes: dict[int, int] = {}
    for component in membership:
        component_sizes[component] = component_sizes.get(component, 0) + 1
    scores: dict[Any, float] = {}
    for index, node in enumerate(names):
        value = raw[index]
        component_size = component_sizes[membership[index]]
        # NetworkX wf_improved scales by the share of nodes this component can reach.
        if value != value or component_size <= 1:
            scores[node] = 0.0
        else:
            scores[node] = float(value) * (component_size - 1) / (node_count - 1)
    return _by_score(scores)


def _community_index(groups: list[set[Any]]) -> dict[Any, int]:
    ordered = sorted(groups, key=lambda members: (-len(members), str(sorted(members, key=str))))
    scores: dict[Any, int] = {}
    for index, members in enumerate(ordered):
        for node in members:
            scores[node] = index
    return dict(sorted(scores.items(), key=lambda item: (item[1], str(item[0]))))


def louvain_communities(G: nx.Graph, weight: str = "weight") -> dict[Any, int]:
    groups = _louvain_communities(G, weight=weight, seed=0)
    return _community_index([set(group) for group in groups])


def leiden_communities(G: nx.Graph, weight: str = "weight") -> dict[Any, int]:
    # leidenalg needs python-igraph. If that import fails, Louvain is the stand-in.
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return louvain_communities(G, weight=weight)

    igraph = ig.Graph.from_networkx(G)
    edge_weight = weight if weight in igraph.es.attributes() else None
    partition = leidenalg.find_partition(
        igraph,
        leidenalg.ModularityVertexPartition,
        weights=edge_weight,
        seed=0,
    )
    names = igraph.vs["_nx_name"]
    groups: dict[int, set[Any]] = {}
    for index, community in enumerate(partition.membership):
        groups.setdefault(community, set()).add(names[index])
    return _community_index(list(groups.values()))


# Girvan-Newman removes the highest-betweenness edge over and over.
# That is O(m^2 n), so a few thousand nodes is already too slow to finish.
GIRVAN_NEWMAN_NODE_LIMIT = 1000
# Stop after this many splits and keep the split with the best modularity.
GIRVAN_NEWMAN_MAX_ITER = 20


def _ranks(scores: dict[Any, float]) -> tuple[dict[Any, int], list[Any]]:
    # Highest score is rank 1. Equal scores break ties by node text so the
    # order does not change between runs.
    ordered = sorted(scores, key=lambda node: (-scores[node], str(node)))
    return {node: index for index, node in enumerate(ordered, start=1)}, ordered


def compare_centralities(G: nx.Graph) -> dict[str, Any]:
    """Run several centrality measures and rank every node in each one.

    The result is a dict of columns, like a table: ``rank`` maps each node
    to its 1-based place in every measure, and ``order`` lists nodes from
    strongest to weakest for that measure.
    """
    measures = {
        "degree": degree_centrality(G),
        # Weight is the edge attribute. NetworkX treats it as path length.
        "betweenness": betweenness_centrality(G, weight="weight"),
        "closeness": closeness_centrality(G),
        "pagerank": pagerank(G, weight="weight"),
    }
    try:
        # NumPy solver. It can fail on a graph that is not a single component.
        measures["eigenvector"] = _by_score(nx.eigenvector_centrality_numpy(G, weight="weight"))
    except Exception as error:  # noqa: BLE001 - report the failure instead of inventing scores
        measures["eigenvector"] = {}
        eigenvector_error = f"{type(error).__name__}: {error}"
    else:
        eigenvector_error = None

    rank: dict[Any, dict[str, int | None]] = {node: {} for node in G.nodes}
    order: dict[str, list[Any]] = {}
    for name, scores in measures.items():
        places, ordered = _ranks(scores) if scores else ({}, [])
        order[name] = ordered
        for node in rank:
            rank[node][name] = places.get(node)
    return {
        "columns": list(measures),
        "rank": rank,
        "order": order,
        "eigenvector_error": eigenvector_error,
    }


def _partition_record(G: nx.Graph, assignment: dict[Any, int], weight: str = "weight") -> dict[str, Any]:
    grouped: dict[int, set[Any]] = {}
    for node, community in assignment.items():
        grouped.setdefault(community, set()).add(node)
    groups = list(grouped.values())
    return {
        "skipped": False,
        "count": len(groups),
        "modularity": float(nx.community.modularity(G, groups, weight=weight)) if groups else 0.0,
        "assignment": assignment,
    }


def _girvan_newman(G: nx.Graph, weight: str = "weight") -> dict[str, Any]:
    node_count = G.number_of_nodes()
    if node_count > GIRVAN_NEWMAN_NODE_LIMIT:
        return {
            "skipped": True,
            "count": None,
            "modularity": None,
            "assignment": {},
            "message": (
                f"Girvan-Newman skipped: this graph has {node_count} nodes. "
                f"It is O(m^2 n), so it only runs on graphs of at most {GIRVAN_NEWMAN_NODE_LIMIT} nodes."
            ),
        }
    # Each step yields a finer split. We keep only the first few and choose
    # the one with the highest modularity.
    best_groups: list[set[Any]] | None = None
    best_score = float("-inf")
    for index, communities in enumerate(nx.community.girvan_newman(G)):
        if index >= GIRVAN_NEWMAN_MAX_ITER:
            break
        groups = [set(community) for community in communities]
        score = nx.community.modularity(G, groups, weight=weight)
        if score > best_score:
            best_score = score
            best_groups = groups
    if not best_groups:
        best_groups = [{node} for node in G.nodes]
    return _partition_record(G, _community_index(best_groups), weight)


def compare_communities(G: nx.Graph) -> dict[str, Any]:
    """Compare Louvain, Leiden, and Girvan-Newman on one graph.

    Girvan-Newman is skipped, with a message, when the graph has more than
    1000 nodes. On smaller graphs it stops after ``GIRVAN_NEWMAN_MAX_ITER``
    splits.
    """
    return {
        "louvain": _partition_record(G, louvain_communities(G)),
        "leiden": _partition_record(G, leiden_communities(G)),
        "girvan_newman": _girvan_newman(G),
    }


def cross_reference(
    G: nx.Graph,
    communities: dict[str, Any],
    centralities: dict[str, Any],
    top_n: int = 10,
) -> list[Any]:
    """Print the community of each top hub, then say if those hubs share one community."""
    del G  # The rankings and partitions already describe the graph.
    seen: set[Any] = set()
    hubs: list[Any] = []
    for ordered in centralities["order"].values():
        for node in ordered[:top_n]:
            if node not in seen:
                seen.add(node)
                hubs.append(node)

    print("COMMUNITY ASSIGNMENT OF TOP HUBS:")
    print("Node | Louvain | Leiden | Girvan-Newman")
    for node in hubs:
        cells = [str(node)]
        for name in ("louvain", "leiden", "girvan_newman"):
            block = communities[name]
            if block.get("skipped"):
                cells.append("skipped")
            else:
                cells.append(str(block["assignment"].get(node, "")))
        print(" | ".join(cells))

    for name in ("louvain", "leiden", "girvan_newman"):
        block = communities[name]
        if block.get("skipped"):
            print(block["message"])
            continue
        # One id means the hubs sit together. Several ids means they bridge groups.
        ids = {block["assignment"][node] for node in hubs if node in block["assignment"]}
        label = name.replace("_", "-")
        if len(ids) <= 1:
            only = next(iter(ids), None)
            print(f"Top hubs are concentrated in one {label} community: {only}.")
        else:
            print(f"Top hubs span {len(ids)} {label} communities: {sorted(ids)}.")
    return hubs


def _graph_counts(G: nx.Graph, label: str) -> None:
    print(f"{label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def _predicted_pairs(G: nx.Graph) -> list[tuple[Any, Any]] | None:
    """Pairs to score. None means every missing pair, which only fits a small graph."""
    missing = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2 - G.number_of_edges()
    # A few million missing pairs can be listed. The 2012 user graph has about 1.5 billion.
    if missing <= 2_000_000:
        return None
    seen: set[tuple[Any, Any]] = set()
    pairs: list[tuple[Any, Any]] = []
    for node in G:
        neighbors = set(G[node])
        for neighbor in neighbors:
            for other in G[neighbor]:
                if other == node or other in neighbors:
                    continue
                # One undirected pair, stored in a stable order.
                pair = (node, other) if str(node) <= str(other) else (other, node)
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
    print(
        f"link candidates: {len(pairs)} pairs that share a neighbor "
        f"(not all {missing} missing pairs)"
    )
    return pairs


def _sorted_predictions(rows) -> list[tuple[Any, Any, float]]:
    return sorted(rows, key=lambda item: (-item[2], str(item[0]), str(item[1])))


def adamic_adar(G: nx.Graph) -> list[tuple[Any, Any, float]]:
    """Score missing edges by how rare the shared neighbors are."""
    _graph_counts(G, "adamic_adar before")
    pairs = _predicted_pairs(G)
    # NetworkX scores every missing pair when ebunch is omitted.
    scored = nx.adamic_adar_index(G, ebunch=pairs)
    ranked = _sorted_predictions(scored)
    _graph_counts(G, "adamic_adar after")
    return ranked


def jaccard(G: nx.Graph) -> list[tuple[Any, Any, float]]:
    """Score missing edges by the share of neighbors two accounts have in common."""
    _graph_counts(G, "jaccard before")
    pairs = _predicted_pairs(G)
    scored = nx.jaccard_coefficient(G, ebunch=pairs)
    ranked = _sorted_predictions(scored)
    _graph_counts(G, "jaccard after")
    return ranked


def preferential_attachment(G: nx.Graph) -> list[tuple[Any, Any, float]]:
    """Score missing edges by the product of the two degrees."""
    _graph_counts(G, "preferential_attachment before")
    pairs = _predicted_pairs(G)
    scored = nx.preferential_attachment(G, ebunch=pairs)
    ranked = _sorted_predictions(scored)
    _graph_counts(G, "preferential_attachment after")
    return ranked


def cpm_communities(G: nx.Graph, k: int = 3) -> dict[Any, list[int]]:
    """Clique percolation. A node can sit in more than one community."""
    _graph_counts(G, "cpm before")
    membership: dict[Any, list[int]] = {node: [] for node in G.nodes}
    # Each yielded set is one k-clique community. They are allowed to overlap.
    for index, community in enumerate(nx.community.k_clique_communities(G, k)):
        for node in community:
            membership[node].append(index)
    _graph_counts(G, "cpm after")
    return membership


def print_cpm_summary(membership: dict[Any, list[int]]) -> None:
    """Count nodes by how many communities they belong to, then list the most overlapped."""
    buckets = {1: 0, 2: 0, "3+": 0}
    for communities in membership.values():
        count = len(communities)
        if count == 1:
            buckets[1] += 1
        elif count == 2:
            buckets[2] += 1
        elif count >= 3:
            buckets["3+"] += 1
    print(f"nodes in 1 community: {buckets[1]}")
    print(f"nodes in 2 communities: {buckets[2]}")
    print(f"nodes in 3+ communities: {buckets['3+']}")
    ranked = sorted(membership, key=lambda node: (-len(membership[node]), str(node)))
    print("top 20 nodes by overlapping communities:")
    for node in ranked[:20]:
        print(f"{node} {len(membership[node])}")
