"""Network measures on a public social graph.

The score functions do not print. ``cross_reference`` is the exception:
it prints which community the top hubs fall in.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities as _louvain_communities
from scipy import stats


def _by_score(scores: dict[Any, float]) -> dict[Any, float]:
    return dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))


def degree_centrality(G: nx.Graph) -> dict[Any, float]:
    return _by_score(nx.degree_centrality(G))


def pagerank(G: nx.Graph, weight: str = "weight") -> dict[Any, float]:
    return _by_score(nx.pagerank(G, weight=weight))


def _undirected_projection(G: nx.Graph) -> nx.Graph:
    """Return an undirected graph for algorithms that reject a DiGraph.

    An undirected graph is returned as itself. Reciprocal gift edges add their weights.
    """
    if not G.is_directed():
        return G
    plain = nx.Graph()
    plain.add_nodes_from(G.nodes(data=True))
    for left, right, data in G.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if plain.has_edge(left, right):
            plain[left][right]["weight"] = float(plain[left][right].get("weight", 1.0)) + weight
            continue
        copied = dict(data)
        copied["weight"] = weight
        plain.add_edge(left, right, **copied)
    return plain


def _as_igraph(G: nx.Graph, weight: str | None):
    """Copy an undirected NetworkX graph into igraph, optionally with edge weights."""
    import igraph as ig

    # igraph is built as a simple undirected graph, so a DiGraph is collapsed first.
    G = _undirected_projection(G)
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
    # Louvain is defined on undirected graphs. Directed gift edges are collapsed first.
    groups = _louvain_communities(_undirected_projection(G), weight=weight, seed=0)
    return _community_index([set(group) for group in groups])


def leiden_communities(G: nx.Graph, weight: str = "weight") -> dict[Any, int]:
    G = _undirected_projection(G)
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
    # Modularity in NetworkX rejects a directed graph. The partition was built on this projection.
    G = _undirected_projection(G)
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
    G = _undirected_projection(G)
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


# Below this size, starting a process pool costs more than the removals themselves.
_ROBUSTNESS_PARALLEL_NODES = 2500
_WORKER_GRAPH = None
_WORKER_ORIGINAL = 0


def _component_stats(graph, original_count: int) -> dict[str, float]:
    remaining = graph.vcount()
    if remaining == 0 or original_count == 0:
        return {"remaining": 0, "largest": 0.0, "components": 0.0, "efficiency": 0.0}
    sizes = graph.connected_components().sizes()
    # Fraction of the original node set, so a removed node counts as lost structure.
    largest = max(sizes) / original_count
    if remaining < 2:
        efficiency = 0.0
    else:
        # Sum of inverse distances, divided by ordered pairs. Same value as networkx.global_efficiency.
        harmonic = graph.harmonic_centrality(normalized=False)
        efficiency = float(sum(harmonic)) / (remaining * (remaining - 1))
    return {
        "remaining": float(remaining),
        "largest": float(largest),
        "components": float(len(sizes)),
        "efficiency": float(efficiency),
    }


def _stats_after_removal(graph, remove_idx: list[int], original_count: int) -> dict[str, float]:
    attacked = graph.copy()
    if remove_idx:
        # One call keeps the indexes on the intact graph. Deleting one by one would shift them.
        attacked.delete_vertices(list(remove_idx))
    return _component_stats(attacked, original_count)


def _init_robustness_worker(edges: list[tuple[int, int]], node_count: int) -> None:
    global _WORKER_GRAPH, _WORKER_ORIGINAL
    import igraph as ig

    _WORKER_ORIGINAL = node_count
    _WORKER_GRAPH = ig.Graph(n=node_count, edges=edges, directed=False)


def _robustness_worker(remove_idx: list[int]) -> dict[str, float]:
    return _stats_after_removal(_WORKER_GRAPH, remove_idx, _WORKER_ORIGINAL)


def _removal_count(node_count: int, ratio: float) -> int:
    # Integer part of the fraction, capped so a ratio of 1 clears the graph and a tiny graph can remove nothing.
    return max(0, min(node_count, int(ratio * node_count)))


def _attack_order(nodes, index: dict, scores: dict | None, degree_of) -> list[int]:
    """Highest score first, on the intact graph. Ties break by node name."""
    if scores:
        # A node missing from a stored table is ranked last, not dropped from the attack.
        ranked = sorted(nodes, key=lambda node: (-float(scores.get(node, 0.0)), str(node)))
    else:
        ranked = sorted(nodes, key=lambda node: (-degree_of(node), str(node)))
    return [index[node] for node in ranked]


def _average_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    count = len(rows)
    averaged = {
        "remaining": rows[0]["remaining"],
        "largest": sum(row["largest"] for row in rows) / count,
        "components": sum(row["components"] for row in rows) / count,
        "efficiency": sum(row["efficiency"] for row in rows) / count,
    }
    return averaged


# Intact graph plus the six attack sizes. One progress step per strategy and ratio.
_DEFAULT_RATIOS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]


def robustness(
    G: nx.Graph,
    strategies: list[str] | None = None,
    remove_ratio: list[float] | None = None,
    runs: int = 10,
    degree_scores: dict | None = None,
    betweenness_scores: dict | None = None,
) -> dict[str, Any]:
    """Remove a fixed prefix of the intact-graph ranking and measure what remains.

    Degree and betweenness are scored once, before any deletion. Later ratios
    delete a longer prefix of that same list. Random draws a fresh set each
    trial and averages `runs` trials. Nothing in the ranking is recomputed
    on the damaged graph.
    """
    if strategies is None:
        strategies = ["random", "degree", "betweenness"]
    else:
        strategies = list(strategies)
    if remove_ratio is None:
        remove_ratio = list(_DEFAULT_RATIOS)
    else:
        remove_ratio = list(remove_ratio)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    unknown = [name for name in strategies if name not in {"random", "degree", "betweenness"}]
    if unknown:
        raise ValueError(f"unknown strategy: {unknown[0]}")

    node_count = G.number_of_nodes()
    empty = {"remaining": 0.0, "largest": 0.0, "components": 0.0, "efficiency": 0.0}
    if node_count == 0:
        return {"baseline": empty, **{name: {ratio: dict(empty) for ratio in remove_ratio} for name in strategies}}

    names, base = _as_igraph(G, None)
    index = {node: position for position, node in enumerate(names)}
    edges = base.get_edgelist()
    nodes = list(names)
    # One ranking of the intact graph. Stored scores are that same ranking when the caller already has them.
    degree_order = _attack_order(nodes, index, degree_scores, G.degree)
    between_order: list[int] = []
    if "betweenness" in strategies:
        if betweenness_scores:
            between_order = _attack_order(nodes, index, betweenness_scores, G.degree)
        else:
            between_order = [index[node] for node in betweenness_centrality(G)]

    import random

    # Fixed seed so the random average does not change between calls.
    rng = random.Random(0)
    # Larger attacks first. Those graphs are smaller, so the progress counter moves before the intact graph is scored.
    ratios_by_cost = sorted(remove_ratio, key=lambda ratio: (-_removal_count(node_count, ratio), ratio))
    groups: list[list[list[int]]] = []
    labels: list[tuple[str, float]] = []
    # Targeted attacks before random trials, so the first finished step is one deletion list, not ten samples.
    strategy_order = [name for name in ("degree", "betweenness", "random") if name in strategies]
    for ratio in ratios_by_cost:
        count = _removal_count(node_count, ratio)
        for name in strategy_order:
            labels.append((name, ratio))
            if count == 0:
                # Every strategy at 0% is the intact graph. One measurement is enough.
                groups.append([[]])
            elif name == "degree":
                groups.append([degree_order[:count]])
            elif name == "betweenness":
                groups.append([between_order[:count]])
            else:
                groups.append([rng.sample(range(node_count), count) for _trial in range(runs)])

    measured = _measure_groups(edges, node_count, groups)
    by_label = {label: stats for label, stats in zip(labels, measured)}
    # 0% removal is the baseline. If the caller did not ask for it, measure the intact graph once.
    if any(ratio == 0.0 for _name, ratio in labels):
        baseline = by_label[(strategies[0], 0.0)]
    else:
        baseline = _measure_groups(edges, node_count, [[[]]])[0]
    results: dict[str, Any] = {"baseline": baseline}
    for name in strategies:
        # Stored in the caller's ratio order, which is what the table and giant_halved_at walk.
        results[name] = {ratio: by_label[(name, ratio)] for ratio in remove_ratio}
    return results


def _progress(done: int, total: int) -> None:
    import sys

    print(f"robustness {done}/{total}", file=sys.stderr, flush=True)


def _measure_groups(edges: list[tuple[int, int]], node_count: int, groups: list[list[list[int]]]) -> list[dict[str, float]]:
    """Average each group. The stderr counter is one step per group, not one step per random trial."""
    total = len(groups)
    _progress(0, total)
    keys_per_group = [[frozenset(trial) for trial in group] for group in groups]
    unique: dict[frozenset[int], list[int]] = {}
    for group, keys in zip(groups, keys_per_group):
        for trial, key in zip(group, keys):
            unique.setdefault(key, trial)

    # Process startup dominates on small graphs. Large graphs spend the time in all-pairs distances.
    if node_count < _ROBUSTNESS_PARALLEL_NODES or len(unique) <= 1:
        import igraph as ig

        base = ig.Graph(n=node_count, edges=edges, directed=False)
        cache: dict[frozenset[int], dict[str, float]] = {}
        averaged = []
        done = 0
        for keys, group in zip(keys_per_group, groups):
            rows = []
            for trial, key in zip(group, keys):
                if key not in cache:
                    cache[key] = _stats_after_removal(base, trial, node_count)
                rows.append(cache[key])
            averaged.append(_average_stats(rows))
            done += 1
            _progress(done, total)
        return averaged

    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    # Two copies of the 2012 graph fit in memory. More workers start swapping and the distances slow down.
    workers = min(2, os_cpu_count(), len(unique))
    # Fork, not spawn: spawn re-imports the parent script, which fails for python -c and for stdin.
    context = mp.get_context("fork")
    pending_groups = [set(keys) for keys in keys_per_group]
    rows_by_key: dict[frozenset[int], dict[str, float]] = {}
    averaged: list[dict[str, float] | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_robustness_worker,
        initargs=(edges, node_count),
    ) as pool:
        future_to_key = {pool.submit(_robustness_worker, trial): key for key, trial in unique.items()}
        pending = set(future_to_key)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                key = future_to_key[future]
                rows_by_key[key] = future.result()
                for index, needed in enumerate(pending_groups):
                    if key not in needed or averaged[index] is not None:
                        continue
                    needed.remove(key)
                    if needed:
                        continue
                    group_rows = [rows_by_key[group_key] for group_key in keys_per_group[index]]
                    averaged[index] = _average_stats(group_rows)
                    done += 1
                    _progress(done, total)
    if any(row is None for row in averaged):
        raise RuntimeError("a robustness step finished without a measurement")
    return [row for row in averaged if row is not None]


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def giant_halved_at(results: dict[str, Any]) -> dict[str, float | None]:
    """Smallest simulated ratio whose largest component is at most half the intact one."""
    half = results["baseline"]["largest"] / 2
    found: dict[str, float | None] = {}
    for name, rows in results.items():
        if name == "baseline":
            continue
        found[name] = None
        for ratio, stats in rows.items():
            if stats["largest"] <= half:
                found[name] = ratio
                break
    return found


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


def _finite(value: float) -> float | None:
    # Assortativity is NaN when every node in the component has the same degree.
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def network_health(G: nx.Graph) -> dict[str, Any]:
    """Return a dict with:
    - assortativity: nx.degree_assortativity_coefficient(G)
    - avg_clustering: nx.average_clustering(G)
    - transitivity: nx.transitivity(G)
    - num_components: len(list(nx.connected_components(G)))
    - avg_degree: mean degree
    - max_degree: max degree
    """
    # Component counts and clustering below use undirected neighborhoods.
    G = _undirected_projection(G)
    degrees = [degree for _node, degree in G.degree()]
    node_count = len(degrees)
    if node_count == 0:
        avg_degree = 0.0
        max_degree = 0
    else:
        avg_degree = float(sum(degrees)) / node_count
        max_degree = int(max(degrees))
    components = list(nx.connected_components(G))
    # Same count as len(list(nx.connected_components(G))). The list is reused below.
    num_components = len(components)
    if node_count == 0:
        # average_clustering rejects an empty graph. The other scalars are already defined.
        return {
            "assortativity": None,
            "avg_clustering": 0.0,
            "transitivity": 0.0,
            "num_components": 0,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
        }
    # Disconnected graphs mix several degree patterns. The coefficient is the largest piece only.
    largest = max(components, key=len)
    if len(largest) < 2:
        assortativity = None
    else:
        try:
            assortativity = _finite(float(nx.degree_assortativity_coefficient(G.subgraph(largest))))
        except nx.NetworkXError:
            assortativity = None
    return {
        "assortativity": assortativity,
        "avg_clustering": float(nx.average_clustering(G)),
        "transitivity": float(nx.transitivity(G)),
        "num_components": num_components,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
    }


def _r_squared(observed, predicted) -> float | None:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = float(np.sum((observed - observed.mean()) ** 2))
    # A flat series has no variance, so the ratio is undefined.
    if total == 0.0:
        return None
    residual = float(np.sum((observed - predicted) ** 2))
    return _finite(1.0 - residual / total)


def _blank_fit() -> dict[str, Any]:
    return {"params": {}, "r_squared": None}


def degree_distribution(G: nx.Graph) -> dict[str, Any]:
    """Fit power-law, lognormal, and exponential to the degree distribution.

    Return dict with best_fit, params, r_squared for each.
    """
    # Isolates have degree 0. log(k) is undefined there, so every fit uses k >= 1.
    positive = [degree for _node, degree in G.degree() if degree > 0]
    names = ("power_law", "lognormal", "exponential")
    blank = {"best_fit": None, **{name: _blank_fit() for name in names}}
    # Two occupied bins are the minimum for a slope.
    if len(positive) < 2:
        return blank
    values, counts = np.unique(positive, return_counts=True)
    values = values.astype(float)
    counts = counts.astype(float)
    if len(values) < 2:
        return blank
    # Each distinct degree is one point. A hub and a leaf weigh the same on the log-log line.
    log_counts = np.log(counts)
    log_degrees = np.log(values)
    slope, intercept = (float(item) for item in np.polyfit(log_degrees, log_counts, 1))
    power_law = {
        # alpha is the positive exponent in k^{-alpha}. xmin is the smallest degree in the fit.
        "params": {"alpha": -slope, "xmin": float(values.min())},
        "r_squared": _r_squared(log_counts, intercept + slope * log_degrees),
    }
    exp_slope, exp_intercept = (float(item) for item in np.polyfit(values, log_counts, 1))
    exponential = {
        # rate is the positive decay in exp(-rate * k) when the semi-log line falls.
        "params": {"rate": -exp_slope},
        "r_squared": _r_squared(log_counts, exp_intercept + exp_slope * values),
    }
    lognormal = _blank_fit()
    try:
        # floc=0 keeps the support on positive degrees. sigma is the shape; mu is log(scale).
        sigma, _loc, scale = stats.lognorm.fit(np.asarray(positive, dtype=float), floc=0)
        if scale > 0 and sigma > 0:
            dist = stats.lognorm(sigma, loc=0, scale=scale)
            # Integer degrees sit in [k - 0.5, k + 0.5) so the cdf difference matches the histogram.
            expected = len(positive) * (dist.cdf(values + 0.5) - dist.cdf(values - 0.5))
            # A bin the model gives no mass still has to be a finite log. Clip only those zeros.
            expected = np.clip(expected, 1e-12, None)
            lognormal = {
                "params": {"sigma": float(sigma), "mu": float(np.log(scale))},
                "r_squared": _r_squared(log_counts, np.log(expected)),
            }
    except (ValueError, RuntimeError, FloatingPointError):
        lognormal = _blank_fit()
    fits = {"power_law": power_law, "lognormal": lognormal, "exponential": exponential}
    best_fit = None
    best_score = None
    for name in names:
        score = fits[name]["r_squared"]
        if score is None:
            continue
        # Strictly greater, so a tie stays with the earlier name: power_law, then lognormal.
        if best_score is None or score > best_score:
            best_fit = name
            best_score = score
    return {"best_fit": best_fit, **fits}


def rich_club(G: nx.Graph, k_values: list[int] | None = None) -> dict[int, float | None]:
    """Return dict {k: rich_club_coefficient} for each k."""
    G = _undirected_projection(G)
    if k_values is None:
        k_values = [10, 20, 50, 100]
    # normalized=True rewires every edge Q times. That is a null model, not the coefficient.
    try:
        coefficients = nx.rich_club_coefficient(G, normalized=False)
    except Exception:
        # NetworkX raises Exception, not NetworkXError, when the graph has a self-loop.
        return {k: None for k in k_values}
    found: dict[int, float | None] = {}
    for k in k_values:
        # Degrees past the last node with a neighbor are absent from the coefficient dict.
        if k not in coefficients:
            found[k] = None
            continue
        found[k] = _finite(float(coefficients[k]))
    return found
