"""Network measures used by the ask path.

Rank, communities, paths, and explain use NetworkX only. igraph, NumPy,
and SciPy are imported inside the functions that can go faster with them.
"""

from __future__ import annotations

import math
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


def _modularity_of(G: nx.Graph, assignment: dict[Any, int], weight: str = "weight") -> float:
    grouped: dict[int, set[Any]] = {}
    for node, community in assignment.items():
        grouped.setdefault(community, set()).add(node)
    groups = list(grouped.values())
    # An edgeless sample has modularity 0. The quality function would divide by zero.
    if not groups or G.number_of_edges() == 0:
        return 0.0
    return float(nx.community.modularity(G, groups, weight=weight))


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
    if remaining < 2 or remaining > 8000:
        # Harmonic centrality is an all-pairs walk. On a large attacked graph it
        # dominates the removal curve, and the fragility sentence uses the largest component.
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

    nodes = list(G.nodes())
    index = {node: position for position, node in enumerate(nodes)}
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

    measured = _measure_robustness(G, nodes, groups)
    by_label = {label: stats for label, stats in zip(labels, measured)}
    # 0% removal is the baseline. If the caller did not ask for it, measure the intact graph once.
    if any(ratio == 0.0 for _name, ratio in labels):
        baseline = by_label[(strategies[0], 0.0)]
    else:
        baseline = _measure_robustness(G, nodes, [[[]]])[0]
    results: dict[str, Any] = {"baseline": baseline}
    for name in strategies:
        # Stored in the caller's ratio order, which is what the table and giant_halved_at walk.
        results[name] = {ratio: by_label[(name, ratio)] for ratio in remove_ratio}
    return results


def _nx_component_stats(graph: nx.Graph, original_count: int) -> dict[str, float]:
    remaining = graph.number_of_nodes()
    if remaining == 0 or original_count == 0:
        return {"remaining": 0.0, "largest": 0.0, "components": 0.0, "efficiency": 0.0}
    components = list(nx.connected_components(graph))
    largest = max(len(component) for component in components) / original_count
    if remaining < 2 or remaining > 8000:
        efficiency = 0.0
    else:
        # Same quantity as igraph's summed harmonic centrality over ordered pairs.
        efficiency = float(nx.global_efficiency(graph))
    return {
        "remaining": float(remaining),
        "largest": float(largest),
        "components": float(len(components)),
        "efficiency": efficiency,
    }


def _measure_groups_networkx(
    graph: nx.Graph, nodes: list, groups: list[list[list[int]]]
) -> list[dict[str, float]]:
    base = graph.to_undirected() if graph.is_directed() else graph
    node_count = len(nodes)
    total = len(groups)
    _progress(0, total)
    cache: dict[frozenset[int], dict[str, float]] = {}
    averaged = []
    for done, group in enumerate(groups, start=1):
        rows = []
        for trial in group:
            key = frozenset(trial)
            if key not in cache:
                attacked = base.copy()
                if trial:
                    attacked.remove_nodes_from(nodes[position] for position in trial)
                cache[key] = _nx_component_stats(attacked, node_count)
            rows.append(cache[key])
        averaged.append(_average_stats(rows))
        _progress(done, total)
    return averaged


def _measure_robustness(
    graph: nx.Graph, nodes: list, groups: list[list[list[int]]]
) -> list[dict[str, float]]:
    """Use igraph when it is installed. NetworkX is the slower fallback."""
    try:
        import igraph as ig

        _names, base = _as_igraph(graph, None)
        if not isinstance(base, ig.Graph):
            raise RuntimeError("igraph did not build the robustness graph")
    except ImportError:
        return _measure_groups_networkx(graph, nodes, groups)
    if base.vcount() != len(nodes):
        raise RuntimeError("igraph node count does not match the NetworkX graph")
    return _measure_groups(base.get_edgelist(), len(nodes), groups)


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
    import numpy as np

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
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
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
