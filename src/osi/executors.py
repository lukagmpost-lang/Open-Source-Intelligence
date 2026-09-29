"""Run one analysis question against a saved graph and return a ResultObject.

The score functions live in ``osi.analysis``. This module only loads the run,
times the call, and fills method, trust, caveats, and runtime.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import networkx as nx

from osi.analysis import (
    _modularity_of,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    degree_distribution,
    leiden_communities,
    louvain_communities,
    network_health as _network_health,
    pagerank,
    robustness,
)
from osi.result import ResultObject, make_caveats
from osi.store import get_run, load_communities, load_graph, load_metrics

# Exact betweenness above this size is the sampled NetworkX routine, k=500.
BETWEENNESS_SAMPLE_NODES = 5000
BETWEENNESS_SAMPLE_SIZE = 500
# Random deletion normally averages 20 trials. Larger graphs use 10.
CRITICALITY_SAMPLE_NODES = 10000
CRITICALITY_EXACT_TRIALS = 20
CRITICALITY_SAMPLED_TRIALS = 10

_METRICS: dict[str, Callable[[nx.Graph], dict]] = {
    "pagerank": pagerank,
    "degree": degree_centrality,
    "betweenness": betweenness_centrality,
    "closeness": closeness_centrality,
}
_COMMUNITIES = {"louvain": louvain_communities, "leiden": leiden_communities}


def _load_graph(run: str) -> nx.Graph:
    meta = get_run(run)
    if meta is None:
        raise LookupError(f"run {run} was not found")
    # The layer is stored with the run so the caller can pass only the run id.
    layer = (meta.get("config") or {}).get("layer")
    if not layer:
        raise LookupError(f"run {run} has no layer")
    graph = load_graph(run, layer)
    if graph is None:
        raise LookupError(f"run {run} has no graph for {layer}")
    return graph


def _sorted_scores(scores: dict) -> dict:
    return dict(sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0]))))


def _trust_for_method(method: str) -> str:
    # Exact numbers are the whole graph. A sample can move, so it stays moderate.
    if method == "exact":
        return "stable"
    return "moderate"


def _trust_from_modularity(modularity: float) -> str:
    if modularity >= 0.5:
        return "stable"
    if modularity >= 0.3:
        return "moderate"
    return "unstable"


def _finish(
    intent: str,
    params: dict,
    values: dict,
    method: str,
    sample_size: int | None,
    trust: str,
    graph_size: int,
    started: float,
) -> ResultObject:
    runtime_ms = int((time.perf_counter() - started) * 1000)
    result = ResultObject(
        intent=intent,
        params=params,
        values=values,
        method=method,
        sample_size=sample_size,
        trust=trust,
        caveats=[],
        runtime_ms=runtime_ms,
    )
    result.caveats = make_caveats(result, graph_size)
    return result


def _exact_metric(run: str, metric: str, graph: nx.Graph) -> dict:
    stored = load_metrics(run, metric)
    if stored:
        return stored
    return _METRICS[metric](graph)


def _metric_scores(run: str, metric: str, graph: nx.Graph) -> tuple[dict, str, int | None]:
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric}")
    # analysis.betweenness_centrality is exact. Above the cutoff, sample 500 sources instead.
    if metric == "betweenness" and graph.number_of_nodes() > BETWEENNESS_SAMPLE_NODES:
        sample_size = min(BETWEENNESS_SAMPLE_SIZE, graph.number_of_nodes())
        raw = nx.betweenness_centrality(graph, k=sample_size, weight="weight", seed=0)
        return _sorted_scores(raw), "sampled", sample_size
    return _exact_metric(run, metric, graph), "exact", None


def rank_nodes(run: str, metric: str = "pagerank", top: int = 10) -> ResultObject:
    """Top nodes by pagerank, degree, betweenness, or closeness."""
    graph = _load_graph(run)
    started = time.perf_counter()
    scores, method, sample_size = _metric_scores(run, metric, graph)
    values = dict(list(scores.items())[:top])
    return _finish(
        "rank_nodes",
        {"run": run, "metric": metric, "top": top},
        values,
        method,
        sample_size,
        _trust_for_method(method),
        graph.number_of_nodes(),
        started,
    )


def list_communities(run: str, algorithm: str = "louvain") -> ResultObject:
    """Named community summary. Trust follows modularity, not the exact/sampled rule."""
    if algorithm not in _COMMUNITIES:
        raise ValueError(f"unknown algorithm {algorithm}")
    graph = _load_graph(run)
    started = time.perf_counter()
    assignment = load_communities(run, algorithm)
    if not assignment:
        assignment = _COMMUNITIES[algorithm](graph)
    counts: dict[Any, int] = {}
    for community in assignment.values():
        counts[community] = counts.get(community, 0) + 1
    modularity = _modularity_of(graph, assignment)
    values = ResultObject.community_values(counts, modularity)
    return _finish(
        "list_communities",
        {"run": run, "algorithm": algorithm},
        values,
        "exact",
        None,
        _trust_from_modularity(modularity),
        graph.number_of_nodes(),
        started,
    )


def network_health(run: str) -> ResultObject:
    """Assortativity, clustering, transitivity, components, degree, and the power-law fit."""
    graph = _load_graph(run)
    started = time.perf_counter()
    health = _network_health(graph)
    fit = degree_distribution(graph)
    values = {
        "assortativity": health["assortativity"],
        "clustering": health["avg_clustering"],
        "transitivity": health["transitivity"],
        "components": health["num_components"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "power_law": fit["power_law"],
    }
    return _finish(
        "network_health",
        {"run": run},
        values,
        "exact",
        None,
        "stable",
        graph.number_of_nodes(),
        started,
    )


def structural_criticality(run: str) -> ResultObject:
    """Largest-component fraction after random, degree, and betweenness removal."""
    graph = _load_graph(run)
    started = time.perf_counter()
    if graph.number_of_nodes() > CRITICALITY_SAMPLE_NODES:
        trials = CRITICALITY_SAMPLED_TRIALS
        method = "sampled"
        sample_size: int | None = trials
    else:
        trials = CRITICALITY_EXACT_TRIALS
        method = "exact"
        sample_size = None
    degree_scores = load_metrics(run, "degree") or None
    between_scores = load_metrics(run, "betweenness") or None
    measured = robustness(
        graph,
        degree_scores=degree_scores,
        betweenness_scores=between_scores,
        runs=trials,
    )
    values = {
        strategy: {ratio: stats["largest"] for ratio, stats in measured[strategy].items()}
        for strategy in ("random", "degree", "betweenness")
    }
    return _finish(
        "structural_criticality",
        {"run": run, "runs": trials},
        values,
        method,
        sample_size,
        _trust_for_method(method),
        graph.number_of_nodes(),
        started,
    )


def _path_values(graph: nx.Graph, source: Any, target: Any) -> dict:
    if source not in graph or target not in graph:
        return {"connected": False}
    try:
        # Weight is tie strength, not distance, so the path is the fewest hops.
        path = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return {"connected": False}
    weight = 0.0
    for left, right in zip(path, path[1:]):
        weight += float(graph[left][right].get("weight", 1.0))
    return {"connected": True, "path": path, "distance": len(path) - 1, "weight": weight}


def connectivity(run: str, source: Any, target: Any) -> ResultObject:
    """Shortest path, hop distance, and the sum of edge weights along it."""
    graph = _load_graph(run)
    started = time.perf_counter()
    values = _path_values(graph, source, target)
    return _finish(
        "connectivity",
        {"run": run, "source": source, "target": target},
        values,
        "exact",
        None,
        "stable",
        graph.number_of_nodes(),
        started,
    )


def _local_structure(graph: nx.Graph, limit: int = 40) -> list[dict]:
    """Each node's neighbors. Large graphs keep the summary stats only."""
    if graph.number_of_nodes() > limit or graph.number_of_nodes() == 0:
        return []
    rows = []
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    for node, degree in ranked:
        neighbors = []
        for other in sorted(graph.neighbors(node), key=str):
            neighbors.append(
                {"node": other, "weight": float(graph[node][other].get("weight", 1.0))}
            )
        rows.append({"node": node, "degree": int(degree), "neighbors": neighbors})
    return rows


def discuss(run: str, question: str = "") -> ResultObject:
    """Facts a reader needs in order to explain the graph in plain language.

    The notes are the measured structure: size, clustering, communities,
    the highest PageRank scores, and, on a small graph, every adjacency.
    """
    graph = _load_graph(run)
    started = time.perf_counter()
    simple = graph.to_undirected() if graph.is_directed() else graph
    health = _network_health(simple)
    ranks = list(pagerank(simple).items())[:5]
    assignment = louvain_communities(simple)
    counts: dict[Any, int] = {}
    for community in assignment.values():
        counts[community] = counts.get(community, 0) + 1
    communities = ResultObject.community_values(counts, _modularity_of(simple, assignment))
    triangles = sum(nx.triangles(simple).values()) // 3
    density = float(nx.density(simple)) if simple.number_of_nodes() else 0.0
    values = {
        "nodes": simple.number_of_nodes(),
        "edges": simple.number_of_edges(),
        "components": health["num_components"],
        "density": round(density, 6),
        "triangles": triangles,
        "clustering": health["avg_clustering"],
        "assortativity": health["assortativity"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "communities": communities["n_communities"],
        "modularity": communities["modularity"],
        "largest_community": communities["largest_size"],
        "top_pagerank": {str(node): score for node, score in ranks},
        "structure": _local_structure(simple),
    }
    return _finish(
        "discuss",
        {"run": run, "question": question},
        values,
        "exact",
        None,
        "stable",
        simple.number_of_nodes(),
        started,
    )


def explain_node(run: str, node: Any) -> ResultObject:
    """Degree, three centralities, Louvain community, and the five heaviest neighbors."""
    graph = _load_graph(run)
    if node not in graph:
        raise LookupError(f"node {node} is not in run {run}")
    started = time.perf_counter()
    # This report is exact. The 5000-node betweenness sample applies only to rank_nodes.
    degree = _exact_metric(run, "degree", graph)
    ranks = _exact_metric(run, "pagerank", graph)
    between = _exact_metric(run, "betweenness", graph)
    close = _exact_metric(run, "closeness", graph)
    communities = load_communities(run, "louvain")
    if not communities:
        communities = louvain_communities(graph)
    neighbors = []
    for other in graph.neighbors(node):
        neighbors.append((other, float(graph[node][other].get("weight", 1.0))))
    neighbors.sort(key=lambda item: (-item[1], str(item[0])))
    values = {
        "degree": degree[node],
        "pagerank": ranks[node],
        "betweenness": between[node],
        "closeness": close[node],
        "community": communities[node],
        "neighbors": [{"node": name, "weight": weight} for name, weight in neighbors[:5]],
    }
    return _finish(
        "explain_node",
        {"run": run, "node": node},
        values,
        "exact",
        None,
        "stable",
        graph.number_of_nodes(),
        started,
    )
