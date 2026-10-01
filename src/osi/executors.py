"""Run one analysis question against a saved graph and return a ResultObject.

The score functions live in ``osi.analysis``. This module only loads the run,
times the call, and fills method, trust, caveats, and runtime.
"""

from __future__ import annotations

import json
import os
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
from osi.findings import (
    COMMUNITY_METRICS,
    HEALTH_SOURCES,
    RANK_SOURCES,
    generate_findings,
    load_baselines,
    pagerank_centrality_text,
    pick_top_findings,
)
from osi.result import ResultObject, make_caveats
from osi.store import get_run, load_communities, load_graph, load_metrics
from osi.vocabulary import apply_vocabulary, get_source, vocabulary_for

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
    *,
    n_edges: int | None = None,
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
        n_nodes=graph_size,
        n_edges=n_edges,
    )
    result.caveats = make_caveats(result, graph_size)
    return result


def _exact_metric(run: str, metric: str, graph: nx.Graph) -> dict:
    stored = load_metrics(run, metric)
    if stored:
        return stored
    if metric == "betweenness":
        return betweenness_centrality(graph, run_id=run)
    return _METRICS[metric](graph)


def _metric_scores(run: str, metric: str, graph: nx.Graph) -> tuple[dict, str, int | None]:
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric}")
    # analysis.betweenness_centrality is exact. Above the cutoff, sample 500 sources instead.
    if metric == "betweenness" and graph.number_of_nodes() > BETWEENNESS_SAMPLE_NODES:
        stored = load_metrics(run, metric)
        if stored:
            return _sorted_scores(stored), "exact", None
        sample_size = min(BETWEENNESS_SAMPLE_SIZE, graph.number_of_nodes())
        raw = betweenness_centrality(graph, run_id=run, k=sample_size)
        return _sorted_scores(raw), "sampled", sample_size
    return _exact_metric(run, metric, graph), "exact", None


def interpret_health(metrics: dict, graph: nx.Graph) -> list[str]:
    """Convert raw metrics into 3-5 plain-language findings.

    Never mention metric names. Never quote raw values unless the value
    is itself interesting (for example a hub with far more connections
    than average). At most three findings are returned, in the order a
    reader would care about them: hubs, then how tightly people are
    connected, then groups, then fragmentation, then who connects to whom.
    """
    ranked: list[tuple[int, str]] = []
    average = float(metrics.get("avg_degree") or 0)
    if average > 10:
        ranked.append(
            (
                5,
                "This is a dense network — most accounts are connected to dozens of others, "
                "which means a message can cross the group in a few introductions.",
            )
        )
    elif average < 3:
        ranked.append(
            (
                5,
                "This is a sparse network — most accounts have only a few connections, "
                "which means most people only ever hear from a couple of others.",
            )
        )

    max_degree = int(metrics.get("max_degree") or 0)
    if average and max_degree > 100 * average:
        ranked.append(
            (
                6,
                "A few hubs dominate — the top node has "
                f"{_shown_number(max_degree)} connections vs an average of {_shown_number(average)}, "
                "which means a handful of accounts do almost all the connecting.",
            )
        )

    modularity = metrics.get("modularity")
    if modularity is None and graph.number_of_nodes() > 1 and graph.number_of_edges() > 0:
        assignment = louvain_communities(graph)
        modularity = _modularity_of(graph, assignment)
    if isinstance(modularity, (int, float)) and not isinstance(modularity, bool):
        if float(modularity) > 0.5:
            ranked.append(
                (4, "This network splits cleanly into groups, which means each crowd mostly talks to itself.")
            )
        elif float(modularity) < 0.4:
            ranked.append(
                (
                    4,
                    "The groups are blurry — they overlap heavily, "
                    "which means the same people show up in several crowds at once.",
                )
            )

    components = metrics.get("components", metrics.get("num_components"))
    nodes = graph.number_of_nodes()
    if nodes <= 0:
        raw_nodes = metrics.get("nodes", metrics.get("n_nodes"))
        if isinstance(raw_nodes, (int, float)) and not isinstance(raw_nodes, bool):
            nodes = int(raw_nodes)
    if (
        nodes
        and isinstance(components, (int, float))
        and not isinstance(components, bool)
        and int(components) > nodes * 0.01
    ):
        ranked.append(
            (
                3,
                "It's fragmented — most nodes are in small disconnected pieces, "
                "which means most accounts never see each other.",
            )
        )

    assortativity = metrics.get("assortativity")
    if isinstance(assortativity, (int, float)) and not isinstance(assortativity, bool):
        if float(assortativity) < -0.5:
            ranked.append(
                (
                    2,
                    "Hubs connect to leaves, not to each other. "
                    "This is a broadcast network, not a community, "
                    "which means if those hubs leave, the audience has no way to reach each other.",
                )
            )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _priority, sentence in ranked[:3]]


def _shown_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def _name_list(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else "nobody"
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def interpret_rank(
    ranked: list[tuple], graph: nx.Graph, metric: str, vocab: dict[str, str] | None = None
) -> list[str]:
    """Turn a ranked list into 3 findings."""
    if not ranked or graph.number_of_nodes() == 0:
        return []
    names = [str(node) for node, _score in ranked[:5]]
    top_name = ranked[0][0]
    top_degree = int(graph.degree(top_name))
    average = sum(degree for _node, degree in graph.degree()) / graph.number_of_nodes()
    findings = [
        f"The most central accounts are {_name_list(names)}, "
        "which means these are the people everyone else has to go through."
    ]
    if average and top_degree > 5 * average:
        ratio = top_degree / average
        multiple = int(ratio // 10) * 10 if ratio >= 10 else int(ratio)
        shown_degree = f"{top_degree:,}"
        findings.append(
            f"{top_name} is connected to {shown_degree} other accounts — more than {multiple}x the average. "
            "That's like knowing everyone in a small town, which means one person carries "
            "connections the rest of the network does not."
        )
    if len(ranked) >= 5:
        findings.append(
            "These five are the hubs of the network, "
            "which means a handful of accounts do almost all the connecting."
        )
    if metric == "pagerank":
        central = pagerank_centrality_text(ranked)
        if central:
            findings.append(central)
    if vocab is None:
        return findings
    return [apply_vocabulary(sentence, vocab) for sentence in findings]


def interpret_communities(n_communities: int, largest_size: int, modularity: float | None) -> list[str]:
    """Turn a community summary into plain findings."""
    findings = [
        f"This network splits into about {int(n_communities)} groups, "
        "which means people mostly stay inside their own crowd."
    ]
    findings.append(
        f"The largest group contains {int(largest_size):,} accounts, "
        "which means that crowd sets the tone for everyone else."
    )
    if isinstance(modularity, (int, float)) and not isinstance(modularity, bool):
        if float(modularity) < 0.4:
            findings.append(
                "The groups overlap heavily, which means the same people show up in several crowds at once."
            )
        elif float(modularity) > 0.5:
            findings.append("The groups separate cleanly, which means each crowd mostly talks to itself.")
    return findings


def _shape_findings(
    metrics: dict,
    communities: dict,
    ranked: list[tuple],
    graph: nx.Graph,
    vocab: dict[str, str] | None = None,
) -> list[str]:
    """The few sentences that describe the shape, with no metric names."""
    lines: list[str] = []
    for sentence in interpret_health(metrics, graph):
        if sentence.startswith("This is a dense") or sentence.startswith("This is a sparse"):
            lines.append(sentence)
            break
    for sentence in interpret_communities(
        int(communities.get("n_communities") or 0),
        int(communities.get("largest_size") or 0),
        communities.get("modularity"),
    ):
        if "modularity" in sentence.lower():
            lines.append(
                "The groups overlap heavily, which means the same people show up in several crowds at once."
            )
        else:
            lines.append(sentence)
    rank_lines = interpret_rank(ranked, graph, "pagerank")
    hub = next((sentence for sentence in rank_lines if "connected to" in sentence), None)
    if hub:
        lines.append(hub)
    elif rank_lines:
        lines.append(rank_lines[0])
    hubs = next((sentence for sentence in rank_lines if sentence.startswith("These five")), None)
    if hubs and len(lines) < 5:
        lines.append(hubs)
    selected_vocab = vocab or vocabulary_for("general")
    return [apply_vocabulary(line, selected_vocab) for line in lines[:5]]


def _selected_findings(
    metrics: dict,
    allowed: set[str] | None = None,
    n: int = 4,
    sources: set[str] | None = None,
    vocab: dict[str, str] | None = None,
) -> list[str]:
    """Plain sentences from the findings engine, highest score first."""
    found = generate_findings(metrics, None, vocab)
    if allowed is not None:
        found = [item for item in found if item["metric"] in allowed]
    if sources is not None:
        found = [item for item in found if item.get("source") in sources]
    return pick_top_findings(found, n)


def rank_nodes(run: str, metric: str = "pagerank", top: int = 10) -> ResultObject:
    """Top nodes by pagerank, degree, betweenness, or closeness."""
    graph = _load_graph(run)
    vocab = vocabulary_for(get_source(run))
    started = time.perf_counter()
    scores, method, sample_size = _metric_scores(run, metric, graph)
    ranked = list(scores.items())
    values = dict(ranked[:top])
    names = [node for node, _score in ranked[:5]]
    node_count = graph.number_of_nodes()
    average = sum(degree for _node, degree in graph.degree()) / node_count if node_count else 0.0
    degrees = [int(graph.degree(node)) for node in names]
    assignment = load_communities(run, "louvain")
    hub_metrics = {
        "nodes": node_count,
        "avg_degree": average,
        "max_degree": max((degree for _node, degree in graph.degree()), default=0),
        "top_node": str(names[0]) if names else None,
        "top_accounts": [str(node) for node in names],
        "top_5_degree": min(degrees) if len(degrees) >= 5 else None,
        "top_communities": (
            [assignment.get(node) for node in names] if assignment and len(names) >= 5 else None
        ),
    }
    base = interpret_rank(ranked[:top], graph, metric, vocab)
    extra = [
        text
        for text in _selected_findings(hub_metrics, sources=RANK_SOURCES, vocab=vocab)
        if text not in base
    ]
    values["findings"] = base + extra
    return _finish(
        "rank_nodes",
        {"run": run, "metric": metric, "top": top},
        values,
        method,
        sample_size,
        _trust_for_method(method),
        graph.number_of_nodes(),
        started,
        n_edges=graph.number_of_edges(),
    )


def list_communities(run: str, algorithm: str = "louvain") -> ResultObject:
    """Named community summary. Trust follows modularity, not the exact/sampled rule."""
    if algorithm not in _COMMUNITIES:
        raise ValueError(f"unknown algorithm {algorithm}")
    graph = _load_graph(run)
    vocab = vocabulary_for(get_source(run))
    started = time.perf_counter()
    assignment = load_communities(run, algorithm)
    if not assignment:
        assignment = _COMMUNITIES[algorithm](graph)
    counts: dict[Any, int] = {}
    for community in assignment.values():
        counts[community] = counts.get(community, 0) + 1
    modularity = _modularity_of(graph, assignment)
    values = ResultObject.community_values(counts, modularity)
    community_metrics = {
        "nodes": graph.number_of_nodes(),
        "modularity": modularity,
        "n_communities": int(values["n_communities"]),
        "largest_size": int(values["largest_size"]),
    }
    values["findings"] = _selected_findings(community_metrics, COMMUNITY_METRICS, vocab=vocab) or [
        apply_vocabulary(sentence, vocab)
        for sentence in interpret_communities(
        int(values["n_communities"]),
        int(values["largest_size"]),
        values.get("modularity"),
        )
    ]
    return _finish(
        "list_communities",
        {"run": run, "algorithm": algorithm},
        values,
        "exact",
        None,
        _trust_from_modularity(modularity),
        graph.number_of_nodes(),
        started,
        n_edges=graph.number_of_edges(),
    )


def network_health(run: str) -> ResultObject:
    """Assortativity, clustering, transitivity, components, degree, and the power-law fit."""
    graph = _load_graph(run)
    vocab = vocabulary_for(get_source(run))
    started = time.perf_counter()
    health = _network_health(graph)
    fit = degree_distribution(graph)
    assignment = load_communities(run, "louvain")
    if not assignment and graph.number_of_nodes() > 0:
        assignment = louvain_communities(graph)
    if assignment and graph.number_of_edges() > 0:
        modularity = _modularity_of(graph, assignment)
        group_count = len(set(assignment.values()))
    else:
        modularity = None
        group_count = 0
    metrics = {
        "assortativity": health["assortativity"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "components": health["num_components"],
        "modularity": modularity,
        "n_communities": group_count,
    }
    ranked_degree = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    top = ranked_degree[:5]
    largest_size = None
    top_communities = None
    if assignment:
        counts: dict[Any, int] = {}
        for community in assignment.values():
            counts[community] = counts.get(community, 0) + 1
        if counts:
            largest_size = max(counts.values())
        if len(top) >= 5:
            top_communities = [assignment.get(node) for node, _degree in top]
    power = fit.get("power_law") or {}
    finding_metrics = {
        "nodes": graph.number_of_nodes(),
        "assortativity": health["assortativity"],
        "avg_clustering": health["avg_clustering"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "components": health["num_components"],
        "modularity": modularity,
        "n_communities": group_count,
        "largest_size": largest_size,
        "power_law_r_squared": power.get("r_squared"),
        "top_node": str(top[0][0]) if top else None,
        "top_accounts": [str(node) for node, _degree in top],
        "top_5_degree": int(top[4][1]) if len(top) >= 5 else None,
        "top_communities": top_communities,
    }
    values = {
        "assortativity": health["assortativity"],
        "clustering": health["avg_clustering"],
        "transitivity": health["transitivity"],
        "components": health["num_components"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "power_law": fit["power_law"],
        "findings": _selected_findings(finding_metrics, sources=HEALTH_SOURCES, vocab=vocab)
        or [
            apply_vocabulary(sentence, vocab)
            for sentence in interpret_health(metrics, graph)
            if not sentence.startswith("A few hubs")
        ],
    }
    _remember_reference(
        run,
        {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "modularity": None if modularity is None else float(modularity),
            "avg_degree": float(health["avg_degree"]),
            "max_degree": int(health["max_degree"]),
            "components": int(health["num_components"]),
            "clustering": float(health["avg_clustering"]),
            "assortativity": health["assortativity"],
        },
    )
    return _finish(
        "network_health",
        {"run": run},
        values,
        "exact",
        None,
        "stable",
        graph.number_of_nodes(),
        started,
        n_edges=graph.number_of_edges(),
    )


def structural_criticality(run: str) -> ResultObject:
    """Largest-component fraction after random, degree, and betweenness removal."""
    from osi.store import get_metric, put_metric

    cached = get_metric(run, "structural_criticality")
    if cached is not None:
        return ResultObject.from_dict(json.loads(cached))
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
        run_id=run,
    )
    values = {
        strategy: {ratio: stats["largest"] for ratio, stats in measured[strategy].items()}
        for strategy in ("random", "degree", "betweenness")
    }
    values.update(_criticality_summary(measured, vocabulary_for(get_source(run))))
    result = _finish(
        "structural_criticality",
        {"run": run, "runs": trials},
        values,
        method,
        sample_size,
        _trust_for_method(method),
        graph.number_of_nodes(),
        started,
        n_edges=graph.number_of_edges(),
    )
    put_metric(run, "structural_criticality", json.dumps(result.to_dict()))
    return ResultObject.from_dict(json.loads(get_metric(run, "structural_criticality")))


def _curve_value(curve: dict, target: float) -> float | None:
    for key, value in curve.items():
        try:
            ratio = float(key)
            largest = float(value)
        except (TypeError, ValueError):
            continue
        if abs(ratio - target) < 1e-9:
            return largest
    return None


def _halving_ratio(curve: dict) -> float | None:
    """First removal ratio where the largest piece falls below half."""
    ratios = []
    for key, value in curve.items():
        try:
            ratio = float(key)
            largest = float(value)
        except (TypeError, ValueError):
            continue
        ratios.append((ratio, largest))
    for ratio, largest in sorted(ratios):
        if ratio > 0 and largest < 0.5:
            return ratio
    return None


def criticality_account_findings(values: dict, vocab: dict[str, str] | None = None) -> list[str]:
    """Who would disconnect the network. These sentences belong to critical_nodes."""
    rows = []
    for key, value in values.items():
        if key == "findings":
            continue
        try:
            rows.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    if not rows:
        findings = [
            "Removing the accounts that sit on the most paths first would disconnect the network, "
            "which means other people lose their way to each other."
        ]
        selected_vocab = vocab or vocabulary_for("general")
        return [apply_vocabulary(sentence, selected_vocab) for sentence in findings]
    rows.sort(key=lambda item: (-item[1], item[0]))
    names = _name_list([name for name, _score in rows[:5]])
    top_name, top_score = rows[0]
    midpoint = sorted(score for _name, score in rows)[len(rows) // 2]
    lines = [
        f"Removing {names} first would disconnect the network, "
        "which means these are the accounts other people have to pass through."
    ]
    if midpoint and top_score > 2 * midpoint:
        ratio = top_score / midpoint
        lines.append(
            f"{top_name} is on {ratio:.0f}x more paths than a typical account, "
            f"which means removing {top_name} cuts other people off from each other."
        )
    selected_vocab = vocab or vocabulary_for("general")
    return [apply_vocabulary(sentence, selected_vocab) for sentence in lines]


def _criticality_findings(
    degree: dict,
    betweenness: dict,
    random: dict,
    n_components: float | None,
) -> list[str]:
    """Plain sentences for how fast each removal strategy breaks the network."""
    halving_degree = _halving_ratio(degree)
    halving_random = _halving_ratio(random)
    frac_at_30 = _curve_value(degree, 0.30)
    findings: list[str] = []
    if halving_degree is None or halving_degree > 0.30:
        findings.append(
            "The network is resilient: it survives removing 30% of the top accounts without halving, "
            "which means people would still find each other if the busiest accounts left."
        )
    elif halving_degree <= 0.05:
        findings.append(
            "The network is extremely fragile: removing just "
            f"{halving_degree * 100:.0f}% of the top accounts by degree halves it, "
            "which means if those accounts leave, most people lose their connection to each other."
        )
    elif halving_degree <= 0.20:
        findings.append(
            "The network is fragile: removing "
            f"{halving_degree * 100:.0f}% of the top accounts halves it, "
            "which means if the top accounts leave, most people lose their connection to each other."
        )
    else:
        findings.append(
            "The network is moderately fragile: removing "
            f"{halving_degree * 100:.0f}% halves it, "
            "which means losing the busiest accounts would cut most people off from each other."
        )
    degree_breaks_sooner = halving_degree is not None and (
        halving_random is None or halving_random > halving_degree * 3
    )
    if degree_breaks_sooner:
        findings.append(
            "Removing random accounts has much less effect — this is a targeted-vulnerability pattern, "
            "which means the risk sits in a few accounts rather than in the crowd."
        )
    if frac_at_30 is not None and frac_at_30 < 0.10 and n_components is not None:
        pieces = int(round(n_components))
        findings.append(
            f"Removing 30% of the top accounts shatters the network into {pieces} disconnected pieces, "
            "which means most people would lose their connection to each other."
        )
    return findings


def _criticality_summary(measured: dict, vocab: dict[str, str] | None = None) -> dict:
    degree = {ratio: stats["largest"] for ratio, stats in measured["degree"].items()}
    betweenness = {ratio: stats["largest"] for ratio, stats in measured["betweenness"].items()}
    random = {ratio: stats["largest"] for ratio, stats in measured["random"].items()}
    components = None
    for ratio, stats in measured["degree"].items():
        if abs(float(ratio) - 0.30) < 1e-9:
            components = float(stats["components"])
            break
    findings = _criticality_findings(degree, betweenness, random, components)
    if vocab is not None:
        findings = [apply_vocabulary(sentence, vocab) for sentence in findings]
    summary = {
        "halving_degree": _halving_ratio(degree),
        "halving_betweenness": _halving_ratio(betweenness),
        "halving_random": _halving_ratio(random),
        "frac_at_30_degree": _curve_value(degree, 0.30),
        "findings": findings,
    }
    return summary


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
        n_edges=graph.number_of_edges(),
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
    """Plain findings about the shape: density, groups, and the main hub.

    The model receives those findings, not the raw metric dict.
    """
    graph = _load_graph(run)
    started = time.perf_counter()
    simple = graph.to_undirected() if graph.is_directed() else graph
    health = _network_health(simple)
    scores = load_metrics(run, "pagerank")
    if not scores:
        scores = _sorted_scores(pagerank(simple))
    ranked = list(scores.items())
    assignment = load_communities(run, "louvain")
    if not assignment:
        assignment = louvain_communities(simple)
    counts: dict[Any, int] = {}
    for community in assignment.values():
        counts[community] = counts.get(community, 0) + 1
    communities = ResultObject.community_values(counts, _modularity_of(simple, assignment))
    metrics = {
        "assortativity": health["assortativity"],
        "avg_degree": health["avg_degree"],
        "max_degree": health["max_degree"],
        "components": health["num_components"],
        "modularity": communities["modularity"],
        "n_communities": communities["n_communities"],
    }
    values = {"findings": _shape_findings(metrics, communities, ranked, simple)}
    return _finish(
        "discuss",
        {"run": run, "question": question},
        values,
        "exact",
        None,
        "stable",
        simple.number_of_nodes(),
        started,
        n_edges=simple.number_of_edges(),
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
        n_edges=graph.number_of_edges(),
    )


# Filled by network_health so a later comparison does not measure the graph again.
_REFERENCE_CACHE: dict[tuple[str, str], dict] = {}
_BASELINE_KEYS = (
    "nodes",
    "edges",
    "modularity",
    "avg_degree",
    "max_degree",
    "components",
    "clustering",
    "assortativity",
)
_BASELINE_ALIASES = {
    "snap": "snap_facebook",
    "snap_facebook": "snap_facebook",
    "facebook": "snap_facebook",
    "reddit_2008": "reddit_2008",
    "reddit2008": "reddit_2008",
    "2008": "reddit_2008",
    "reddit_2012": "reddit_2012",
    "reddit2012": "reddit_2012",
    "2012": "reddit_2012",
    "github_follows": "github_follows",
    "github": "github_follows",
    "bluesky_follows": "bluesky_follows",
    "bluesky": "bluesky_follows",
    "github_co_contribution": "github_co_contribution",
    "co_contribution": "github_co_contribution",
    "forum": "reddit_2008",
    "normal_forum": "reddit_2008",
}


def _cache_key(run: str) -> tuple[str, str]:
    return (os.environ.get("OSI_STORE", ""), run)


def _remember_reference(run: str, metrics: dict) -> None:
    _REFERENCE_CACHE[_cache_key(run)] = dict(metrics)


def _reference_metrics(run: str) -> dict:
    """Metrics lined up with baselines.json. Reuse the health pass when it already ran."""
    cached = _REFERENCE_CACHE.get(_cache_key(run))
    if cached:
        return cached
    network_health(run)
    return _REFERENCE_CACHE[_cache_key(run)]


def _baseline_key(name: str) -> str:
    token = (name or "snap_facebook").strip().lower().replace(" ", "_").replace("-", "_")
    key = _BASELINE_ALIASES.get(token)
    if key is None:
        known = ", ".join(sorted(set(_BASELINE_ALIASES.values())))
        raise ValueError(f"unknown baseline {name}. Choose one of: {known}")
    return key


def _median_rows(rows: list[dict]) -> dict[str, float]:
    typical: dict[str, float] = {}
    for key in _BASELINE_KEYS:
        values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
        if not values:
            continue
        middle = len(values) // 2
        if len(values) % 2:
            typical[key] = values[middle]
        else:
            typical[key] = (values[middle - 1] + values[middle]) / 2
    return typical


def _fmt_number(value) -> str:
    number = float(value)
    if abs(number) >= 100 or abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    if abs(number) >= 10:
        text = f"{number:.1f}"
    else:
        text = f"{number:.2f}"
    return text.rstrip("0").rstrip(".")


def _fmt_ratio(ratio: float) -> str:
    if abs(ratio - round(ratio)) < 0.05 or ratio >= 10:
        return f"{int(round(ratio))}x"
    return f"{ratio:.1f}x"


_BASELINE_TITLES = {
    "snap_facebook": "SNAP Facebook",
    "reddit_2008": "Reddit 2008",
    "reddit_2012": "Reddit 2012",
    "github_follows": "GitHub follows",
    "bluesky_follows": "Bluesky follows",
    "github_co_contribution": "GitHub co-contribution",
}


def _fold(ratio: float) -> float:
    """How many times larger the bigger side is. 0.1 and 10 are both a 10x gap."""
    magnitude = abs(float(ratio))
    if magnitude == 0:
        return float("inf")
    return magnitude if magnitude >= 1 else 1 / magnitude


def _ratio_to(current, reference) -> float | None:
    if current is None or reference is None:
        return None
    reference = float(reference)
    if reference == 0:
        return None
    return float(current) / reference


def _baseline_finding(metric: str, current, reference, ratio: float, title: str) -> str:
    """Say what a large gap means for the people in the network."""
    current_text = _fmt_number(current)
    reference_text = _fmt_number(reference)
    if ratio is None or ratio <= 0:
        return (
            f"This network does not mix people the way {title} does "
            f"({current_text} vs {reference_text}), which means there is no inner circle "
            "holding the rest of the accounts together."
        )
    bigger = ratio >= 1
    times = _fmt_ratio(ratio if bigger else 1 / ratio)
    sentences = {
        ("nodes", True): (
            f"There are {times} more accounts here than on {title} ({current_text} vs {reference_text}), "
            "which means this is no longer a room where everyone can know each other."
        ),
        ("nodes", False): (
            f"There are {times} fewer accounts here than on {title} ({current_text} vs {reference_text}), "
            "which means it is small enough that people can still recognize each other."
        ),
        ("edges", True): (
            f"People have formed {times} more links than on {title} ({current_text} vs {reference_text}), "
            "which means the crowd is tied together much more tightly than usual."
        ),
        ("edges", False): (
            f"People have formed {times} fewer links than on {title} ({current_text} vs {reference_text}), "
            "which means most accounts barely know anyone here."
        ),
        ("modularity", True): (
            f"The groups pull apart {times} more sharply than on {title}, "
            "which means people mostly stay inside their own neighborhood."
        ),
        ("modularity", False): (
            f"The groups blur together {times} more than on {title}, "
            "which means the same people show up in several crowds."
        ),
        ("avg_degree", True): (
            f"The average account knows about {current_text} people, {times} more than on {title}, "
            "which means introductions travel much faster than usual."
        ),
        ("avg_degree", False): (
            f"The average account knows about {current_text} people, {times} fewer than on {title}, "
            "which means most people only ever hear from a couple of others."
        ),
        ("max_degree", True): (
            f"One account is connected to {current_text} others — {times} more than on {title}. "
            "That means one person knows more people than most people will ever meet."
        ),
        ("max_degree", False): (
            f"The biggest account reaches {current_text} people, {times} fewer than on {title}, "
            "which means no single person knows the whole room."
        ),
        ("components", True): (
            f"The network is shattered into {current_text} disconnected fragments — "
            f"{times} more pieces than {title} — which means most accounts never see each other."
        ),
        ("components", False): (
            f"This network hangs together in {current_text} piece, {times} fewer fragments than {title}, "
            "which means almost everyone can still reach everyone else."
        ),
        ("clustering", True): (
            f"Friends of friends already know each other {times} more often than on {title}, "
            "which means this feels like a tight neighborhood."
        ),
        ("clustering", False): (
            f"Friends of friends rarely know each other — {times} less often than on {title} — "
            "which means people hear one voice and do not talk among themselves."
        ),
        ("assortativity", True): (
            f"Well-connected accounts stick together {times} more than on {title}, "
            "which means an inner circle hears the news before everyone else."
        ),
        ("assortativity", False): (
            f"Well-connected accounts do not stick together the way they do on {title}, "
            f"{times} less, which means there is no inner circle holding the rest of the network."
        ),
    }
    sentence = sentences.get((metric, bigger))
    if sentence:
        return sentence
    direction = "more" if bigger else "less"
    return (
        f"This network is {times} {direction} than {title} ({current_text} vs {reference_text}), "
        "which means the two networks do not feel like the same kind of place."
    )


def baseline_compare(run: str, baseline: str = "snap_facebook") -> ResultObject:
    """Compare this graph's metrics against one stored baseline.

    A finding is kept when the metric differs by more than 2x. The four
    largest gaps are returned, biggest first.
    """
    key = _baseline_key(baseline)
    started = time.perf_counter()
    current = _reference_metrics(run)
    reference = dict(load_baselines()[key])
    title = _BASELINE_TITLES.get(key, key)
    ratios: dict[str, float] = {}
    gaps: list[tuple[float, str]] = []
    for metric in _BASELINE_KEYS:
        ratio = _ratio_to(current.get(metric), reference.get(metric))
        if ratio is None:
            continue
        ratios[metric] = ratio
        if _fold(ratio) > 2:
            gaps.append((_fold(ratio), metric))
    gaps.sort(key=lambda item: (-item[0], item[1]))
    findings = [
        _baseline_finding(metric, current.get(metric), reference.get(metric), ratios[metric], title)
        for _fold_value, metric in gaps[:4]
    ]
    if not findings:
        findings.append(
            f"This network stays within twice of {title} on every comparison, "
            "which means nothing here would surprise someone who already knows that network."
        )
    values = {
        "this_graph": {metric: current.get(metric) for metric in _BASELINE_KEYS},
        "baseline_name": key,
        "baseline_metrics": {metric: reference.get(metric) for metric in _BASELINE_KEYS},
        "ratios": ratios,
        "findings": findings,
    }
    return _finish(
        "baseline_compare",
        {"run": run, "baseline": key},
        values,
        "exact",
        None,
        "stable",
        int(current["nodes"]),
        started,
        n_edges=int(current["edges"]),
    )


def anomaly_scan(run: str) -> ResultObject:
    """Compare every metric with the median baseline and return the three largest gaps."""
    started = time.perf_counter()
    current = _reference_metrics(run)
    typical = _median_rows(list(load_baselines().values()))
    ranked: list[tuple[float, str, float | None]] = []
    for metric in _BASELINE_KEYS:
        ratio = _ratio_to(current.get(metric), typical.get(metric))
        if ratio is None:
            if current.get(metric) is None or typical.get(metric) is None:
                continue
            distance = abs(float(current[metric]) - float(typical[metric]))
        else:
            distance = _fold(ratio)
        ranked.append((distance, metric, ratio))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    anomalies = []
    for _distance, metric, ratio in ranked[:3]:
        sentence = _baseline_finding(
            metric,
            current.get(metric),
            typical.get(metric),
            ratio if ratio is not None else 1.0,
            "a typical network",
        )
        anomalies.append(
            {
                "metric": metric,
                "this": current.get(metric),
                "typical": typical.get(metric),
                "ratio": ratio,
                "finding": sentence,
            }
        )
    values = {
        "anomalies": anomalies,
        "findings": [item["finding"] for item in anomalies],
    }
    return _finish(
        "anomaly_scan",
        {"run": run},
        values,
        "exact",
        None,
        "stable",
        int(current["nodes"]),
        started,
        n_edges=int(current["edges"]),
    )
