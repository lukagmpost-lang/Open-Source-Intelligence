"""Query a run saved in the SQLite store. One query, one table."""

from __future__ import annotations

import argparse
import operator
import re
import sys
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from osi.analysis import (
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    louvain_communities,
    pagerank,
)
from osi.store import get_run, list_runs, load_communities, load_graph, load_metrics

# Checked in this order. "top N in …" must be tried before "top N by …".
_METRIC = r"degree|pagerank|betweenness|closeness"
_PATTERNS = (
    ("top_in", re.compile(rf"^top (\d+) in (\d+) by ({_METRIC})$")),
    ("top", re.compile(rf"^top (\d+) by ({_METRIC})$")),
    ("neighbors", re.compile(r"^neighbors of (.+)$")),
    ("community", re.compile(r"^community of (.+)$")),
    ("sizes", re.compile(r"^communities by size$")),
    # "paths" is its own pattern. "path from" does not match it.
    ("paths", re.compile(r"^paths from (.+?) to (.+) limit (\d+)$")),
    ("path", re.compile(r"^path from (.+?) to (.+)$")),
    ("distance", re.compile(r"^distance between (.+?) and (.+)$")),
    ("neighborhood", re.compile(r"^neighborhood of (.+) radius (\d+)$")),
    # >= and <= are listed first so the shorter operators do not take the "=".
    ("where", re.compile(rf"^nodes where ({_METRIC}) (>=|<=|>|<) ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")),
)
_SUPPORTED = (
    "top N by METRIC",
    "top N in COMMUNITY by METRIC",
    "neighbors of NODE",
    "community of NODE",
    "communities by size",
    "nodes where METRIC OP VALUE",
    "path from NODE to NODE",
    "paths from NODE to NODE limit K",
    "distance between NODE and NODE",
    "neighborhood of NODE radius R",
)
_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


def _compute_metric(graph: nx.Graph, metric: str) -> dict:
    if metric == "degree":
        return degree_centrality(graph)
    if metric == "pagerank":
        return pagerank(graph)
    if metric == "betweenness":
        return betweenness_centrality(graph)
    return closeness_centrality(graph)


def _metric_scores(run_id: str, graph: nx.Graph, metric: str) -> dict:
    stored = load_metrics(run_id, metric)
    # Empty means this metric was never written. Compute it from the saved graph.
    return stored or _compute_metric(graph, metric)


def _communities(run_id: str, graph: nx.Graph) -> dict:
    saved = load_communities(run_id, "louvain")
    return saved or louvain_communities(graph)


def _print_table(headers: tuple[str, ...], rows: list[tuple]) -> None:
    text = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def emit(cells: list[str]) -> None:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip())

    emit(list(headers))
    for row in text:
        emit(row)


def _score_cell(value: float) -> str:
    return f"{value:.6f}"


def _edge_weight(graph: nx.Graph, left, right) -> float:
    data = graph.get_edge_data(left, right) or {}
    # Same default as the neighbor table: an edge with no stored weight counts as 1.
    return float(data.get("weight", 1.0))


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance. Kept local so a missing name does not import the report tool."""
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _closest_names(graph: nx.Graph, name: str, limit: int = 10) -> list[str]:
    # Full edit distance, not a cutoff of 2, so the list is the ten nearest names even when all are farther.
    ranked = sorted((_edit_distance(name, str(node)), str(node)) for node in graph.nodes if str(node) != name)
    return [other for _distance, other in ranked[:limit]]


def _print_missing(graph: nx.Graph, *names: str) -> bool:
    missing = [name for name in names if name not in graph]
    if not missing:
        return False
    for name in missing:
        print(name)
        print("similar names:")
        for other in _closest_names(graph, name):
            print(f"  {other}")
    return True


def _print_path(graph: nx.Graph, source: str, target: str) -> int:
    if _print_missing(graph, source, target):
        return 1
    try:
        # Weight is tie strength. Passing it as length would walk around a strong tie through weaker ones.
        nodes = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        print("no path (disconnected)")
        return 1
    cumulative = 0.0
    if len(nodes) == 1:
        print(f"{nodes[0]}  {_score_cell(cumulative)}")
        return 0
    for left, right in zip(nodes, nodes[1:]):
        cumulative += _edge_weight(graph, left, right)
        print(f"{left}  {right}  {_score_cell(cumulative)}")
    return 0


def _print_paths(graph: nx.Graph, source: str, target: str, limit: int) -> int:
    if _print_missing(graph, source, target):
        return 1
    try:
        # Yen's algorithm. Stop at K so a dense community does not enumerate every simple path.
        found = list(islice(nx.shortest_simple_paths(graph, source, target), limit))
    except nx.NetworkXNoPath:
        found = []
    if not found:
        print("no path (disconnected)")
        return 1
    for nodes in found:
        print(" ".join(str(node) for node in nodes))
    return 0


def _print_distance(graph: nx.Graph, source: str, target: str) -> int:
    if _print_missing(graph, source, target):
        return 1
    try:
        nodes = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        print("inf")
        return 1
    # Hop count. The same path the single-path query walks.
    print(len(nodes) - 1)
    return 0


def _print_neighborhood(graph: nx.Graph, node: str, radius: int) -> int:
    if _print_missing(graph, node):
        return 1
    if radius > 5:
        print("warning: radius above 5 may be slow", file=sys.stderr)
    lengths = nx.single_source_shortest_path_length(graph, node, cutoff=radius)
    # The source is included at distance 0, so the count is everyone the radius reaches.
    ordered = sorted(lengths.items(), key=lambda item: (item[1], str(item[0])))
    print(len(ordered))
    for other, dist in ordered:
        print(f"{other}  {dist}")
    return 0


def _run_query(kind: str, groups: re.Match[str], run_id: str, graph: nx.Graph) -> int:
    if kind == "top":
        count, metric = int(groups.group(1)), groups.group(2)
        rows = list(_metric_scores(run_id, graph, metric).items())[:count]
        _print_table(("rank", "node", metric), [(index, node, _score_cell(score)) for index, (node, score) in enumerate(rows, start=1)])
        return 0
    if kind == "top_in":
        count, community_id, metric = int(groups.group(1)), int(groups.group(2)), groups.group(3)
        membership = _communities(run_id, graph)
        # The stored scores are already strongest-first, so filtering keeps that order.
        chosen = [(node, score) for node, score in _metric_scores(run_id, graph, metric).items() if membership.get(node) == community_id]
        _print_table(("rank", "node", metric), [(index, node, _score_cell(score)) for index, (node, score) in enumerate(chosen[:count], start=1)])
        return 0
    if kind == "neighbors":
        node = groups.group(1)
        # A supra-graph stores "person|layer". The bare person id still selects those nodes.
        if node in graph:
            names = [node]
        else:
            names = sorted(other for other, data in graph.nodes(data=True) if data.get("person") == node)
        if not names:
            _print_missing(graph, node)
            return 1
        linked = []
        for name in names:
            linked.extend(graph.edges(name, data=True))
        # Stronger shared-thread ties first. Missing weight is the same default as the archive loader.
        linked = sorted(linked, key=lambda item: (-float(item[2].get("weight", 1.0)), str(item[1])))
        _print_table(("neighbor", "weight"), [(other, _score_cell(float(data.get("weight", 1.0)))) for _left, other, data in linked])
        return 0
    if kind == "community":
        node = groups.group(1)
        membership = _communities(run_id, graph)
        rows = [(node, membership[node])] if node in membership else []
        _print_table(("node", "community"), rows)
        return 0
    if kind == "sizes":
        counts: dict[int, int] = {}
        for community_id in _communities(run_id, graph).values():
            counts[int(community_id)] = counts.get(int(community_id), 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        _print_table(("community", "size"), ordered)
        return 0
    if kind == "path":
        return _print_path(graph, groups.group(1).strip(), groups.group(2).strip())
    if kind == "paths":
        return _print_paths(graph, groups.group(1).strip(), groups.group(2).strip(), int(groups.group(3)))
    if kind == "distance":
        return _print_distance(graph, groups.group(1).strip(), groups.group(2).strip())
    if kind == "neighborhood":
        return _print_neighborhood(graph, groups.group(1).strip(), int(groups.group(2)))
    metric, symbol, raw_value = groups.group(1), groups.group(2), float(groups.group(3))
    compare = _OPS[symbol]
    matched = [(node, score) for node, score in _metric_scores(run_id, graph, metric).items() if compare(score, raw_value)]
    _print_table(("node", metric), [(node, _score_cell(score)) for node, score in matched])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query a stored run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("query")
    args = parser.parse_args(argv)
    matched = None
    for kind, pattern in _PATTERNS:
        found = pattern.fullmatch(args.query.strip())
        if found:
            matched = (kind, found)
            break
    if matched is None:
        for line in _SUPPORTED:
            print(line)
        return 1
    meta = get_run(args.run)
    if meta is None:
        for run_id, source, _created_at, _notes in list_runs():
            print(f"{run_id} {source}")
        return 1
    layer = (meta.get("config") or {}).get("layer")
    graph = load_graph(args.run, layer) if layer else None
    if graph is None:
        for run_id, source, _created_at, _notes in list_runs():
            print(f"{run_id} {source}")
        return 1
    return _run_query(matched[0], matched[1], args.run, graph)


if __name__ == "__main__":
    raise SystemExit(main())
