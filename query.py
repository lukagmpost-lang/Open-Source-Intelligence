"""Query a run saved in the SQLite store. One query, one table."""

from __future__ import annotations

import argparse
import operator
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from osi.analysis import betweenness_centrality, degree_centrality, louvain_communities, pagerank
from osi.store import get_run, list_runs, load_communities, load_graph, load_metrics

# Checked in this order. "top N in …" must be tried before "top N by …".
_METRIC = r"degree|pagerank|betweenness|closeness"
_PATTERNS = (
    ("top_in", re.compile(rf"^top (\d+) in (\d+) by ({_METRIC})$")),
    ("top", re.compile(rf"^top (\d+) by ({_METRIC})$")),
    ("neighbors", re.compile(r"^neighbors of (.+)$")),
    ("community", re.compile(r"^community of (.+)$")),
    ("sizes", re.compile(r"^communities by size$")),
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
)
_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


def _compute_metric(graph: nx.Graph, metric: str) -> dict:
    if metric == "degree":
        return degree_centrality(graph)
    if metric == "pagerank":
        return pagerank(graph)
    if metric == "betweenness":
        return betweenness_centrality(graph)
    # compare_centralities uses unweighted closeness. Weight is path length, not tie strength.
    raw = nx.closeness_centrality(graph)
    return dict(sorted(raw.items(), key=lambda item: (-item[1], str(item[0]))))


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


def _run_query(kind: str, groups: re.Match[str], run_id: str, graph: nx.Graph) -> None:
    if kind == "top":
        count, metric = int(groups.group(1)), groups.group(2)
        rows = list(_metric_scores(run_id, graph, metric).items())[:count]
        _print_table(("rank", "node", metric), [(index, node, _score_cell(score)) for index, (node, score) in enumerate(rows, start=1)])
        return
    if kind == "top_in":
        count, community_id, metric = int(groups.group(1)), int(groups.group(2)), groups.group(3)
        membership = _communities(run_id, graph)
        # The stored scores are already strongest-first, so filtering keeps that order.
        chosen = [(node, score) for node, score in _metric_scores(run_id, graph, metric).items() if membership.get(node) == community_id]
        _print_table(("rank", "node", metric), [(index, node, _score_cell(score)) for index, (node, score) in enumerate(chosen[:count], start=1)])
        return
    if kind == "neighbors":
        node = groups.group(1)
        # Stronger shared-thread ties first. Missing weight is the same default as the archive loader.
        linked = sorted(graph.edges(node, data=True), key=lambda item: (-float(item[2].get("weight", 1.0)), str(item[1])))
        _print_table(("neighbor", "weight"), [(other, _score_cell(float(data.get("weight", 1.0)))) for _left, other, data in linked])
        return
    if kind == "community":
        node = groups.group(1)
        membership = _communities(run_id, graph)
        rows = [(node, membership[node])] if node in membership else []
        _print_table(("node", "community"), rows)
        return
    if kind == "sizes":
        counts: dict[int, int] = {}
        for community_id in _communities(run_id, graph).values():
            counts[int(community_id)] = counts.get(int(community_id), 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        _print_table(("community", "size"), ordered)
        return
    metric, symbol, raw_value = groups.group(1), groups.group(2), float(groups.group(3))
    compare = _OPS[symbol]
    matched = [(node, score) for node, score in _metric_scores(run_id, graph, metric).items() if compare(score, raw_value)]
    _print_table(("node", metric), [(node, _score_cell(score)) for node, score in matched])


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
    _run_query(matched[0], matched[1], args.run, graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
