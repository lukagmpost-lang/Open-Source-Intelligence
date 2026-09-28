"""Inspect one user across saved runs.

Usage:
  python3 report.py --user USERNAME [--runs RUN1,RUN2,...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from osi.analysis import leiden_communities
from osi.store import get_run, list_runs, load_communities, load_graph, load_metrics
from query import _communities, _metric_scores, _score_cell

# Printed in this order. PageRank is the rank the cross-run bands use.
METRICS = ("pagerank", "degree", "betweenness", "closeness")
LABELS = {
    "pagerank": "PageRank:",
    "degree": "Degree:",
    "betweenness": "Betweenness:",
    "closeness": "Closeness:",
}
# "Betweenness:" is the longest label. Padding to it lines the values up.
_LABEL_WIDTH = len(LABELS["betweenness"])
# Names farther than this are not "similar" for the missing-user list.
_NAME_LIMIT = 2
_NEIGHBORS = 10


@dataclass
class _Present:
    run_id: str
    source: str
    created_at: str
    scores: dict[str, dict]
    louvain: dict
    leiden: dict


def levenshtein(left: str, right: str, limit: int = _NAME_LIMIT) -> int | None:
    """Return the edit distance, or None when it is already past limit."""
    if abs(len(left) - len(right)) > limit:
        return None
    # Distance is symmetric, so the shorter string is the row we store.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        # Every completed alignment passes through this row, so its cheapest cell is a lower bound.
        row_best = current[0]
        for column, right_char in enumerate(right, start=1):
            cost = min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            )
            current.append(cost)
            if cost < row_best:
                row_best = cost
        if row_best > limit:
            return None
        previous = current
    distance = previous[-1]
    if distance > limit:
        return None
    return distance


def _field(label: str, text: str) -> str:
    return f"  {label.ljust(_LABEL_WIDTH)}  {text}"


def _placement(scores: dict, user: str) -> tuple[int, int, float] | None:
    if user not in scores:
        return None
    # load_metrics and the analysis helpers already sort by score, then node name.
    for rank, (node, value) in enumerate(scores.items(), start=1):
        if node == user:
            return rank, len(scores), float(value)
    return None


def _in_band(rank: int, total: int, fraction: float) -> bool:
    # Same percentile ratio as compare.py: position divided by the number of scored nodes.
    return total > 0 and (rank / total) <= fraction


def _bands(placement: tuple[int, int, float] | None) -> str:
    if placement is None:
        return "top 100: n/a  top 1%: n/a  top 10%: n/a"
    rank, total, _value = placement
    top_100 = "yes" if rank <= 100 else "no"
    top_1 = "yes" if _in_band(rank, total, 0.01) else "no"
    top_10 = "yes" if _in_band(rank, total, 0.10) else "no"
    return f"top 100: {top_100}  top 1%: {top_1}  top 10%: {top_10}"


def _load_run_graph(run_id: str) -> nx.Graph | None:
    meta = get_run(run_id)
    if not meta:
        return None
    # main.py stores the layer name in the run config. query.py reads it the same way.
    layer = (meta.get("config") or {}).get("layer")
    if not layer:
        return None
    return load_graph(run_id, layer)


def _stored(run_id: str) -> tuple[dict[str, dict], dict, dict]:
    scores = {metric: load_metrics(run_id, metric) for metric in METRICS}
    return scores, load_communities(run_id, "louvain"), load_communities(run_id, "leiden")


def _user_in(scores: dict[str, dict], louvain: dict, leiden: dict, user: str) -> bool:
    if any(user in table for table in scores.values()):
        return True
    return user in louvain or user in leiden


def _indexed(scores: dict[str, dict], louvain: dict, leiden: dict) -> bool:
    # A saved metric or community table is the node set for that run.
    return any(scores.values()) or bool(louvain) or bool(leiden)


def _fill_gaps(run_id: str, scores: dict[str, dict], louvain: dict, leiden: dict) -> tuple[dict[str, dict], dict, dict]:
    missing_metric = any(not scores[metric] for metric in METRICS)
    # Computing Leiden or betweenness is the expensive path, so skip it when the store is complete.
    if not missing_metric and louvain and leiden:
        return scores, louvain, leiden
    graph = _load_run_graph(run_id)
    if graph is None:
        return scores, louvain, leiden
    for metric in METRICS:
        if not scores[metric]:
            scores[metric] = _metric_scores(run_id, graph, metric)
    if not louvain:
        louvain = _communities(run_id, graph)
    if not leiden:
        leiden = leiden_communities(graph)
    return scores, louvain, leiden


def _select_runs(spec: str | None) -> list[tuple[str, str, str, str]] | None:
    saved = list_runs()
    if not spec:
        return saved
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    known = {run_id for run_id, _source, _created_at, _notes in saved}
    # A typo should not quietly drop a run from the comparison.
    if not wanted or any(name not in known for name in wanted):
        for run_id, source, _created_at, _notes in saved:
            print(f"{run_id} {source}")
        return None
    chosen = set(wanted)
    # list_runs is oldest first. Consecutive deltas follow that clock, not the comma order.
    return [row for row in saved if row[0] in chosen]


def _present_runs(selected: list[tuple[str, str, str, str]], user: str) -> list[_Present]:
    found: list[_Present] = []
    for run_id, source, created_at, _notes in selected:
        scores, louvain, leiden = _stored(run_id)
        if not _user_in(scores, louvain, leiden, user):
            # The graph is only opened when this run never saved a node index.
            if _indexed(scores, louvain, leiden):
                continue
            graph = _load_run_graph(run_id)
            if graph is None or user not in graph:
                continue
        scores, louvain, leiden = _fill_gaps(run_id, scores, louvain, leiden)
        found.append(_Present(run_id, source, created_at, scores, louvain, leiden))
    return found


def _candidate_names(selected: list[tuple[str, str, str, str]]) -> set[str]:
    names: set[str] = set()
    for run_id, _source, _created_at, _notes in selected:
        scores, louvain, leiden = _stored(run_id)
        added = False
        for table in scores.values():
            if table:
                # One complete metric lists every scored node. The others repeat that set.
                names.update(str(node) for node in table)
                added = True
                break
        if added:
            continue
        membership = louvain or leiden
        if membership:
            names.update(str(node) for node in membership)
            continue
        graph = _load_run_graph(run_id)
        if graph is not None:
            names.update(str(node) for node in graph.nodes)
    return names


def _print_similar(names: set[str], user: str) -> None:
    ranked = []
    for name in names:
        if name == user:
            continue
        distance = levenshtein(user, name)
        if distance is None:
            continue
        ranked.append((distance, name))
    # Closer edits first. Equal distances fall back to the name so the ten names are stable.
    ranked.sort()
    print("similar names:")
    for _distance, name in ranked[:10]:
        print(f"  {name}")


def _print_run(row: _Present, user: str) -> None:
    print(f"{row.run_id} ({row.source}, {row.created_at})")
    for metric in METRICS:
        placement = _placement(row.scores[metric], user)
        if placement is None:
            print(_field(LABELS[metric], "n/a"))
            continue
        rank, total, value = placement
        # The spec asks for the population size on PageRank only.
        if metric == "pagerank":
            detail = f"{_score_cell(value)}  (rank {rank} / {total})"
        else:
            detail = f"{_score_cell(value)}  (rank {rank})"
        print(_field(LABELS[metric], detail))
    louvain_id = row.louvain[user] if user in row.louvain else "n/a"
    leiden_id = row.leiden[user] if user in row.leiden else "n/a"
    print(_field("Louvain:", str(louvain_id)))
    print(_field("Leiden:", str(leiden_id)))


def _print_ranks(rows: list[_Present], user: str) -> None:
    print("Cross-run")
    for row in rows:
        bits = []
        for metric in METRICS:
            placement = _placement(row.scores[metric], user)
            if placement is None:
                token = "n/a"
            elif metric == "pagerank":
                token = f"{placement[0]}/{placement[1]}"
            else:
                token = str(placement[0])
            bits.append(f"{metric} {token}")
        # Bands follow PageRank. The per-run section is the one that prints that rank against the total.
        pagerank_place = _placement(row.scores["pagerank"], user)
        print(f"  {row.run_id}  " + "  ".join(bits) + f"  {_bands(pagerank_place)}")
    if len(rows) < 2:
        return
    print()
    for earlier, later in zip(rows, rows[1:]):
        for metric in METRICS:
            left = _placement(earlier.scores[metric], user)
            right = _placement(later.scores[metric], user)
            label = LABELS[metric]
            if left is None or right is None:
                print(f"  {label}  {earlier.run_id} -> {later.run_id}  n/a")
                continue
            # Later minus earlier, so a drop in the metric is negative.
            delta = right[2] - left[2]
            print(
                f"  {label}  {earlier.run_id} -> {later.run_id}  "
                f"{_score_cell(left[2])} -> {_score_cell(right[2])} (delta {delta:+.6f})"
            )


def _print_neighbors(row: _Present, user: str) -> None:
    print()
    print(f"Neighbors in {row.run_id}")
    graph = _load_run_graph(row.run_id)
    if graph is None or user not in graph:
        print("  graph not stored")
        return
    # Same order as query.py: stronger shared-thread ties first, then the neighbor name.
    linked = sorted(graph.edges(user, data=True), key=lambda item: (-float(item[2].get("weight", 1.0)), str(item[1])))
    if not linked:
        print("  none")
        return
    for _left, other, data in linked[:_NEIGHBORS]:
        weight = float(data.get("weight", 1.0))
        louvain_id = row.louvain[other] if other in row.louvain else "n/a"
        leiden_id = row.leiden[other] if other in row.leiden else "n/a"
        print(f"  {other}  {_score_cell(weight)}  louvain {louvain_id}  leiden {leiden_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect one user across saved runs.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--runs", help="Comma-separated run ids. Default: every saved run.")
    args = parser.parse_args(argv)
    if not args.user.strip():
        parser.error("--user is required")
    selected = _select_runs(args.runs)
    if selected is None:
        return 1
    present = _present_runs(selected, args.user)
    if not present:
        _print_similar(_candidate_names(selected), args.user)
        return 1
    for index, row in enumerate(present):
        if index:
            print()
        _print_run(row, args.user)
    print()
    _print_ranks(present, args.user)
    # The list is oldest first, so the last row is the newest save that contains the user.
    _print_neighbors(present[-1], args.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
