"""Compare two stored runs on one metric, or on community overlap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.store import get_run, list_runs, load_communities, load_metrics

# A match has to clear this before it counts as split, merge, or survival.
_MATCH = 0.3
# One-to-one overlap has to clear this before the community is called stable.
_STABLE = 0.7


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


def _score(value: float) -> str:
    return f"{value:.6f}"


def _require_run(run_id: str) -> bool:
    if get_run(run_id) is not None:
        return True
    for saved_id, source, _created_at, _notes in list_runs():
        print(f"{saved_id} {source}")
    return False


def _metric_tables(run_a: str, run_b: str, metric: str, top_n: int) -> int:
    left = load_metrics(run_a, metric)
    right = load_metrics(run_b, metric)
    # An empty load means the metric was never stored. Do not recompute it here.
    if not left or not right:
        missing = run_a if not left else run_b
        print(f"{missing} has no {metric}")
        return 1
    both = set(left) & set(right)
    # Equal scores are neither risers nor fallers.
    risers = sorted(((node, right[node] - left[node]) for node in both if right[node] > left[node]), key=lambda item: (-item[1], str(item[0])))
    fallers = sorted(((node, right[node] - left[node]) for node in both if right[node] < left[node]), key=lambda item: (item[1], str(item[0])))
    # New and gone are ranked by the score in the snapshot that contains them.
    new_nodes = sorted((set(right) - set(left)), key=lambda node: (-right[node], str(node)))
    gone_nodes = sorted((set(left) - set(right)), key=lambda node: (-left[node], str(node)))

    print("RISERS")
    _print_table(("node", "a", "b", "delta"), [(node, _score(left[node]), _score(right[node]), _score(delta)) for node, delta in risers[:top_n]])
    print("FALLERS")
    _print_table(("node", "a", "b", "delta"), [(node, _score(left[node]), _score(right[node]), _score(delta)) for node, delta in fallers[:top_n]])
    print("NEW")
    _print_table(("node", "b"), [(node, _score(right[node])) for node in new_nodes[:top_n]])
    print("GONE")
    _print_table(("node", "a"), [(node, _score(left[node])) for node in gone_nodes[:top_n]])
    return 0


def _groups(membership: dict) -> dict[int, set]:
    groups: dict[int, set] = {}
    for node, community_id in membership.items():
        groups.setdefault(int(community_id), set()).add(node)
    return groups


def _jaccard(intersection: int, size_a: int, size_b: int) -> float:
    union = size_a + size_b - intersection
    # A community with no members has no overlap with anything.
    return intersection / union if union else 0.0


def _classify(best: float, strong: list[int], absorbed: set[int]) -> str:
    # Split wins over merge: a community that breaks apart is not one stable piece.
    if not strong:
        return "DISSOLVED"
    if len(strong) >= 2:
        return "SPLIT"
    if strong[0] in absorbed:
        return "MERGED"
    if best > _STABLE:
        return "STABLE"
    # One 2012 community overlaps this one by more than 0.3, but not by more than 0.7.
    return "below-stable"


def _community_table(run_a: str, run_b: str, algorithm: str) -> int:
    left = load_communities(run_a, algorithm)
    right = load_communities(run_b, algorithm)
    if not left or not right:
        missing = run_a if not left else run_b
        print(f"{missing} has no {algorithm}")
        return 1
    groups_a = _groups(left)
    groups_b = _groups(right)
    # Node to its B community, so each A member is counted without scanning every B community.
    home_b = {node: int(community_id) for node, community_id in right.items()}
    overlaps: dict[int, list[tuple[int, float]]] = {}
    for community_id, members in groups_a.items():
        counts: dict[int, int] = {}
        for node in members:
            other = home_b.get(node)
            # Absent from B: the node stays in the union through size_a and adds no intersection.
            if other is None:
                continue
            counts[other] = counts.get(other, 0) + 1
        scored = [
            (other, _jaccard(count, len(members), len(groups_b[other])))
            for other, count in counts.items()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        overlaps[community_id] = scored

    # A B community absorbs two A communities when each overlaps it by more than the match cutoff.
    absorbed: set[int] = set()
    partners: dict[int, list[int]] = {}
    for community_id, scored in overlaps.items():
        for other, score in scored:
            if score > _MATCH:
                partners.setdefault(other, []).append(community_id)
    for other, sources in partners.items():
        if len(sources) >= 2:
            absorbed.add(other)

    rows = []
    for community_id, members in sorted(groups_a.items(), key=lambda item: (-len(item[1]), item[0])):
        scored = overlaps[community_id]
        best_id, best = scored[0] if scored else ("", 0.0)
        strong = [other for other, score in scored if score > _MATCH]
        label = _classify(best, strong, absorbed)
        rows.append((community_id, len(members), best_id, _score(best), label))
    _print_table(("a_community", "size", "b_community", "overlap", "classification"), rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two stored runs.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--metric")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--algorithm", choices=("louvain", "leiden"))
    parser.add_argument("--mode", choices=("communities",))
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    metric_mode = args.metric is not None and args.mode is None
    community_mode = args.mode == "communities"
    if metric_mode == community_mode:
        parser.error("pass --metric, or --mode communities with --algorithm")
    if community_mode and not args.algorithm:
        parser.error("--algorithm is required with --mode communities")
    if not _require_run(args.a) or not _require_run(args.b):
        return 1
    if community_mode:
        return _community_table(args.a, args.b, args.algorithm)
    return _metric_tables(args.a, args.b, args.metric, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
