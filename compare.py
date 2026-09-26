"""Compare two stored runs on one metric, or on community containment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.store import get_run, list_runs, load_communities, load_metrics

# Lift above the uniform null. Two destinations past 10 is a split; one past 50 is stable.
_LIFT_SPLIT = 10
_LIFT_STABLE = 50


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


def _containment(intersection: int, size_a: int) -> float:
    # Denominator is the earlier community, so a large later community is not punished.
    return intersection / size_a if size_a else 0.0


def _lift(containment: float, n_later: int) -> float:
    # Uniform null: members spread evenly, about one per later community.
    # Expected containment is 1 / n_later (about 0.001 for the 2012 partition).
    if n_later <= 0:
        return 0.0
    return containment * n_later


def _classify(best_lift: float, n_destinations: int, absorbs_other: bool) -> str:
    # Two destinations each well above chance means the earlier community broke apart.
    if n_destinations >= 2:
        return "SPLIT"
    if best_lift < _LIFT_SPLIT:
        return "DISSOLVED"
    if n_destinations == 1 and absorbs_other:
        return "MERGED"
    if best_lift > _LIFT_STABLE and n_destinations == 1:
        return "STABLE"
    return "unclassified"


def _community_table(run_a: str, run_b: str, algorithm: str) -> int:
    left = load_communities(run_a, algorithm)
    right = load_communities(run_b, algorithm)
    if not left or not right:
        missing = run_a if not left else run_b
        print(f"{missing} has no {algorithm}")
        return 1
    groups_a = _groups(left)
    groups_b = _groups(right)
    n_later = len(groups_b)
    # Node to its B community, so each A member is counted without scanning every B community.
    home_b = {node: int(community_id) for node, community_id in right.items()}
    # Each entry is (later community, containment, lift).
    overlaps: dict[int, list[tuple[int, float, float]]] = {}
    for community_id, members in groups_a.items():
        counts: dict[int, int] = {}
        for node in members:
            other = home_b.get(node)
            # Absent from B: counted in the denominator, and in no later community.
            if other is None:
                continue
            counts[other] = counts.get(other, 0) + 1
        scored = []
        for other, count in counts.items():
            contained = _containment(count, len(members))
            scored.append((other, contained, _lift(contained, n_later)))
        scored.sort(key=lambda item: (-item[2], item[0]))
        overlaps[community_id] = scored

    # Later communities that hold lift > 10 from at least two earlier communities.
    absorbed: set[int] = set()
    holders: dict[int, list[int]] = {}
    for community_id, scored in overlaps.items():
        for other, _contained, lift in scored:
            if lift > _LIFT_SPLIT:
                holders.setdefault(other, []).append(community_id)
    for other, sources in holders.items():
        if len(sources) >= 2:
            absorbed.add(other)

    print(f"null containment {_score(1 / n_later if n_later else 0.0)} (1/{n_later})")
    rows = []
    for community_id, members in sorted(groups_a.items(), key=lambda item: (-len(item[1]), item[0])):
        scored = overlaps[community_id]
        best_id, best, best_lift = scored[0] if scored else ("", 0.0, 0.0)
        n_destinations = sum(1 for _other, _contained, lift in scored if lift > _LIFT_SPLIT)
        absorbs_other = best_id in absorbed if scored else False
        label = _classify(best_lift, n_destinations, absorbs_other)
        rows.append((community_id, len(members), best_id, _score(best), f"{best_lift:.2f}", n_destinations, label))
    _print_table(
        ("a_community", "size", "b_community", "containment", "lift", "destinations", "classification"),
        rows,
    )

    # Reverse map: earlier communities with lift > 10 into this later community.
    fed: dict[int, list[tuple[int, float]]] = {}
    for community_id, scored in overlaps.items():
        for other, _contained, lift in scored:
            if lift > _LIFT_SPLIT:
                fed.setdefault(other, []).append((community_id, lift))
    print("FED")
    fed_rows = []
    for other, sources in sorted(fed.items(), key=lambda item: (-len(item[1]), item[0])):
        sources.sort(key=lambda item: (-item[1], item[0]))
        listed = ", ".join(f"{community_id}:{lift:.2f}" for community_id, lift in sources)
        fed_rows.append((other, listed))
    _print_table(("b_community", "a_communities"), fed_rows)
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
