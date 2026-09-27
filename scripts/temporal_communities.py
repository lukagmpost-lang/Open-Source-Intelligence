"""Multislice communities for stored graph runs.

    python3 scripts/temporal_communities.py --runs r2008-v2,r2012-v2 --omega 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.store import get_run, load_graph  # noqa: E402
from osi.temporal_communities import (  # noqa: E402
    community_persistence,
    mean_persistence,
    multislice_communities,
    multislice_modularity,
)


def _load_run(run_id: str):
    """The graph saved with the run. The store is not written."""
    meta = get_run(run_id)
    if meta is None:
        raise SystemExit(f"run {run_id} is not in the store")
    layer = (meta.get("config") or {}).get("layer")
    graph = load_graph(run_id, layer)
    if graph is None:
        raise SystemExit(f"run {run_id} has no graph")
    return graph


def _print_report(names: list[str], graphs, assignments, omega: float) -> None:
    modularity = multislice_modularity(graphs, assignments, omega)
    print(f"omega {omega}")
    print(f"modularity {modularity:.6f}")
    for name, graph, assignment in zip(names, graphs, assignments):
        print(
            f"{name}: {len(set(assignment.values()))} communities, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
    # Each step is one pair of neighboring snapshots. Two runs produce one step.
    for index in range(len(assignments) - 1):
        rows = community_persistence(assignments[index], assignments[index + 1])
        print(
            f"persistence {names[index]} -> {names[index + 1]}: "
            f"{mean_persistence(rows):.6f} over {len(rows)} communities with shared members"
        )
        print("community  size  shared  fraction")
        for community, size, shared, fraction in rows:
            print(f"{community}  {size}  {shared}  {fraction:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multislice communities for stored runs.")
    parser.add_argument("--runs", required=True, help="Comma-separated store run ids, in time order.")
    parser.add_argument("--omega", type=float, required=True, help="Inter-slice coupling strength.")
    args = parser.parse_args(argv)
    names = [part.strip() for part in args.runs.split(",") if part.strip()]
    if len(names) < 2:
        raise SystemExit("pass at least two runs")
    graphs = [_load_run(name) for name in names]
    assignments = multislice_communities(graphs, args.omega)
    _print_report(names, graphs, assignments, args.omega)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
