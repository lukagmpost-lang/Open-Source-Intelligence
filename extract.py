"""Pull one community or one neighborhood out of a saved run.

Usage:
  python3 extract.py --run RUN_ID --community ID [--format graphml|json|html] --out FILE
  python3 extract.py --run RUN_ID --node NAME --radius R [--format ...] --out FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from main import write_graph
from osi.analysis import louvain_communities, pagerank
from osi.store import get_run, list_runs, load_communities, load_graph, load_metrics
from query import _closest_names, _score_cell
from viz.interactive import write_pyvis


def _load_run(run_id: str) -> tuple[nx.Graph, dict, dict] | None:
    meta = get_run(run_id)
    if meta is None:
        for saved_id, source, _created_at, _notes in list_runs():
            print(f"{saved_id} {source}")
        return None
    # Graphs are stored under the layer recorded in the run config, not under the run id.
    layer = (meta.get("config") or {}).get("layer")
    graph = load_graph(run_id, layer) if layer else None
    if graph is None:
        for saved_id, source, _created_at, _notes in list_runs():
            print(f"{saved_id} {source}")
        return None
    # Empty stored tables are falsy, so a run without those columns falls back to a fresh pass.
    communities = load_communities(run_id, "louvain") or louvain_communities(graph)
    scores = load_metrics(run_id, "pagerank") or pagerank(graph)
    return graph, communities, scores


def _community_sizes(communities: dict) -> list[tuple]:
    counts: dict = {}
    for community_id in communities.values():
        counts[community_id] = counts.get(community_id, 0) + 1
    # Largest first, then the id, matching the communities-by-size table.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _induced(graph: nx.Graph, nodes) -> nx.Graph:
    # subgraph().copy() keeps the edges that have both ends in the selection.
    return graph.subgraph(nodes).copy()


def _select(graph: nx.Graph, communities: dict, args: argparse.Namespace) -> nx.Graph | None:
    if args.community is not None:
        chosen = [node for node, community_id in communities.items() if community_id == args.community]
        if not chosen:
            for community_id, size in _community_sizes(communities):
                print(f"{community_id} {size}")
            return None
        return _induced(graph, chosen)
    if args.node not in graph:
        print(args.node)
        print("similar names:")
        for other in _closest_names(graph, args.node):
            print(f"  {other}")
        return None
    if args.radius > 5:
        print("warning: radius above 5 may be slow", file=sys.stderr)
    # Hop distance, including the source at 0. Same cutoff as the neighborhood query.
    lengths = nx.single_source_shortest_path_length(graph, args.node, cutoff=args.radius)
    return _induced(graph, lengths)


def _write(graph: nx.Graph, fmt: str, path: str, scores: dict, communities: dict) -> None:
    if fmt == "json":
        write_graph(graph, path)
        return
    if fmt == "html":
        # Stored PageRank and Louvain, not a fresh partition of the extract.
        write_pyvis(graph, path, scores, communities)
        return
    nx.write_graphml(graph, path)


def _summarize(graph: nx.Graph, communities: dict, scores: dict) -> None:
    represented = {communities[node] for node in graph.nodes if node in communities}
    print(f"nodes {graph.number_of_nodes()}")
    print(f"edges {graph.number_of_edges()}")
    print(f"communities {len(represented)}")
    ranked = sorted(graph.nodes, key=lambda node: (-float(scores.get(node, 0.0)), str(node)))
    for index, node in enumerate(ranked[:5], start=1):
        print(f"{index} {node} {_score_cell(float(scores.get(node, 0.0)))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a community or a neighborhood from a saved run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--community", type=int)
    parser.add_argument("--node")
    parser.add_argument("--radius", type=int)
    parser.add_argument("--format", choices=("graphml", "json", "html"), default="graphml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    has_community = args.community is not None
    has_node = args.node is not None or args.radius is not None
    # Exactly one mode: community, or a named node with a radius.
    if has_community == has_node:
        parser.error("pass --community ID, or pass both --node NAME and --radius R")
    if (args.node is None) != (args.radius is None):
        parser.error("pass both --node NAME and --radius R")
    if args.radius is not None and args.radius < 0:
        parser.error("--radius must be at least 0")
    loaded = _load_run(args.run)
    if loaded is None:
        return 1
    graph, communities, scores = loaded
    selected = _select(graph, communities, args)
    if selected is None:
        return 1
    _write(selected, args.format, args.out, scores, communities)
    _summarize(selected, communities, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
