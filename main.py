"""Build a public graph, analyze it, and write node-link JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from osi.analysis import (  # noqa: E402
    betweenness_centrality,
    degree_centrality,
    leiden_communities,
    louvain_communities,
    pagerank,
)
from osi.datasets import load_snap_facebook  # noqa: E402
from osi.graph import fetch_github_graph  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public social graph and analyze it.")
    parser.add_argument("--source", choices=("github", "snap_facebook"), default="github")
    parser.add_argument("--username", help="Public GitHub login. Required when --source is github.")
    parser.add_argument("--analyze", choices=("all", "centrality", "communities"), default="all")
    parser.add_argument("--out", default="graph.json", help="Node-link JSON output path.")
    args = parser.parse_args(argv)
    if args.source == "github" and not args.username:
        parser.error("--username is required when --source is github")
    return args


def build_graph(args: argparse.Namespace) -> nx.Graph:
    if args.source == "snap_facebook":
        return load_snap_facebook()
    return fetch_github_graph(args.username)


def community_sizes(membership: dict) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for community in membership.values():
        counts[int(community)] = counts.get(int(community), 0) + 1
    return sorted(counts.items())


def print_pagerank(graph: nx.Graph) -> None:
    degree_centrality(graph)
    ranks = pagerank(graph)
    # Betweenness is the slow centrality. It still runs when centrality is requested.
    betweenness_centrality(graph)
    print("PageRank top 20:")
    for index, (node, score) in enumerate(list(ranks.items())[:20], start=1):
        print(f"{index}. {node} {score:.6f}")


def print_communities(graph: nx.Graph) -> None:
    for name, membership in (
        ("Louvain", louvain_communities(graph)),
        ("Leiden", leiden_communities(graph)),
    ):
        sizes = community_sizes(membership)
        print(f"{name} communities: {len(sizes)}")
        for community, size in sizes:
            print(f"  community {community}: {size}")


def write_graph(graph: nx.Graph, path: str) -> None:
    payload = nx.node_link_data(graph, edges="links")
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph = build_graph(args)
    if args.analyze in ("all", "centrality"):
        print_pagerank(graph)
    if args.analyze in ("all", "communities"):
        print_communities(graph)
    write_graph(graph, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
