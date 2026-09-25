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
    compare_centralities,
    compare_communities,
    cross_reference,
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
    parser.add_argument(
        "--compare",
        choices=("all", "centrality", "communities", "cross"),
        help="Print a side-by-side comparison. Omit this flag to keep the original report.",
    )
    parser.add_argument("--out", default="graph.json", help="Node-link JSON output path.")
    # Draw graph.png after the text report. Off unless this flag is present.
    parser.add_argument("--plot", action="store_true", help="Save graph.png after the analysis.")
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


def print_centrality_table(centralities: dict, top_n: int = 10) -> None:
    # The example table shows these four. Eigenvector stays in the data for callers.
    columns = ["degree", "betweenness", "closeness", "pagerank"]
    print("TOP NODES BY EACH CENTRALITY MEASURE:")
    print("Rank | Degree | Betweenness | Closeness | PageRank")
    for rank in range(1, top_n + 1):
        cells = [str(rank)]
        for name in columns:
            ordered = centralities["order"][name]
            cells.append(str(ordered[rank - 1]) if rank <= len(ordered) else "")
        print(" | ".join(cells))
    if centralities.get("eigenvector_error"):
        print(f"Eigenvector centrality failed: {centralities['eigenvector_error']}")


def print_community_table(communities: dict) -> None:
    print("COMMUNITY ALGORITHMS:")
    print("Method | Communities | Modularity")
    for name, label in (("louvain", "Louvain"), ("leiden", "Leiden"), ("girvan_newman", "Girvan-Newman")):
        block = communities[name]
        if block.get("skipped"):
            print(f"{label} | skipped | {block['message']}")
            continue
        print(f"{label} | {block['count']} | {block['modularity']:.6f}")


def print_comparison(graph: nx.Graph, mode: str) -> None:
    # Cross-reference needs both results, so "all" and "cross" compute both.
    need_centrality = mode in ("all", "centrality", "cross")
    need_communities = mode in ("all", "communities", "cross")
    centralities = compare_centralities(graph) if need_centrality else None
    communities = compare_communities(graph) if need_communities else None
    if mode in ("all", "centrality") and centralities is not None:
        print_centrality_table(centralities)
    if mode in ("all", "communities") and communities is not None:
        print_community_table(communities)
    if mode in ("all", "cross") and centralities is not None and communities is not None:
        cross_reference(graph, communities, centralities)


def save_plot(graph: nx.Graph, path: str = "graph.png") -> None:
    # PageRank sets the dot size and decides which few nodes get a name.
    scores = pagerank(graph)
    # Louvain community ids are the color key. tab20 cycles every 20 groups.
    communities = louvain_communities(graph)
    # Betweenness order is strongest first, so the first five are the bridges.
    bridges = list(betweenness_centrality(graph))[:5]

    # Draw without a window. Agg writes a file on a machine with no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same seed and spring settings every run, so the picture does not jump around.
    positions = nx.spring_layout(graph, k=0.15, iterations=50, seed=42)
    figure, axes = plt.subplots(figsize=(20, 20))

    # Scale scores into pixel areas. 20000 makes hubs large and leaves the rest small.
    node_size = [scores.get(node, 0.0) * 20000 for node in graph.nodes]
    node_color = [communities.get(node, 0) for node in graph.nodes]
    # Faint edges keep the communities visible without a solid gray sheet.
    nx.draw_networkx_edges(graph, positions, ax=axes, alpha=0.15, width=0.3)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.tab20,
        linewidths=0,
    )

    # A red ring on the top 5 betweenness nodes, using the same size as the fill.
    if bridges:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=bridges,
            ax=axes,
            node_size=[scores.get(node, 0.0) * 20000 for node in bridges],
            node_color="none",
            edgecolors="red",
            linewidths=2,
        )

    # Names only for the handful of nodes above 0.003, about the top five on SNAP.
    labels = {node: str(node) for node, score in scores.items() if score > 0.003}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axes, font_size=10)
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("Saved graph.png")


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
    if args.compare:
        print_comparison(graph, args.compare)
    write_graph(graph, args.out)
    # Plot last so "Saved graph.png" is the final line of the run.
    if args.plot:
        save_plot(graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
