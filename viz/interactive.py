"""Standalone force-directed HTML for one NetworkX graph."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from osi.analysis import louvain_communities, pagerank


def _community_color(community_id: int) -> str:
    # tab20 repeats every 20 communities, matching the static plot.
    red, green, blue, _alpha = plt.colormaps["tab20"](community_id % 20)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def filter_graph(G: nx.Graph, max_nodes: int = 1000, min_edge_weight: float = 2) -> nx.Graph:
    """Keep the top PageRank nodes and drop edges lighter than min_edge_weight."""
    scores = pagerank(G)
    # Tie-break on the node id so the same graph always keeps the same cut.
    ranked = sorted(G.nodes, key=lambda node: (-scores.get(node, 0.0), str(node)))
    keep = set(ranked[:max_nodes])
    filtered = nx.Graph()
    filtered.add_nodes_from(node for node in ranked if node in keep)
    for left, right, data in G.edges(data=True):
        if left not in keep or right not in keep:
            continue
        # Missing weight counts as 1, so the default threshold drops unweighted edges.
        if float(data.get("weight", 1.0)) < min_edge_weight:
            continue
        filtered.add_edge(left, right, **data)
    return filtered


def to_interactive_html(
    G: nx.Graph,
    output: str = "graph.html",
    max_nodes: int = 1000,
    min_edge_weight: float = 2,
) -> str:
    """Write a self-contained HTML file with search, filter, and physics."""
    print(f"interactive before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Always filter. Drawing every Reddit edge makes PyVis write for minutes.
    view = filter_graph(G, max_nodes=max_nodes, min_edge_weight=min_edge_weight)
    print(f"filtered: {view.number_of_nodes()} nodes, {view.number_of_edges()} edges")
    scores = pagerank(view)
    communities = louvain_communities(view)
    # in_line embeds the scripts so the file works without a network connection.
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1c1917",
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )
    # Barnes-Hut is PyVis's force-directed layout.
    net.barnes_hut()
    net.toggle_physics(True)
    for node in view.nodes:
        community = communities.get(node, 0)
        score = scores.get(node, 0.0)
        net.add_node(
            str(node),
            label=str(node),
            # Same size scale as the static plot: PageRank times 20000.
            size=max(score * 20000, 1),
            color=_community_color(community),
            title=f"user: {node} | community: {community} | pagerank: {score:.6f}",
        )
    for left, right in view.edges:
        net.add_edge(str(left), str(right))
    net.write_html(output, notebook=False, open_browser=False)
    print(f"interactive after: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved {output}")
    return output
