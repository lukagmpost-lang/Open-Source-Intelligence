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


def to_interactive_html(G: nx.Graph, output: str = "graph.html") -> str:
    """Write a self-contained HTML file with search, filter, and physics."""
    print(f"interactive before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    scores = pagerank(G)
    communities = louvain_communities(G)
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
    for node in G.nodes:
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
    for left, right in G.edges:
        net.add_edge(str(left), str(right))
    net.write_html(output, notebook=False)
    print(f"interactive after: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved {output}")
    return output
