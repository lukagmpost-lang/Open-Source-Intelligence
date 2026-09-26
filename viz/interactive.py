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


def _prune_light_edges(G: nx.Graph, min_edge_weight: float) -> nx.Graph:
    """Copy the graph without edges lighter than min_edge_weight."""
    pruned = nx.Graph()
    pruned.add_nodes_from(G.nodes)
    for left, right, data in G.edges(data=True):
        # Missing weight counts as 1, so the default threshold drops unweighted edges.
        if float(data.get("weight", 1.0)) < min_edge_weight:
            continue
        pruned.add_edge(left, right, **data)
    return pruned


def _select_component_nodes(components: list[set], scores: dict, max_nodes: int) -> list:
    """Keep whole components, then trim only a component larger than max_nodes."""
    def peak(component: set) -> float:
        return max(scores.get(node, 0.0) for node in component)

    # Highest peak PageRank first, so the main mass is considered before a side clique.
    nontrivial = sorted(
        (component for component in components if len(component) > 1),
        key=lambda component: (-peak(component), -len(component), min(map(str, component))),
    )
    chosen: list = []
    # A component bigger than the cap cannot be kept whole. Trim that one later.
    oversized = None
    for component in nontrivial:
        if len(chosen) + len(component) <= max_nodes:
            chosen.extend(component)
            continue
        if oversized is None and len(component) > max_nodes:
            oversized = component
    room = max_nodes - len(chosen)
    if room > 0 and oversized is not None:
        ranked = sorted(oversized, key=lambda node: (-scores.get(node, 0.0), str(node)))
        chosen.extend(ranked[:room])
        room = max_nodes - len(chosen)
    if room > 0:
        isolates = [next(iter(component)) for component in components if len(component) == 1]
        isolates.sort(key=lambda node: (-scores.get(node, 0.0), str(node)))
        chosen.extend(isolates[:room])
    return chosen


def filter_graph(G: nx.Graph, max_nodes: int = 1000, min_edge_weight: float = 2) -> nx.Graph:
    """Prune light edges, then keep components up to max_nodes."""
    # Order matters. Dropping light edges first is what defines the components.
    pruned = _prune_light_edges(G, min_edge_weight)
    scores = pagerank(pruned)
    components = list(nx.connected_components(pruned))
    keep = _select_component_nodes(components, scores, max_nodes)
    return pruned.subgraph(keep).copy()


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
