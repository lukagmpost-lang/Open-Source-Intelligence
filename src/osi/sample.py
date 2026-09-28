"""A small NetworkX example for the local viewer.

Three dense groups share one hub. The hub's degree is higher than any other node.
"""

from __future__ import annotations

import networkx as nx

# Twelve people in each group, plus the hub, stays inside the 30–50 node example.
_GROUP_SIZE = 12
_GROUPS = ("alpha", "beta", "gamma")


def sample_graph() -> nx.Graph:
    """Return an undirected graph with three communities and one clear hub."""
    graph = nx.Graph()
    for name in _GROUPS:
        members = [f"{name}-{index}" for index in range(_GROUP_SIZE)]
        for index, member in enumerate(members):
            # A ring plus the skip-one chords makes a community without a full clique.
            graph.add_edge(member, members[(index + 1) % _GROUP_SIZE], weight=1.0)
            graph.add_edge(member, members[(index + 2) % _GROUP_SIZE], weight=1.0)
    # The hub sits in alpha and also reaches the other two groups.
    for index in range(_GROUP_SIZE):
        graph.add_edge("hub", f"alpha-{index}", weight=1.0)
    for name in ("beta", "gamma"):
        graph.add_edge("hub", f"{name}-0", weight=1.0)
        graph.add_edge("hub", f"{name}-1", weight=1.0)
    # One bridge keeps beta and gamma in the same component as alpha.
    graph.add_edge("beta-6", "gamma-6", weight=1.0)
    return graph
