"""Shared co-participation edges.

Two accounts are linked when they both show up in the same group
(a repository, a thread). The weight is how many groups they share.
A group bigger than the cap is skipped so one crowded room cannot
become a clique of hundreds of pairs. That is the Reddit thread rule.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx

# Same ceiling as THREAD_AUTHOR_CAP in layers/reddit_archive.py.
GROUP_CAP = 30


def co_participation_graph(
    groups: list[list[str]],
    layer: str,
    cap: int = GROUP_CAP,
) -> tuple[nx.Graph, int, int]:
    """Build the weighted graph. Returns the graph, groups over the cap, and groups under 2.

    ``groups`` must already have bots and blank names removed. Spellings that
    differ only by case are one account; the first spelling is kept.
    """
    weights: dict[tuple[str, str], int] = defaultdict(int)
    spellings: dict[str, str] = {}
    over = 0
    under = 0
    for raw in groups:
        accounts: list[str] = []
        seen: set[str] = set()
        for name in raw:
            text = str(name).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            accounts.append(spellings.setdefault(key, text))
        # One participant cannot form a pair. Over the cap, the pairs would
        # be the whole room rather than a real shared activity.
        if len(accounts) < 2:
            under += 1
            continue
        if len(accounts) > cap:
            over += 1
            continue
        for left, right in combinations(sorted(accounts, key=str.casefold), 2):
            weights[(left, right)] += 1

    graph = nx.Graph()
    for (left, right), weight in weights.items():
        graph.add_node(left, layer=layer)
        graph.add_node(right, layer=layer)
        graph.add_edge(left, right, weight=float(weight), layer=layer)
    return graph, over, under
