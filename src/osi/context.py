"""Build a compact numeric context for the LLM."""
from __future__ import annotations

import networkx as nx

from osi.analysis import (
    betweenness_centrality,
    degree_centrality,
    louvain_communities,
    pagerank,
)


def build_context(graph: nx.Graph, top_n: int = 15) -> dict:
    """Return compact string tables — not nested dicts — for the LLM."""
    pr = pagerank(graph)
    bc = betweenness_centrality(graph)
    dc = degree_centrality(graph)

    node_comm = {}
    comms = []
    try:
        comms = list(nx.community.louvain_communities(graph, seed=42))
        for i, c in enumerate(comms):
            for node in c:
                node_comm[node] = i
    except Exception:
        pass

    ranked = sorted(pr.items(), key=lambda kv: -kv[1])[:top_n]

    node_lines = ["name | pagerank | betw | deg | msgs | comm"]
    for name, _ in ranked:
        node_lines.append(
            f"{name} | {pr.get(name,0):.4f} | {bc.get(name,0):.4f} | "
            f"{dc.get(name,0):.4f} | {graph.nodes[name].get('message_count',0)} | "
            f"{node_comm.get(name,'?')}"
        )
    node_table = "\n".join(node_lines)

    comm_lines = ["id | size | hub | density | hub_share | verdict"]
    for i, members in enumerate(comms):
        n = len(members)
        max_int = n * (n - 1) / 2
        internal = [(u, v) for u, v in graph.edges() if u in members and v in members]
        density = len(internal) / max_int if max_int else 0
        hub = max(members, key=lambda m: pr.get(m, 0))
        hub_e = sum(1 for u, v in internal if hub in (u, v))
        hub_share = hub_e / len(internal) if internal else 0
        if density > 0.3 and hub_share < 0.6:
            verdict = "real"
        elif hub_share > 0.75 and density < 0.15:
            verdict = "artifact"
        else:
            verdict = "mixed"
        comm_lines.append(
            f"{i} | {n} | {hub} | {density:.2f} | {hub_share:.2f} | {verdict}"
        )
    comm_table = "\n".join(comm_lines)

    edge_lines = ["a <-> b | weight | pos | neg | label | reason"]
    for u, v, data in graph.edges(data=True):
        if data.get("llm_label"):
            r = (data.get("llm_reason") or "")[:100].replace("\n", " ")
            edge_lines.append(
                f"{u} <-> {v} | {data.get('weight',1)} | "
                f"{data.get('positive_count',0)} | {data.get('negative_count',0)} | "
                f"{data.get('llm_label')} | {r}"
            )
    if len(edge_lines) == 1:
        edge_lines.append("(none scored yet)")
    edge_table = "\n".join(edge_lines)

    return {
        "node_table": node_table,
        "comm_table": comm_table,
        "edge_table": edge_table,
        "summary": (
            f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
            f"communities={len(comms)}"
        ),
    }
