"""Tool catalog for LLM tool-use planning.

Each tool wraps an existing executor. The catalog is what the LLM sees
when it decides what to call.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from osi.analysis import (
    betweenness_centrality,
    degree_centrality,
    louvain_communities,
    pagerank,
)
from osi.store import load_graph, get_run


# ============================================================
# Tool catalog — what the LLM sees
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rank_nodes",
            "description": (
                "Rank accounts in the network by a centrality metric. "
                "Use direction='top' for most/highest/strongest/most important. "
                "Use direction='bottom' for least/lowest/weakest/least important."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["pagerank", "betweenness", "degree"],
                        "description": "Centrality metric to rank by.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["top", "bottom"],
                        "description": "top = highest scores, bottom = lowest scores",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many accounts to return (default 10)",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_communities",
            "description": "Return the communities (clusters) in the network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["louvain", "leiden"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "most_tension",
            "description": (
                "Return pairs of accounts with the most tense or hostile "
                "relationships, based on LLM-scored edges. Only works if "
                "edges have been scored (the graph contains llm_label)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_node",
            "description": (
                "Give details about a specific account: community, degree, "
                "centrality, and top connections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The account name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "neighbors",
            "description": "Return the accounts directly connected to a given account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "community_audit",
            "description": (
                "Report whether each community is a real social group or a "
                "hub-and-spoke artifact. Returns density and hub-share per "
                "community."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============================================================
# Dispatch — run the tool against the loaded graph
# ============================================================

def _load(run_id: str) -> nx.Graph:
    meta = get_run(run_id)
    if meta is None:
        raise RuntimeError(f"run {run_id} not found")
    layer = (meta.get("config") or {}).get("layer")
    if not layer:
        raise RuntimeError(f"run {run_id} has no layer")
    g = load_graph(run_id, layer)
    if g is None:
        raise RuntimeError(f"run {run_id} has no graph for layer {layer}")
    return g


def _rank_nodes(run_id, metric="pagerank", direction="top", limit=10):
    g = _load(run_id)
    if metric == "pagerank":
        scores = pagerank(g)
    elif metric == "betweenness":
        scores = betweenness_centrality(g)
    elif metric == "degree":
        scores = degree_centrality(g)
    else:
        raise ValueError(f"unknown metric: {metric}")
    reverse = (direction == "top")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1] if reverse else kv[1])
    return {
        "metric": metric,
        "direction": direction,
        "results": [{"name": n, "score": round(v, 5)} for n, v in ranked[:limit]],
    }


def _list_communities(run_id, algorithm="louvain"):
    g = _load(run_id)
    comms = louvain_communities(g)
    groups = {}
    for node, cid in comms.items():
        groups.setdefault(cid, []).append(node)
    pr = pagerank(g)
    sizes = sorted(
        (
            {
                "id": cid,
                "size": len(members),
                "hub": max(members, key=lambda m: pr.get(m, 0)),
            }
            for cid, members in groups.items()
        ),
        key=lambda x: -x["size"],
    )
    return {"algorithm": algorithm, "communities": sizes}


def _most_tension(run_id, limit=5):
    g = _load(run_id)
    rows = []
    for u, v, data in g.edges(data=True):
        label = data.get("llm_label")
        if label in ("tense", "hostile", "condescending", "dismissive"):
            rows.append({
                "a": u,
                "b": v,
                "label": label,
                "polarity": data.get("llm_polarity"),
                "reason": (data.get("llm_reason") or "")[:200],
            })
    rows.sort(key=lambda r: (r["polarity"] if r["polarity"] is not None else 0))
    return {"count": len(rows), "pairs": rows[:limit]}


def _explain_node(run_id, name):
    g = _load(run_id)
    if name not in g:
        return {"error": f"{name} not found in graph"}
    pr = pagerank(g)
    bc = betweenness_centrality(g)
    dc = degree_centrality(g)
    neighbors = sorted(g.neighbors(name), key=lambda n: -pr.get(n, 0))
    return {
        "name": name,
        "pagerank": round(pr.get(name, 0), 5),
        "betweenness": round(bc.get(name, 0), 5),
        "degree": round(dc.get(name, 0), 5),
        "message_count": g.nodes[name].get("message_count", 0),
        "top_neighbors": [
            {"name": n, "weight": g[name][n].get("weight", 1)}
            for n in neighbors[:8]
        ],
    }


def _neighbors(run_id, name, limit=10):
    g = _load(run_id)
    if name not in g:
        return {"error": f"{name} not found"}
    ns = sorted(
        g.neighbors(name),
        key=lambda n: -g[name][n].get("weight", 1),
    )[:limit]
    return {"name": name, "neighbors": ns}


def _community_audit(run_id):
    from osi.viz import audit_communities
    g = _load(run_id)
    rows = audit_communities(g)
    return {
        "communities": [
            {
                "id": r["id"],
                "size": r["size"],
                "hub": r["hub"],
                "density": r["density"],
                "hub_share": r["hub_share"],
                "verdict": r["verdict"],
            }
            for r in rows
        ]
    }


DISPATCH = {
    "rank_nodes": _rank_nodes,
    "list_communities": _list_communities,
    "most_tension": _most_tension,
    "explain_node": _explain_node,
    "neighbors": _neighbors,
    "community_audit": _community_audit,
}


def execute_tool(name: str, args: dict, run_id: str):
    """Run a tool call. Returns a structured result."""
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(run_id, **args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
