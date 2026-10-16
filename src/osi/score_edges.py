"""Batch-score graph edges with the LLM, with a disk cache."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from osi.loaders.whatsapp import _parse_export
from osi.sentiment_llm import score_edge_llm

CACHE_PATH = Path.home() / ".osi" / "edge_scores.json"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def score_top_edges(
    graph: nx.Graph,
    messages: list[dict],
    top_n: int = 40,
    force: bool = False,
) -> int:
    """Score the top-N edges by weight. Cached by (sender_a, sender_b)."""
    cache = _load_cache()
    edges = sorted(graph.edges(data=True), key=lambda e: -e[2].get("weight", 0))[:top_n]

    scored = 0
    skipped = 0
    for i, (u, v, data) in enumerate(edges, 1):
        key = _key(u, v)
        if not force and key in cache:
            result = cache[key]
            skipped += 1
        else:
            print(f"[{i}/{len(edges)}] scoring {u} <-> {v}...", flush=True)
            try:
                result = score_edge_llm(messages, u, v)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
            cache[key] = result
            _save_cache(cache)
            scored += 1

        graph[u][v]["llm_label"] = result.get("label")
        graph[u][v]["llm_polarity"] = result.get("polarity")
        graph[u][v]["llm_reason"] = result.get("reason")

    print(f"\nscored: {scored}, from cache: {skipped}, total: {len(edges)}")
    return scored


if __name__ == "__main__":
    chat = "/Users/2950639/Downloads/_chat.txt"
    from osi.loaders.whatsapp import load_whatsapp_export

    _, messages = _parse_export(chat)
    g, _ = load_whatsapp_export(chat, edge_rule="sequence")
    score_top_edges(g, messages, top_n=40)

    # Save the augmented graph
    out = Path("/Users/2950639/Downloads/graph_scored.json")
    data = nx.node_link_data(g)
    out.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"saved to {out}")

    from osi.viz import render_interactive_html
    render_interactive_html(g, "/Users/2950639/Downloads/graph.html",
                            top_edges=120, min_edge_weight=2.0, top_labels=40)
    print("rendered /Users/2950639/Downloads/graph.html")
