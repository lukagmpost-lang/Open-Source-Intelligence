from __future__ import annotations

"""LLM-based sentiment scoring for edges in a WhatsApp chat graph.

This module provides functions to score the relationship between two users
based on their messages using an LLM, and to score multiple edges in a graph.
"""

from osi.answer import call_llm
import sys


def score_edge_llm(messages: list[dict], sender_a: str, sender_b: str) -> dict:
    """Score the relationship between two senders using an LLM.

    Args:
        messages: List of parsed message dictionaries (each with 'sender' and 'text').
        sender_a: First sender name.
        sender_b: Second sender name.

    Returns:
        A dictionary with keys: 'polarity' (float), 'label' (str), 'reason' (str).
    """
    # Filter messages to only those from sender_a or sender_b
    filtered = [
        msg for msg in messages
        if msg.get("sender") == sender_a or msg.get("sender") == sender_b
    ]

    # Balanced sample: top 5 by length from each sender
    from_a = sorted(
        [m for m in filtered if m.get("sender") == sender_a],
        key=lambda m: len(m.get("text", "")),
        reverse=True,
    )[:5]
    from_b = sorted(
        [m for m in filtered if m.get("sender") == sender_b],
        key=lambda m: len(m.get("text", "")),
        reverse=True,
    )[:5]

    # Balanced sample: top 5 from each sender (up to 10 total)
    selected = from_a + from_b
    selected.sort(key=lambda m: m.get("date"))

    # If fewer than 4 messages, return insufficient
    if len(selected) < 4:
        return {
            "polarity": 0.0,
            "label": "insufficient",
            "reason": "fewer than 4 messages"
        }

    # Build the prompt
    n = len(selected)
    prompt_lines = [
        "You are analyzing the relationship between two people in a group chat.",
        "",
        f"Person A: {sender_a}",
        f"Person B: {sender_b}",
        "",
        f"Here are {n} messages between them:",
    ]
    for msg in selected:
        sender = msg.get("sender", "unknown")
        text = msg.get("text", "").strip()
        prompt_lines.append(f"{sender}: {text}")
    prompt_lines.extend([
        "",
        "On a scale from -1 (hostile, cold) to +1 (warm, affectionate), how",
        "would you score the overall relationship between these two people?",
        "",
        "Pay close attention to TONE and POWER DYNAMICS:",
        "  - One person may be overly formal or polite while the other is casual,",
        "    blunt, or emotional.",
        "  - Note who talks more, who initiates, and who responds.",
        "  - Notice if one person is managing, correcting, or talking down to the",
        "    other while maintaining a superficially correct tone.",
        "  - Do NOT confuse formal politeness with warmth. A politely worded",
        "    rebuke is still a rebuke.",
        "Consider sarcasm, jokes, and context — not just individual words.",
        "",
        "Reply in exactly this format:",
        "SCORE: <number between -1 and 1>",
        "LABEL: <one of: warm, friendly, neutral, tense, hostile, condescending>",
        "REASON: <2-3 sentences describing the tone dynamic>",
    ])
    prompt = "\n".join(prompt_lines)

    try:
        # Call the LLM with temperature=0.0
        response = call_llm(prompt, temperature=0.0)
        # Parse the response
        polarity = 0.0
        label = "unknown"
        reason = ""
        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    polarity = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("LABEL:"):
                label = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return {
            "polarity": polarity,
            "label": label,
            "reason": reason
        }
    except Exception as e:
        return {
            "polarity": 0.0,
            "label": "error",
            "reason": str(e)
        }


def score_graph_edges_llm(graph, messages, top_n=20) -> dict:
    """Score the top edges in a graph by message count using an LLM.

    Args:
        graph: A NetworkX graph where nodes are senders and edges have at least
               a 'message_count' attribute.
        messages: List of parsed message dictionaries.
        top_n: Maximum number of edges to score (default 20).

    Returns:
        A dictionary summary with keys: 'scored' (int), 'skipped' (int),
        'results' (list of dicts for each scored edge).
    """
    # Filter edges to those with message_count >= 5
    eligible_edges = []
    for u, v, data in graph.edges(data=True):
        if data.get("message_count", 0) >= 5:
            eligible_edges.append((u, v, data["message_count"]))

    # Sort by message_count descending
    eligible_edges.sort(key=lambda x: x[2], reverse=True)

    # Take top_n
    selected_edges = eligible_edges[:top_n]

    results = []
    scored = 0
    skipped = 0

    for i, (u, v, msg_count) in enumerate(selected_edges, start=1):
        # Print progress to stderr
        print(f"[sentiment_llm] scoring edge {i}/{len(selected_edges)}: {u} <-> {v}", file=sys.stderr)
        # Score the edge
        result = score_edge_llm(messages, u, v)
        # Attach to the edge
        graph[u][v]["llm_polarity"] = result["polarity"]
        graph[u][v]["llm_label"] = result["label"]
        graph[u][v]["llm_reason"] = result["reason"]
        # Record result
        results.append({
            "edge": (u, v),
            "message_count": msg_count,
            "polarity": result["polarity"],
            "label": result["label"],
            "reason": result["reason"]
        })
        scored += 1

    # The skipped count is the number of eligible edges beyond top_n that we didn't score
    skipped = max(0, len(eligible_edges) - top_n)

    return {
        "scored": scored,
        "skipped": skipped,
        "results": results
    }