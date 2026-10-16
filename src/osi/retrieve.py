"""Cheap message retrieval. Pure Python."""
from __future__ import annotations

from collections import Counter


def execute_strategy(strategy: dict, messages: list) -> list:
    kind = strategy.get("kind")
    if kind == "sample_by_sender":
        names = set(strategy.get("names") or [])
        per = int(strategy.get("per_sender", 10))
        buckets = {n: [] for n in names}
        for m in reversed(messages):
            s = m.get("sender")
            if s in buckets and len(buckets[s]) < per:
                buckets[s].append(m)
        out = []
        for b in buckets.values():
            out.extend(b)
        return out

    if kind == "search":
        text = (strategy.get("text") or "").lower()
        limit = int(strategy.get("limit", 50))
        if not text:
            return []
        return [m for m in messages if text in (m.get("text") or "").lower()][:limit]

    if kind == "filter_by_sender":
        name = strategy.get("name")
        limit = int(strategy.get("limit", 100))
        return [m for m in messages if m.get("sender") == name][:limit]

    if kind == "sample_recent":
        n = int(strategy.get("n", 100))
        return messages[-n:]

    if kind == "mentions_others":
        # Messages where the sender names at least one other participant.
        limit = int(strategy.get("limit", 200))
        names = set(strategy.get("names") or [])
        if not names:
            names = {m.get("sender") for m in messages if m.get("sender")}
        lowered = [(n, n.lower()) for n in names if n]
        out = []
        for m in messages:
            sender = m.get("sender", "")
            text_lc = (m.get("text") or "").lower()
            for full, short in lowered:
                if full == sender:
                    continue
                if short and short in text_lc:
                    out.append(m)
                    break
            if len(out) >= limit:
                break
        return out

    return []


def execute_all(strategies: list, messages: list, cap: int = 300) -> list:
    seen = set()
    out = []
    for s in strategies or []:
        for m in execute_strategy(s, messages):
            key = (m.get("sender"), m.get("date"), m.get("text"))
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
            if len(out) >= cap:
                return out
    return out


def top_senders(messages: list, n: int = 20) -> list:
    counts = Counter(m.get("sender") for m in messages)
    return counts.most_common(n)
