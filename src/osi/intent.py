"""Classify a question into an intent and a scope, then name the tools it needs.

The model reads the question. Phrase lists do not.
"""

from __future__ import annotations

import json
import re

from osi.answer import call_llm

_INTENTS = ("describe", "rank", "diagnose", "explain", "compare", "target", "locate", "trace")
_SCOPES = ("whole", "community", "node", "edge", "time_range")
_INTENT_SET = set(_INTENTS)
_SCOPE_SET = set(_SCOPES)

INTENT_TOOLS = {
    "describe": ["network_health", "list_communities"],
    "rank": ["rank_nodes"],
    "diagnose": ["critical_nodes", "structural_criticality"],
    "explain": ["explain_node"],
    "compare": ["network_health"],
    "target": ["rank_nodes", "explain_node"],
    "locate": ["explain_node"],
    "trace": ["connectivity"],
}

DOMAIN_ADJUSTMENTS = {
    "company": {
        "diagnose": ["critical_nodes", "structural_criticality", "network_health"],
    },
    "activist": {
        "diagnose": ["critical_nodes", "structural_criticality"],
    },
    "general": {},
}

_CLASSIFIER_SYSTEM = "You classify questions about networks. Reply with JSON only."

_CLASSIFIER_PROMPT = """\
Read this question about a network and classify it.

Question: {question}

Intents (pick one or more):
  describe  — asking what the network looks like
  rank      — asking for a top list or ordering
  diagnose  — asking what could go wrong, where it's weak, or what to worry about
  explain   — asking why something is the way it is
  compare   — asking about a difference between two things
  target    — asking for specific names, not categories
  locate    — asking where something sits in the network
  trace     — asking how two things are connected

Scope (pick one):
  whole       — the entire graph
  community   — one community
  node        — one account
  edge        — a relationship between two accounts
  time_range  — a period of time

Reply with JSON:
{{"intents": ["diagnose", "target"], "scope": "whole", "scope_target": null}}

Examples:
"what should I be worried about" → {{"intents": ["diagnose", "target"], "scope": "whole"}}
"who is alice" → {{"intents": ["explain", "locate"], "scope": "node", "scope_target": "alice"}}
"how is alice connected to bob" → {{"intents": ["trace"], "scope": "edge", "scope_target": ["alice", "bob"]}}
"how fragmented is this" → {{"intents": ["describe"], "scope": "whole"}}
"what changed since 2008" → {{"intents": ["compare"], "scope": "time_range", "scope_target": "2008"}}
"""

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def classify_intent(question: str) -> dict:
    """Use the LLM to classify the question into an intent and scope.

    Returns ``intents``, ``scope``, and ``scope_target``. ``scope_target`` is
    the specific node, community, edge, or time when the scope is not the
    whole graph.
    """
    prompt = _CLASSIFIER_PROMPT.format(question=question)
    try:
        reply = call_llm(prompt, system=_CLASSIFIER_SYSTEM)
    except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
        return _fallback()
    return _parse(reply)


def required_tools(intents, scope, domain) -> list[str]:
    """Combine the tool sets for each intent, apply domain adjustments, and drop duplicates.

    ``scope`` travels with the classification so callers can pass it through.
    The tool list is decided by the intents and the domain.
    """
    del scope
    ordered: list[str] = []
    for intent in intents or []:
        for name in INTENT_TOOLS.get(intent, []):
            if name not in ordered:
                ordered.append(name)
    primary = intents[0] if intents else "general"
    adjustments = DOMAIN_ADJUSTMENTS.get(domain) or {}
    for name in adjustments.get(primary) or []:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _fallback() -> dict:
    return {"intents": ["describe"], "scope": "whole", "scope_target": None}


def _parse(reply: str) -> dict:
    match = _OBJECT.search(reply or "")
    if match is None:
        return _fallback()
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _fallback()
    if not isinstance(payload, dict):
        return _fallback()
    raw_intents = payload.get("intents")
    if isinstance(raw_intents, str):
        raw_intents = [raw_intents]
    intents = []
    for item in raw_intents or []:
        name = str(item).strip().lower()
        if name in _INTENT_SET and name not in intents:
            intents.append(name)
    if not intents:
        return _fallback()
    scope = str(payload.get("scope") or "").strip().lower()
    if scope not in _SCOPE_SET:
        scope = "whole"
    target = payload.get("scope_target", None)
    if target == "":
        target = None
    return {"intents": intents, "scope": scope, "scope_target": target}
