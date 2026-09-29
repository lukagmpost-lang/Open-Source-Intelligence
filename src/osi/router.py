"""Turn a question into an executor name and its parameters.

``route`` does not load a graph and does not call an executor.
"""

from __future__ import annotations

import re

# The four score names rank_nodes already accepts. Anything else is not a metric.
_METRICS = {"pagerank", "degree", "betweenness", "closeness"}
_DEFAULT_TOP = 10

_TOP_BY = re.compile(r"^top (\d+) by (\S+)$", re.IGNORECASE)
_TOP = re.compile(r"^top (\d+) (\S+)$", re.IGNORECASE)
_PATH = re.compile(r"^path from (\S+) to (\S+)$", re.IGNORECASE)
_DISTANCE = re.compile(r"^distance between (\S+) and (\S+)$", re.IGNORECASE)
_CONNECTED = re.compile(r"^are (\S+) and (\S+) connected$", re.IGNORECASE)
_ABOUT = re.compile(r"^tell me about (\S+)$", re.IGNORECASE)
_WHO = re.compile(r"^who is (\S+)$", re.IGNORECASE)
_DESCRIBE = re.compile(r"^describe (\S+)$", re.IGNORECASE)

# Longer phrases first so "most connections" is not read as something shorter.
_RANK_PHRASES = (
    ("most important", "pagerank"),
    ("most influential", "pagerank"),
    ("most connections", "degree"),
    ("most connected", "degree"),
    ("connects the most", "betweenness"),
    ("closest to everyone", "closeness"),
    ("well located", "closeness"),
)
_RANK_WORDS = (
    ("bridges", "betweenness"),
    ("brokers", "betweenness"),
)
_HEALTH_PHRASES = ("how healthy", "power law")
_HEALTH_WORDS = ("health", "assortativity", "clustering")
_CRITICAL_PHRASES = (
    "what happens if we remove",
    "what if we take out",
    "how fragile",
)
_CRITICAL_WORDS = ("robustness", "resilience")
_COMMUNITY_PHRASES = ("how many communities", "what communities exist")
_COMMUNITY_WORDS = ("communities", "groups", "clusters")


def route(question: str, run_id: str) -> dict:
    """Parse a natural-language question into {intent, params}.

    Returns dict with:
      intent: rank_nodes | list_communities | network_health |
              structural_criticality | connectivity | explain_node |
              unsupported
      params: dict for that executor
      confidence: "high" | "low"
    Never calls an executor. Only parses.
    """
    text = _normalize(question)
    if not text:
        return _answer("unsupported", {}, "low", run_id)

    parsed = (
        _rank_top(text)
        or _connectivity(text)
        or         _explain(text)
        or _critical(text)
        or _phrase_rank(text)
        or _health(text)
        or _communities(text)
    )
    if parsed is None:
        return _answer("unsupported", {}, "low", run_id)
    intent, params = parsed
    return _answer(intent, params, "high", run_id)


def _normalize(question: str) -> str:
    # Collapse whitespace first so "  path  from  alice  to  bob  " is one pattern.
    text = re.sub(r"\s+", " ", question.strip())
    # A trailing ? or . is punctuation on the question, not part of a node name.
    return text.rstrip("?.").strip()


def _clean_capture(token: str) -> str:
    # \S+ keeps dots inside a handle. Only the punctuation stuck to the end comes off.
    return token.rstrip(".,?;:!")


def _answer(intent: str, params: dict, confidence: str, run_id: str) -> dict:
    # ``run`` is the executor argument. ``run_id`` is only the name this function receives.
    payload = {"run": run_id, **params}
    return {"intent": intent, "params": payload, "confidence": confidence}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _rank_top(text: str) -> tuple[str, dict] | None:
    match = _TOP_BY.match(text) or _TOP.match(text)
    if match is None:
        return None
    metric = match.group(2).casefold()
    if metric not in _METRICS:
        return None
    return "rank_nodes", {"metric": metric, "top": int(match.group(1))}


def _phrase_rank(text: str) -> tuple[str, dict] | None:
    folded = text.casefold()
    for phrase, metric in _RANK_PHRASES:
        if phrase in folded:
            return "rank_nodes", {"metric": metric, "top": _DEFAULT_TOP}
    for word, metric in _RANK_WORDS:
        if _has_word(text, word):
            return "rank_nodes", {"metric": metric, "top": _DEFAULT_TOP}
    return None


def _connectivity(text: str) -> tuple[str, dict] | None:
    for pattern in (_PATH, _DISTANCE, _CONNECTED):
        match = pattern.match(text)
        if match is not None:
            return "connectivity", {
                "source": _clean_capture(match.group(1)),
                "target": _clean_capture(match.group(2)),
            }
    return None


def _explain(text: str) -> tuple[str, dict] | None:
    for pattern in (_ABOUT, _WHO, _DESCRIBE):
        match = pattern.match(text)
        if match is not None:
            return "explain_node", {"node": _clean_capture(match.group(1))}
    return None


def _critical(text: str) -> tuple[str, dict] | None:
    folded = text.casefold()
    if any(phrase in folded for phrase in _CRITICAL_PHRASES):
        return "structural_criticality", {}
    if any(_has_word(text, word) for word in _CRITICAL_WORDS):
        return "structural_criticality", {}
    return None


def _health(text: str) -> tuple[str, dict] | None:
    folded = text.casefold()
    if any(phrase in folded for phrase in _HEALTH_PHRASES):
        return "network_health", {}
    if any(_has_word(text, word) for word in _HEALTH_WORDS):
        return "network_health", {}
    return None


def _communities(text: str) -> tuple[str, dict] | None:
    folded = text.casefold()
    if any(phrase in folded for phrase in _COMMUNITY_PHRASES):
        return "list_communities", {"algorithm": "louvain"}
    if any(_has_word(text, word) for word in _COMMUNITY_WORDS):
        return "list_communities", {"algorithm": "louvain"}
    return None
