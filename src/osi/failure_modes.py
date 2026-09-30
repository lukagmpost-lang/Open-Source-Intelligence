"""Reason from a domain to the failures that domain can actually have.

A forum, a company, and a movement do not fail in the same way. These
checks start from that context, then look at the graph for evidence.
"""

from __future__ import annotations

import json

import networkx as nx

from osi.analysis import _modularity_of, louvain_communities

_SEVERITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_STRUCTURAL = {
    "growth_fragmentation": 0,
    "silo_formation": 1,
    "bridge_loss": 2,
    "key_person_risk": 3,
    "moderation_concentration": 4,
    "manager_overload": 5,
    "hub_departure": 6,
    "communication_break": 7,
    "compromise": 8,
}
_LARGE_GRAPH = 2000


def top_betweenness_is_cross_community(graph) -> bool:
    """True when the highest-betweenness account has neighbors in two groups."""
    return _bridge_node(graph) is not None


def bridge_in_multiple_teams(graph) -> bool:
    """True when one employee is the link between two teams."""
    return _bridge_node(graph) is not None


def bridge_in_multiple_cells(graph) -> bool:
    """True when one member is the link between two cells."""
    return _bridge_node(graph) is not None


def moderation_like_top_accounts(graph) -> bool:
    """True when a few accounts reach many low-degree accounts."""
    if graph is None or graph.number_of_nodes() < 4 or graph.number_of_edges() == 0:
        return False
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    top_n = max(1, graph.number_of_nodes() // 100)
    top = [node for node, _degree in ranked[:top_n]]
    low = 0
    touched = 0
    for node in top:
        for neighbor in graph.neighbors(node):
            touched += 1
            if graph.degree(neighbor) <= 2:
                low += 1
    if touched == 0:
        return False
    edge_share = sum(graph.degree(node) for node in top) / (2 * graph.number_of_edges())
    return edge_share >= 0.3 and (low / touched) >= 0.5


DOMAIN_MODELS = {
    "forum": {
        "description": (
            "A user-driven discussion platform. "
            "Accounts are individuals, not employees. "
            "Popularity compounds. Moderation is "
            "decentralized."
        ),
        "failure_modes": [
            {
                "name": "hub_departure",
                "description": (
                    "A top poster stops participating. "
                    "Their community loses its anchor."
                ),
                "check": lambda m: m["max_degree"] > 50 * m["avg_degree"],
                "evidence": ["max_degree", "avg_degree"],
                "severity": "high",
            },
            {
                "name": "bridge_loss",
                "description": (
                    "A cross-community participant stops "
                    "posting. Topic clusters stop sharing "
                    "users."
                ),
                "check": lambda m, g: top_betweenness_is_cross_community(g),
                "evidence": ["betweenness_top", "betweenness_communities"],
                "severity": "high",
            },
            {
                "name": "growth_fragmentation",
                "description": (
                    "As the platform grows, the graph "
                    "fragments into more disconnected "
                    "pieces. This is expected, not a bug."
                ),
                "check": lambda m: m["components"] > m["nodes"] * 0.01,
                "evidence": ["components", "nodes"],
                "severity": "medium",
                "is_expected": True,
            },
            {
                "name": "moderation_concentration",
                "description": (
                    "Moderation authority concentrated "
                    "in a few accounts. If they stop, "
                    "subreddits decay."
                ),
                "check": lambda m, g: moderation_like_top_accounts(g),
                "evidence": ["top_accounts_are_mods"],
                "severity": "medium",
            },
        ],
    },
    "company": {
        "description": (
            "An organization. Accounts are employees. "
            "There is hierarchy. Retention is a "
            "management concern."
        ),
        "failure_modes": [
            {
                "name": "key_person_risk",
                "description": (
                    "A bridging employee leaves. Two "
                    "teams lose their translator."
                ),
                "check": lambda m, g: bridge_in_multiple_teams(g),
                "severity": "high",
            },
            {
                "name": "silo_formation",
                "description": "A team stops talking to others.",
                "check": lambda m: m["modularity"] > 0.7,
                "severity": "medium",
            },
            {
                "name": "manager_overload",
                "description": (
                    "One manager connects too many "
                    "reports, becomes a bottleneck."
                ),
                "check": lambda m, g: m["max_degree"] > 5 * m["median_degree"],
                "severity": "medium",
            },
        ],
    },
    "activist": {
        "description": (
            "A decentralized movement. Accounts "
            "are members. Compartmentalization is "
            "intentional. Opsec matters."
        ),
        "failure_modes": [
            {
                "name": "compromise",
                "description": (
                    "A bridge member is compromised. "
                    "Multiple cells are exposed."
                ),
                "check": lambda m, g: bridge_in_multiple_cells(g),
                "severity": "critical",
            },
            {
                "name": "communication_break",
                "description": "A courier stops. Cells lose contact.",
                "check": lambda m: m["betweenness_max"] > 0.1,
                "severity": "high",
            },
        ],
    },
}


def apply_failure_modes(metrics, graph, domain):
    """Return the failure modes whose check passes, with evidence and severity."""
    model = DOMAIN_MODELS.get(str(domain or "").casefold())
    if model is None:
        return []
    metrics = _with_defaults(metrics, graph)
    matched = []
    for mode in model["failure_modes"]:
        if not _fires(mode["check"], metrics, graph):
            continue
        matched.append(
            {
                "name": mode["name"],
                "description": mode["description"],
                "severity": mode["severity"],
                "is_expected": mode.get("is_expected", False),
                "evidence": _evidence(mode, metrics, graph),
            }
        )
    return matched


def rank_by_question(matched, question):
    """Order failure modes by what the question is actually asking.

    A worry question keeps the severe surprises and drops modes that are
    expected for that domain. A structure question prefers the shape of
    the graph. A vulnerability question sorts by severity and keeps the
    expected modes.
    """
    text = str(question or "").casefold()
    if "worried" in text or "worry" in text:
        kept = [mode for mode in matched if not mode.get("is_expected")]
        return sorted(kept, key=lambda mode: _SEVERITY.get(str(mode.get("severity")), 9))
    if "structured" in text or "structure" in text or "how is" in text:
        return sorted(matched, key=lambda mode: _STRUCTURAL.get(mode.get("name"), 50))
    if "vulnerable" in text:
        return sorted(matched, key=lambda mode: _SEVERITY.get(str(mode.get("severity")), 9))
    return list(matched)


def worry_prompt(domain: str, modes: list[dict]) -> str:
    """The synthesis instructions for a worry question."""
    lines = []
    for mode in modes:
        evidence = mode.get("evidence") or {}
        bits = ", ".join(f"{key}={_shown(value)}" for key, value in evidence.items())
        extra = f" Evidence: {bits}." if bits else ""
        lines.append(f"- {mode['name']} ({mode['severity']}): {mode['description']}{extra}")
    body = "\n".join(lines) if lines else "- none"
    return (
        "The user asked what they should be worried about. "
        f"Based on the domain (a {domain} network) and the metrics, these failure modes apply:\n\n"
        f"{body}\n\n"
        "Write 3-5 sentences explaining which of these matters most and why. "
        "Name specific accounts. Give specific numbers. State what would happen.\n\n"
        "Do not describe the shape of the network. Explain the risk."
    )


def quantifying_tools(modes: list[dict], graph) -> list[tuple[str, dict]]:
    """Tools that put numbers on the top failure modes."""
    planned: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for mode in modes[:3]:
        for name, params in _plans(str(mode.get("name")), graph):
            key = name + json.dumps(params, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            planned.append((name, params))
    return planned


def _plans(name: str, graph) -> list[tuple[str, dict]]:
    if name in {"hub_departure", "manager_overload", "moderation_concentration"}:
        return [("rank_nodes", {"metric": "degree", "top": 5})]
    if name in {"bridge_loss", "key_person_risk", "compromise", "communication_break"}:
        node = _bridge_node(graph) or _top_betweenness_node(graph)
        if node is not None:
            return [("explain_node", {"node": node})]
        if graph is not None and graph.number_of_nodes() <= _LARGE_GRAPH:
            return [("critical_nodes", {"top_n": 5})]
        return [("list_communities", {"algorithm": "louvain"})]
    if name in {"silo_formation", "growth_fragmentation"}:
        return [("list_communities", {"algorithm": "louvain"})]
    return []


def _fires(check, metrics, graph) -> bool:
    try:
        try:
            return bool(check(metrics, graph))
        except TypeError as error:
            if "positional" not in str(error):
                raise
            return bool(check(metrics))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, nx.NetworkXError):
        return False


def _with_defaults(metrics, graph) -> dict:
    metrics = dict(metrics or {})
    if graph is None:
        return metrics
    degrees = [degree for _node, degree in graph.degree()]
    nodes = graph.number_of_nodes()
    metrics.setdefault("nodes", nodes)
    if degrees:
        metrics.setdefault("max_degree", max(degrees))
        metrics.setdefault("avg_degree", sum(degrees) / len(degrees))
        ordered = sorted(degrees)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = float(ordered[mid])
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2
        metrics.setdefault("median_degree", median)
    else:
        metrics.setdefault("max_degree", 0)
        metrics.setdefault("avg_degree", 0.0)
        metrics.setdefault("median_degree", 0.0)
    if "components" not in metrics:
        raw = metrics.get("num_components")
        metrics["components"] = int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else (
            nx.number_connected_components(graph) if nodes else 0
        )
    scores = metrics.get("betweenness")
    if isinstance(scores, dict) and scores:
        graph.graph.setdefault("betweenness", scores)
    assignment = metrics.get("assignment") or metrics.get("communities")
    if isinstance(assignment, dict) and assignment:
        graph.graph.setdefault("assignment", assignment)
    cached = _betweenness_scores(graph)
    if cached and "betweenness_max" not in metrics:
        metrics["betweenness_max"] = max(float(value) for value in cached.values())
    if metrics.get("modularity") is None and 0 < nodes <= _LARGE_GRAPH and graph.number_of_edges():
        groups = _group_assignment(graph)
        if groups:
            try:
                metrics["modularity"] = float(_modularity_of(graph, groups))
            except (ZeroDivisionError, nx.NetworkXError, ValueError):
                pass
    return metrics


def _evidence(mode: dict, metrics: dict, graph) -> dict:
    shown = {}
    for key in mode.get("evidence") or []:
        if key == "betweenness_top":
            shown[key] = _top_betweenness_node(graph)
            continue
        if key == "betweenness_communities":
            shown[key] = _neighbor_group_count(graph, _top_betweenness_node(graph))
            continue
        if key == "top_accounts_are_mods":
            shown[key] = True
            continue
        value = metrics.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            shown[key] = value
    if mode["name"] == "hub_departure" and graph is not None and graph.number_of_nodes():
        node, degree = max(graph.degree(), key=lambda item: (item[1], str(item[0])))
        shown["top_account"] = str(node)
        shown["top_degree"] = int(degree)
    if mode["name"] in {"bridge_loss", "key_person_risk", "compromise"}:
        node = _bridge_node(graph)
        if node is not None:
            shown["bridge_account"] = str(node)
    return shown


def _betweenness_scores(graph) -> dict:
    if graph is None:
        return {}
    cached = graph.graph.get("betweenness")
    if isinstance(cached, dict) and cached:
        return cached
    if graph.number_of_nodes() == 0 or graph.number_of_nodes() > _LARGE_GRAPH:
        return {}
    scores = nx.betweenness_centrality(graph)
    graph.graph["betweenness"] = scores
    return scores


def _group_assignment(graph) -> dict:
    if graph is None:
        return {}
    cached = graph.graph.get("assignment")
    if isinstance(cached, dict) and cached:
        return cached
    if graph.number_of_nodes() == 0 or graph.number_of_nodes() > _LARGE_GRAPH:
        return {}
    assignment = louvain_communities(graph)
    graph.graph["assignment"] = assignment
    return assignment


def _top_betweenness_node(graph):
    scores = _betweenness_scores(graph)
    if not scores:
        return None
    node = max(scores, key=lambda name: (float(scores[name]), str(name)))
    if float(scores[node]) <= 0 or graph is None or node not in graph:
        return None
    return node


def _bridge_node(graph):
    node = _top_betweenness_node(graph)
    if node is None:
        return None
    if _neighbor_group_count(graph, node) < 2:
        return None
    return node


def _neighbor_group_count(graph, node) -> int:
    if graph is None or node is None or node not in graph:
        return 0
    assignment = _group_assignment(graph)
    if not assignment:
        return 0
    groups = {assignment.get(neighbor) for neighbor in graph.neighbors(node)}
    groups.discard(None)
    return len(groups)


def _shown(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
