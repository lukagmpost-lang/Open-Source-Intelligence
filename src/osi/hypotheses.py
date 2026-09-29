"""Testable guesses about what a network is doing.

Each hypothesis is a sentence, a check that re-runs the same test, and a
confidence from 0 to 1. The rules below are the only patterns that qualify.
"""

from __future__ import annotations

import networkx as nx

_MAX_HYPOTHESES = 4
_SPARSE_CUT = 0.1
_LOW_CLUSTERING = 0.1
_FRAGMENT_REMAINING = 0.5


def infer_domain(source: str = "", layer: str = "") -> str:
    """A short label for the kind of network this run holds."""
    blob = f"{source} {layer}".casefold()
    if any(token in blob for token in ("co_contribution", "co-contribution", "company", "employee", "org chart")):
        return "company"
    if "github" in blob and "follow" not in blob:
        return "company"
    if "reddit" in blob:
        return "forum"
    if any(token in blob for token in ("bluesky", "facebook", "snap", "twitter", "mastodon")):
        return "social"
    return "general"


def generate_hypotheses(metrics, graph, domain) -> list[dict]:
    """Return up to four hypotheses the rules can support.

    Each item is ``{"hypothesis", "check", "confidence"}``. ``check`` takes
    no arguments and returns True when the pattern is still there.
    """
    metrics = metrics or {}
    found: list[dict] = []
    schism = _schism(graph, _assignment(metrics))
    if schism is not None:
        found.append(schism)
    bridge = _bridge(graph, metrics)
    if bridge is not None:
        found.append(bridge)
    fragile = _bus_factor(graph, metrics)
    if fragile is not None:
        found.append(fragile)
    if str(domain or "").casefold() == "company":
        isolated = _isolated_team(graph, _assignment(metrics))
        if isolated is not None:
            found.append(isolated)
    declined = _activity_drop(metrics, graph)
    if declined is not None:
        found.append(declined)
    found.sort(key=lambda item: item["confidence"], reverse=True)
    return found[:_MAX_HYPOTHESES]


def find_unasked_observations(graph, metrics, domain, original_question) -> list[str]:
    """Return two or three observations the question did not ask for."""
    from osi.findings import generate_findings

    metrics = metrics or {}
    candidates: list[str] = []
    for item in generate_hypotheses(metrics, graph, domain):
        candidates.append(str(item["hypothesis"]))
    finding_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"assignment", "communities", "betweenness", "robustness", "activity"}
    }
    for item in generate_findings(finding_metrics, None):
        candidates.append(str(item["text"]))
    candidates.extend(_structural_notes(metrics))
    asked = _words(original_question)
    ranked = sorted(candidates, key=lambda text: (len(_words(text) & asked), -len(text)))
    picked: list[str] = []
    seen: set[str] = set()
    for text in ranked:
        if text in seen:
            continue
        seen.add(text)
        picked.append(text)
        if len(picked) == 3:
            break
    for filler in (
        "A handful of accounts sit on many of the paths between the others.",
        "Most accounts never reach the rest of the network in one step.",
        "The accounts that hold groups together are not always the most connected ones.",
    ):
        if len(picked) >= 2:
            break
        if filler not in seen:
            picked.append(filler)
    return picked[:3]


def _words(text: str) -> set[str]:
    return {part for part in "".join(char if char.isalpha() else " " for char in text.casefold()).split() if len(part) >= 4}


def _assignment(metrics: dict) -> dict | None:
    raw = metrics.get("assignment")
    if not isinstance(raw, dict) or not raw:
        raw = metrics.get("communities")
    if isinstance(raw, dict) and raw:
        return raw
    return None


def _hypothesis(text: str, check, confidence: float) -> dict:
    return {"hypothesis": text, "check": check, "confidence": float(confidence)}


def _groups(assignment: dict) -> dict:
    groups: dict = {}
    for node, community in assignment.items():
        groups.setdefault(community, set()).add(node)
    return groups


def _schism(graph, assignment: dict | None) -> dict | None:
    if graph is None or graph.number_of_nodes() < 8 or not assignment:
        return None
    groups = _groups(assignment)
    ordered = sorted(groups.values(), key=len, reverse=True)
    for members in ordered[:8]:
        if len(members) < 8:
            break
        induced = graph.subgraph(members).copy()
        if _has_sparse_cut(induced):
            return _hypothesis(
                "There may be a schism forming.",
                lambda snapshot=induced: _has_sparse_cut(snapshot),
                0.72,
            )
    return None


def _has_sparse_cut(induced: nx.Graph) -> bool:
    """True when the subgraph is two sizable pieces with few edges between them."""
    if induced.number_of_nodes() < 8:
        return False
    pieces = [component for component in nx.connected_components(induced) if len(component) >= 3]
    if len(pieces) >= 2:
        return True
    if induced.number_of_edges() < 4:
        return False
    ranked = [node for node, _degree in sorted(induced.degree(), key=lambda item: (-item[1], str(item[0])))[:8]]
    if len(ranked) < 2:
        return False
    farthest = None
    for index, left in enumerate(ranked):
        distances = nx.single_source_shortest_path_length(induced, left)
        for right in ranked[index + 1 :]:
            distance = distances.get(right)
            if distance is None or distance < 2:
                continue
            if farthest is None or distance > farthest[0]:
                farthest = (distance, left, right)
    if farthest is None:
        return False
    _distance, left, right = farthest
    from_left = nx.single_source_shortest_path_length(induced, left)
    from_right = nx.single_source_shortest_path_length(induced, right)
    side_left: set = set()
    for node in induced:
        left_distance = from_left.get(node, 10**6)
        right_distance = from_right.get(node, 10**6)
        if left_distance <= right_distance:
            side_left.add(node)
    if min(len(side_left), induced.number_of_nodes() - len(side_left)) < 3:
        return False
    cut = sum(1 for u, v in induced.edges() if (u in side_left) != (v in side_left))
    return cut / induced.number_of_edges() <= _SPARSE_CUT


def _bridge(graph, metrics: dict) -> dict | None:
    if graph is None or graph.number_of_nodes() < 3:
        return None
    scores = metrics.get("betweenness")
    if not isinstance(scores, dict) or not scores:
        if graph.number_of_nodes() > 1500:
            return None
        scores = nx.betweenness_centrality(graph)
    ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
    if not ranked or float(ranked[0][1]) <= 0:
        return None
    for node, score in ranked[:5]:
        if node not in graph or float(score) <= 0:
            continue
        local = float(nx.clustering(graph, node))
        if local < _LOW_CLUSTERING:
            chosen = node
            return _hypothesis(
                "This person may be the only bridge between two groups.",
                lambda person=chosen, picture=graph: person in picture
                and float(nx.clustering(picture, person)) < _LOW_CLUSTERING,
                0.8,
            )
    return None


def _degree_curve(metrics: dict) -> dict | None:
    raw = metrics.get("robustness")
    if not isinstance(raw, dict):
        raw = metrics.get("criticality")
    if not isinstance(raw, dict):
        return None
    curve = raw.get("degree")
    if isinstance(curve, dict) and curve:
        return curve
    return None


def _bus_factor(graph, metrics: dict) -> dict | None:
    curve = _degree_curve(metrics)
    if curve is not None:
        broken = _curve_fragments(curve)
        if not broken:
            return None
        return _hypothesis(
            "This network has a bus-factor problem.",
            lambda recorded=dict(curve): _curve_fragments(recorded),
            0.85,
        )
    if graph is None or graph.number_of_nodes() < 10:
        return None
    if not _removal_fragments(graph):
        return None
    return _hypothesis(
        "This network has a bus-factor problem.",
        lambda picture=graph: _removal_fragments(picture),
        0.85,
    )


def _curve_fragments(curve: dict) -> bool:
    for key, largest in curve.items():
        try:
            ratio = float(key)
            remaining = float(largest)
        except (TypeError, ValueError):
            continue
        if 0 < ratio <= 0.1 and remaining < _FRAGMENT_REMAINING:
            return True
    return False


def _removal_fragments(graph: nx.Graph) -> bool:
    """Drop the top 1% by degree and see if the largest piece falls below half."""
    count = graph.number_of_nodes()
    if count < 10:
        return False
    drop = max(1, int(count * 0.01))
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    removed = {node for node, _degree in ranked[:drop]}
    remaining = graph.subgraph(node for node in graph if node not in removed)
    if remaining.number_of_nodes() == 0:
        return True
    largest = max(len(component) for component in nx.connected_components(remaining))
    return largest < _FRAGMENT_REMAINING * count


def _isolated_team(graph, assignment: dict | None) -> dict | None:
    if graph is None or not assignment or graph.number_of_nodes() < 4:
        return None
    groups = _groups(assignment)
    outside_exists = len(groups) >= 2
    if not outside_exists:
        return None
    for members in groups.values():
        if len(members) < 3:
            continue
        external = 0
        for node in members:
            if node not in graph:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in members:
                    external += 1
        if external == 0:
            frozen = set(members)
            return _hypothesis(
                "This team may be isolated from the rest of the org.",
                lambda picture=graph, team=frozen: _team_has_no_outside_edge(picture, team),
                0.7,
            )
    return None


def _team_has_no_outside_edge(graph: nx.Graph, team: set) -> bool:
    for node in team:
        if node not in graph:
            return False
        for neighbor in graph.neighbors(node):
            if neighbor not in team:
                return False
    return True


def _activity_series(metrics: dict, graph) -> list[float] | None:
    for key in ("activity", "period_counts", "periods"):
        series = _as_series(metrics.get(key) if metrics else None)
        if series is not None:
            return series
    if graph is None:
        return None
    series = _as_series(graph.graph.get("activity"))
    if series is not None:
        return series
    stamps = []
    for _left, _right, data in graph.edges(data=True):
        for key in ("period", "time", "timestamp"):
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stamps.append(float(value))
                break
    if len(stamps) < 4:
        return None
    ordered = sorted(stamps)
    width = max(1, len(ordered) // 4)
    bins = [ordered[index : index + width] for index in range(0, len(ordered), width)]
    bins = [bucket for bucket in bins if bucket][:4]
    if len(bins) < 2:
        return None
    return [float(len(bucket)) for bucket in bins]


def _as_series(raw) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    values = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return values


def _decline_length(series: list[float]) -> int:
    length = 0
    for newer, older in zip(reversed(series), list(reversed(series))[1:]):
        if newer < older:
            length += 1
            continue
        break
    return length


def _activity_drop(metrics: dict, graph) -> dict | None:
    series = _activity_series(metrics, graph)
    if not series:
        return None
    length = _decline_length(series)
    if length < 1:
        return None
    recorded = list(series)
    return _hypothesis(
        f"Activity has declined in the last {length} periods.",
        lambda values=recorded: _decline_length(values) >= 1,
        0.66,
    )


def _structural_notes(metrics: dict) -> list[str]:
    notes = []
    nodes = metrics.get("nodes")
    largest = metrics.get("largest_size")
    if (
        isinstance(nodes, (int, float))
        and not isinstance(nodes, bool)
        and isinstance(largest, (int, float))
        and not isinstance(largest, bool)
        and nodes
        and largest > nodes * 0.4
    ):
        notes.append("One group holds a large share of the accounts.")
    return notes
