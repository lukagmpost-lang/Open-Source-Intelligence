"""Turn raw graph metrics into plain-language observations.

Each rule checks one pattern. A match becomes a finding with the sentence,
an interest score from 0 to 10, and the metric that triggered it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import networkx as nx

_BASELINES_PATH = Path(__file__).with_name("baselines.json")
# Clustering and assortativity are expensive on the large stored graphs.
_AUTO_MEASURE_LIMIT = 5000


def _finding(text: str, score: float, metric: str) -> dict:
    return {"text": text, "score": float(score), "metric": metric}


def _present(metrics: dict, *keys):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _nodes(metrics: dict, graph) -> int:
    recorded = _present(metrics, "nodes", "n_nodes")
    if recorded is not None:
        return int(recorded)
    if graph is not None:
        return int(graph.number_of_nodes())
    return 0


def _can_measure(graph) -> bool:
    return graph is not None and 0 < graph.number_of_nodes() <= _AUTO_MEASURE_LIMIT


def _avg_degree(metrics: dict, graph) -> float | None:
    recorded = _present(metrics, "avg_degree")
    if recorded is not None:
        return float(recorded)
    if graph is None or graph.number_of_nodes() == 0:
        return None
    total = sum(degree for _node, degree in graph.degree())
    return total / graph.number_of_nodes()


def _max_degree(metrics: dict, graph) -> int | None:
    recorded = _present(metrics, "max_degree")
    if recorded is not None:
        return int(recorded)
    if graph is None or graph.number_of_nodes() == 0:
        return None
    return max(degree for _node, degree in graph.degree())


def _top_node(metrics: dict, graph) -> str:
    recorded = _present(metrics, "top_node")
    if recorded:
        return str(recorded)
    if graph is None or graph.number_of_nodes() == 0:
        return "one account"
    node, _degree = max(graph.degree(), key=lambda item: (item[1], str(item[0])))
    return str(node)


def _top_accounts(metrics: dict, graph) -> list[str]:
    recorded = metrics.get("top_accounts")
    if recorded:
        return [str(node) for node in list(recorded)[:5]]
    if graph is None or graph.number_of_nodes() == 0:
        return []
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    return [str(node) for node, _degree in ranked[:5]]


def _top_5_degree(metrics: dict, graph) -> int | None:
    recorded = _present(metrics, "top_5_degree")
    if recorded is not None:
        return int(recorded)
    if graph is None or graph.number_of_nodes() < 5:
        return None
    ranked = sorted((degree for _node, degree in graph.degree()), reverse=True)
    return int(ranked[4])


def _components(metrics: dict, graph) -> int | None:
    recorded = _present(metrics, "components", "num_components")
    if recorded is not None:
        return int(recorded)
    if graph is None or graph.number_of_nodes() == 0:
        return None
    return int(nx.number_connected_components(graph))


def _clustering(metrics: dict, graph) -> float | None:
    recorded = _present(metrics, "avg_clustering", "clustering")
    if recorded is not None:
        return float(recorded)
    if not _can_measure(graph):
        return None
    return float(nx.average_clustering(graph))


def _assortativity(metrics: dict, graph) -> float | None:
    recorded = _present(metrics, "assortativity")
    if recorded is not None:
        return float(recorded)
    if not _can_measure(graph) or graph.number_of_edges() == 0:
        return None
    try:
        value = float(nx.degree_assortativity_coefficient(graph))
    except nx.NetworkXError:
        return None
    if value != value:  # NaN
        return None
    return value


def _name_list(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else "nobody"
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _degree_rules(metrics: dict, graph) -> list[dict]:
    found: list[dict] = []
    average = _avg_degree(metrics, graph)
    if average is None:
        return found
    if average > 20:
        found.append(
            _finding(
                "This is a dense network — most accounts connect to dozens of others, "
                "which means a message can cross the group in a few introductions.",
                min(10, average / 5),
                "avg_degree",
            )
        )
    if average < 3:
        found.append(
            _finding(
                "This is a sparse network — most accounts have only a few connections, "
                "which means most people only ever hear from a couple of others.",
                6 if average < 2 else 4,
                "avg_degree",
            )
        )
    maximum = _max_degree(metrics, graph)
    if maximum is None:
        return found
    if average > 0 and maximum > 50 * average:
        ratio = maximum / average
        found.append(
            _finding(
                f"{_top_node(metrics, graph)} is connected to {maximum:,} others — {ratio:.0f}x the average. "
                "That means one person reaches far more people than anyone else.",
                8,
                "max_degree",
            )
        )
    if maximum > 500:
        found.append(
            _finding(
                f"One account, {_top_node(metrics, graph)}, is connected to {maximum:,} others. "
                "That means one person knows more people than most people will ever meet.",
                9,
                "max_degree",
            )
        )
    return found


def _community_rules(metrics: dict, graph) -> list[dict]:
    found: list[dict] = []
    modularity = _present(metrics, "modularity")
    if modularity is not None:
        modularity = float(modularity)
        if modularity > 0.6:
            found.append(
                _finding(
                    "The communities are clean and distinct, which means people mostly talk inside their own group.",
                    7,
                    "modularity",
                )
            )
        if modularity < 0.3:
            found.append(
                _finding(
                    "Communities overlap heavily, which means most accounts belong to several groups at once.",
                    6,
                    "modularity",
                )
            )
    communities = _present(metrics, "n_communities")
    nodes = _nodes(metrics, graph)
    if communities is not None:
        count = int(communities)
        if count > 100 and nodes and count > nodes * 0.02:
            found.append(
                _finding(
                    f"The network fragments into {count:,} tiny groups, "
                    "which means most people never meet anyone outside their own corner.",
                    7,
                    "n_communities",
                )
            )
    largest = _present(metrics, "largest_size")
    if largest is not None and nodes and int(largest) > nodes * 0.5:
        found.append(
            _finding(
                f"One giant group contains {int(largest):,} accounts — over half the network — "
                "which means that group sets the tone for everyone else.",
                8,
                "largest_size",
            )
        )
    return found


def _component_rules(metrics: dict, graph) -> list[dict]:
    components = _components(metrics, graph)
    if components is None:
        return []
    if components > 100:
        return [
            _finding(
                f"The network is highly fragmented — it breaks into {components:,} disconnected islands, "
                "which means most accounts never see each other.",
                9,
                "components",
            )
        ]
    if components == 1:
        return [
            _finding(
                "Everyone is reachable from everyone else, which means a message can find any account.",
                5,
                "components",
            )
        ]
    return []


def _clustering_rules(metrics: dict, graph) -> list[dict]:
    clustering = _clustering(metrics, graph)
    if clustering is None:
        return []
    if clustering > 0.7:
        return [
            _finding(
                "Neighbors of each account are also connected to each other, "
                "which means friends of friends already know each other.",
                7,
                "clustering",
            )
        ]
    if clustering < 0.1:
        return [
            _finding(
                "Accounts connect to hubs but not to each other — this is a broadcast pattern, "
                "which means one voice reaches everyone and the audience does not talk back.",
                8,
                "clustering",
            )
        ]
    return []


def _assortativity_rules(metrics: dict, graph) -> list[dict]:
    mixing = _assortativity(metrics, graph)
    if mixing is None:
        return []
    if mixing < -0.5:
        return [
            _finding(
                "Hubs connect to isolated accounts, not to each other, "
                "which means removing the top hubs would break the network.",
                9,
                "assortativity",
            )
        ]
    if mixing > 0.3:
        return [
            _finding(
                "The most connected accounts cluster together, which means there's an inner circle.",
                7,
                "assortativity",
            )
        ]
    if -0.1 < mixing < 0.1:
        return [
            _finding(
                "Connections are mixed, which means there's no clear elite and no single voice everyone else depends on.",
                3,
                "assortativity",
            )
        ]
    return []


def _power_law_rules(metrics: dict, graph) -> list[dict]:
    recorded = _present(metrics, "power_law_r_squared")
    if recorded is None:
        return []
    if float(recorded) > 0.85:
        return [
            _finding(
                "A few accounts have almost all the connections and everyone else has a handful, "
                "which means the network depends on those few accounts.",
                6,
                "power_law",
            )
        ]
    return []


def _score_rows(scores) -> list[tuple[str, float]]:
    pairs = scores.items() if isinstance(scores, dict) else scores
    rows = []
    for node, value in pairs or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        rows.append((str(node), number))
    return rows


def pagerank_centrality_text(scores) -> str | None:
    """Name the leader when its PageRank is more than 10x the median."""
    rows = _score_rows(scores)
    if not rows:
        return None
    rows.sort(key=lambda item: (-item[1], item[0]))
    top_node, top_pagerank = rows[0]
    median_pagerank = statistics.median([value for _node, value in rows])
    if top_pagerank > 10 * median_pagerank:
        ratio = top_pagerank / median_pagerank
        return (
            f"{top_node} is {ratio:.0f}x more central than the typical account. "
            "That means one person sits on far more of the conversation than anyone else."
        )
    return None


def _pagerank_rules(metrics: dict, graph) -> list[dict]:
    scores = metrics.get("pagerank") if isinstance(metrics, dict) else None
    if not isinstance(scores, dict) or not scores:
        return []
    text = pagerank_centrality_text(scores)
    if not text:
        return []
    return [_finding(text, 9, "pagerank")]


def _rank_rules(metrics: dict, graph) -> list[dict]:
    found: list[dict] = []
    average = _avg_degree(metrics, graph)
    top_degree = _top_5_degree(metrics, graph)
    if average and top_degree is not None and top_degree > 5 * average:
        names = _name_list(_top_accounts(metrics, graph))
        who = f" — {names} —" if names and names != "nobody" else ""
        found.append(
            _finding(
                f"The top five accounts{who} are each connected to at least {top_degree:,} people, "
                f"while the average account knows around {average:.0f}. "
                "That means a handful of accounts do almost all the connecting.",
                7,
                "top_degree",
            )
        )
    communities = metrics.get("top_communities")
    if not communities or len(communities) < 5 or any(item is None for item in communities):
        return found
    distinct = {item for item in communities}
    if len(distinct) == 5:
        found.append(
            _finding(
                "The hubs are spread across communities, which means no single group dominates.",
                8,
                "hub_communities",
            )
        )
    elif len(distinct) == 1:
        found.append(
            _finding(
                f"The top {len(communities)} accounts all sit in the same community, "
                "which means there's a clear inner circle.",
                9,
                "hub_communities",
            )
        )
    return found


def load_baselines() -> dict:
    """Reference metrics for the six known graphs."""
    return json.loads(_BASELINES_PATH.read_text(encoding="utf-8"))


def typical_baseline() -> dict[str, float]:
    """Median of each field across the reference graphs."""
    rows = list(load_baselines().values())
    typical: dict[str, float] = {}
    for key in rows[0]:
        values = sorted(float(row[key]) for row in rows)
        middle = len(values) // 2
        if len(values) % 2:
            typical[key] = values[middle]
        else:
            typical[key] = (values[middle - 1] + values[middle]) / 2
    return typical


def _comparison_rules(metrics: dict, graph) -> list[dict]:
    baseline = typical_baseline()
    found: list[dict] = []
    components = _components(metrics, graph)
    if components is not None and baseline["components"] and components > baseline["components"] * 10:
        ratio = components / baseline["components"]
        found.append(
            _finding(
                f"The network is shattered into {components:,} disconnected fragments — "
                f"{ratio:.0f}x more pieces than a typical network — "
                "which means most accounts never see each other.",
                9,
                "baseline_components",
            )
        )
    modularity = _present(metrics, "modularity")
    if modularity is not None and float(modularity) > baseline["modularity"] * 1.3:
        found.append(
            _finding(
                "Communities are much more distinct than typical, "
                "which means groups barely talk to each other.",
                7,
                "baseline_modularity",
            )
        )
    maximum = _max_degree(metrics, graph)
    if maximum is not None and maximum > baseline["max_degree"] * 2:
        found.append(
            _finding(
                "The top hub is twice as large as typical, "
                "which means one account reaches far more people than a normal network would.",
                8,
                "baseline_max_degree",
            )
        )
    return found


_RULES = (
    _degree_rules,
    _community_rules,
    _component_rules,
    _clustering_rules,
    _assortativity_rules,
    _power_law_rules,
    _rank_rules,
    _comparison_rules,
    _pagerank_rules,
)


def generate_findings(metrics: dict, graph=None) -> list[dict]:
    """Apply every rule.

    Each match has ``text``, ``score`` (0-10), and ``metric``. The list is
    sorted with the most interesting finding first.
    """
    metrics = dict(metrics or {})
    found: list[dict] = []
    for rule in _RULES:
        found.extend(rule(metrics, graph))
    found.sort(key=lambda item: item["score"], reverse=True)
    return found


def pick_top_findings(findings: list[dict], n: int = 4) -> list[str]:
    """Return the top N finding texts.

    Two findings about the same metric keep only the higher-scored one.
    """
    ordered = sorted(findings, key=lambda item: item["score"], reverse=True)
    seen: set[str] = set()
    texts: list[str] = []
    for item in ordered:
        metric = item["metric"]
        if metric in seen:
            continue
        seen.add(metric)
        texts.append(item["text"])
        if len(texts) == n:
            break
    return texts


HUB_METRICS = {"max_degree", "top_degree", "hub_communities", "pagerank"}
COMMUNITY_METRICS = {"modularity", "n_communities", "largest_size", "baseline_modularity"}
