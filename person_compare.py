"""Compare the five logins that sit in both the GitHub multi-ego layer and Reddit.

Usage:
  python3 person_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.fingerprint import compute_fingerprint, fingerprint_similarity
from osi.github_graph import MULTI_EGO_LAYER, MULTI_EGO_RUN
from osi.store import get_run, load_communities, load_graph, load_metrics

# The five same-name accounts found in both layers, in the order they were named.
PEOPLE = ("antirez", "spez", "rtomayko", "hadley", "chromakode")
_REDDIT_RUN = "r2008-v2"
# Top 10% of a score list. rank/n <= 0.10 is above the 90th percentile.
_HIGH_FRACTION = 0.10
_FIELDS = ("degree", "pagerank", "betweenness", "closeness")


def _placement(scores: dict, user: str) -> tuple[int, int, float]:
    """1-based rank in a strongest-first map, plus how many nodes were scored."""
    for rank, (node, value) in enumerate(scores.items(), start=1):
        if node == user:
            return rank, len(scores), float(value)
    raise KeyError(user)


def _degree_scores(graph) -> dict:
    """Degree, highest first. A tie breaks on the node name, matching the stored metrics."""
    pairs = ((node, float(degree)) for node, degree in graph.degree())
    return dict(sorted(pairs, key=lambda item: (-item[1], str(item[0]))))


def _above_percentile(rank: int, total: int) -> bool:
    # Same ratio as report.py: position divided by the number of scored nodes.
    return total > 0 and (rank / total) <= _HIGH_FRACTION


def classify(github_rank: int, github_total: int, reddit_rank: int, reddit_total: int) -> str:
    """PageRank standing on each layer. High means above the 90th percentile."""
    github_high = _above_percentile(github_rank, github_total)
    reddit_high = _above_percentile(reddit_rank, reddit_total)
    if github_high and reddit_high:
        return "UNIVERSAL"
    if github_high and not reddit_high:
        return "GITHUB-DOMINANT"
    if reddit_high and not github_high:
        return "REDDIT-DOMINANT"
    return "PERIPHERAL"


def _bundle(run_id: str) -> dict:
    metrics = {name: load_metrics(run_id, name) for name in ("pagerank", "betweenness", "closeness")}
    metrics["community"] = load_communities(run_id, "louvain")
    return metrics


def _community_sizes(community: dict) -> dict:
    sizes: dict = {}
    for community_id in community.values():
        sizes[community_id] = sizes.get(community_id, 0) + 1
    return sizes


def _score_cell(value: float, rank: int, total: int | None) -> str:
    shown = f"{int(value)}" if float(value).is_integer() else f"{value:.6f}"
    # Degree is the line that carries the graph size. The other ranks stay short.
    if total is None:
        return f"{shown} (rank {rank})"
    return f"{shown} (rank {rank} / {total})"


def _rows(github_scores: dict, reddit_scores: dict, user: str) -> list[tuple[str, str, str]]:
    rendered = []
    for name in _FIELDS:
        github_rank, github_total, github_value = _placement(github_scores[name], user)
        reddit_rank, _reddit_total, reddit_value = _placement(reddit_scores[name], user)
        # Only degree prints the denominator. The header already names which graph is which.
        denominator = github_total if name == "degree" else None
        reddit_denominator = _reddit_total if name == "degree" else None
        rendered.append(
            (
                name,
                _score_cell(github_value, github_rank, denominator),
                _score_cell(reddit_value, reddit_rank, reddit_denominator),
            )
        )
    return rendered


def _print_person(user: str, github_scores: dict, reddit_scores: dict, similarity: float, kind: str) -> None:
    rows = _rows(github_scores, reddit_scores, user)
    github_community, github_size = github_scores["community_cell"][user]
    reddit_community, reddit_size = reddit_scores["community_cell"][user]
    rows.append(("community", f"{github_community} (size {github_size})", f"{reddit_community} (size {reddit_size})"))
    width = max(len(cell) for _label, cell, _other in rows)
    print(user)
    print(f"  {'':<14}{'GITHUB':<{width}}  REDDIT")
    for label, github_cell, reddit_cell in rows:
        print(f"  {label:<14}{github_cell:<{width}}  {reddit_cell}")
    print(f"  {'fingerprint':<14}{similarity:.4f}")
    print(f"  {'class':<14}{kind}")
    print()


def report(people: tuple[str, ...] = PEOPLE) -> int:
    reddit_meta = get_run(_REDDIT_RUN)
    if reddit_meta is None:
        print(f"missing run {_REDDIT_RUN}", file=sys.stderr)
        return 1
    reddit_layer = (reddit_meta.get("config") or {}).get("layer")
    reddit = load_graph(_REDDIT_RUN, reddit_layer) if reddit_layer else None
    github = load_graph(MULTI_EGO_RUN, MULTI_EGO_LAYER)
    if reddit is None or github is None:
        print("missing GitHub multi-ego graph or Reddit layer", file=sys.stderr)
        return 1
    reddit_metrics = _bundle(_REDDIT_RUN)
    github_metrics = _bundle(MULTI_EGO_RUN)
    github_scores = {name: github_metrics[name] for name in ("pagerank", "betweenness", "closeness")}
    reddit_scores = {name: reddit_metrics[name] for name in ("pagerank", "betweenness", "closeness")}
    github_scores["degree"] = _degree_scores(github)
    reddit_scores["degree"] = _degree_scores(reddit)
    github_sizes = _community_sizes(github_metrics["community"])
    reddit_sizes = _community_sizes(reddit_metrics["community"])
    github_scores["community_cell"] = {
        node: (community_id, github_sizes[community_id])
        for node, community_id in github_metrics["community"].items()
    }
    reddit_scores["community_cell"] = {
        node: (community_id, reddit_sizes[community_id])
        for node, community_id in reddit_metrics["community"].items()
    }
    # One line so the class is read off PageRank, not off raw scores from graphs of different sizes.
    print("class uses PageRank rank; above the 90th percentile means rank/nodes <= 0.10")
    print()
    for user in people:
        # Both vectors are within-graph rank percentiles, so cosine compares profiles, not raw scores.
        similarity = fingerprint_similarity(
            compute_fingerprint(github, user, github_metrics),
            compute_fingerprint(reddit, user, reddit_metrics),
        )
        github_rank, github_total, _value = _placement(github_scores["pagerank"], user)
        reddit_rank, reddit_total, _value = _placement(reddit_scores["pagerank"], user)
        kind = classify(github_rank, github_total, reddit_rank, reddit_total)
        _print_person(user, github_scores, reddit_scores, similarity, kind)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv:
        print("usage: python3 person_compare.py", file=sys.stderr)
        return 2
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
