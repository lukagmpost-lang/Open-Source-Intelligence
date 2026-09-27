"""Match accounts by name and by structural fingerprint."""

from __future__ import annotations

from typing import Any

import networkx as nx

from osi.fingerprint import compute_fingerprint, fingerprint_similarity, normalize_fingerprints

# Name agreement is the smaller share. Structure has to carry the match.
_STRUCTURE_WEIGHT = 0.7
_NAME_WEIGHT = 0.3
_CONFIRMED_SIM = 0.7
_STRUCTURAL_ONLY_SIM = 0.8
_TOP_STRUCTURAL = 5


def _name_exact(reddit_handle: str, github_handle: str) -> bool:
    # GitHub logins are case-insensitive, so IHaveAnIdea and IhaveAnIdea are one name.
    return reddit_handle.casefold() == github_handle.casefold()


def _row(reddit_handle: str, github_handle: str, struct_sim: float, name_exact: bool) -> dict[str, Any]:
    return {
        "reddit": reddit_handle,
        "github": github_handle,
        "struct_sim": struct_sim,
        "name_exact": name_exact,
        "combined_score": _STRUCTURE_WEIGHT * struct_sim + _NAME_WEIGHT * (1.0 if name_exact else 0.0),
    }


def match_by_structure(
    G_github: nx.Graph,
    G_reddit: nx.Graph,
    name_candidates: list[tuple[str, str]],
    reddit_nodes: list | None = None,
    github_metrics: dict | None = None,
    reddit_metrics: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Score name pairs and the closest structural neighbors.

    Returns confirmed matches (same name and similarity above 0.7), structural-only
    matches (no shared name, similarity above 0.8, at most five GitHub nodes per
    Reddit user), and name-only pairs (same name, similarity at or below 0.7).
    """
    github_metrics = github_metrics or {}
    reddit_metrics = reddit_metrics or {}
    if reddit_nodes is None:
        reddit_nodes = list(G_reddit.nodes)
    github_by_fold = {str(node).casefold(): node for node in G_github.nodes}
    reddit_by_fold = {str(node).casefold(): node for node in G_reddit.nodes}
    # Full graphs, not the shortlist. Scaling only the hubs would line the hubs up again.
    reddit_raw = {node: compute_fingerprint(G_reddit, node, reddit_metrics) for node in G_reddit.nodes}
    github_raw = {node: compute_fingerprint(G_github, node, github_metrics) for node in G_github.nodes}
    reddit_fp, github_fp = normalize_fingerprints(reddit_raw, github_raw)

    confirmed: list[dict] = []
    name_only: list[dict] = []
    seen_names: set[tuple[str, str]] = set()
    for reddit_handle, github_handle in name_candidates:
        reddit_node = reddit_by_fold.get(str(reddit_handle).casefold())
        github_node = github_by_fold.get(str(github_handle).casefold())
        # A handle that is not in its graph has no fingerprint to compare.
        if reddit_node is None or github_node is None:
            continue
        pair = (str(reddit_node).casefold(), str(github_node).casefold())
        if pair in seen_names:
            continue
        seen_names.add(pair)
        similarity = fingerprint_similarity(reddit_fp[reddit_node], github_fp[github_node])
        # Name only changes the reported confidence. The similarity above is structural.
        exact = _name_exact(str(reddit_node), str(github_node))
        row = _row(str(reddit_node), str(github_node), similarity, exact)
        if exact and similarity > _CONFIRMED_SIM:
            confirmed.append(row)
        elif exact:
            name_only.append(row)

    structural: list[dict] = []
    github_nodes = list(G_github.nodes)
    for reddit_node in reddit_nodes:
        if reddit_node not in G_reddit:
            folded = reddit_by_fold.get(str(reddit_node).casefold())
            if folded is None:
                continue
            reddit_node = folded
        scored = []
        for github_node in github_nodes:
            similarity = fingerprint_similarity(reddit_fp[reddit_node], github_fp[github_node])
            scored.append((similarity, github_node))
        # Highest similarity first. The cutoff is applied after the name filter.
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        for similarity, github_node in scored[:_TOP_STRUCTURAL]:
            if _name_exact(str(reddit_node), str(github_node)):
                continue
            if similarity <= _STRUCTURAL_ONLY_SIM:
                continue
            structural.append(_row(str(reddit_node), str(github_node), similarity, False))

    confirmed.sort(key=lambda row: (-row["combined_score"], -row["struct_sim"], row["reddit"]))
    structural.sort(key=lambda row: (-row["struct_sim"], row["reddit"], row["github"]))
    name_only.sort(key=lambda row: (-row["struct_sim"], row["reddit"]))
    return confirmed, structural, name_only
