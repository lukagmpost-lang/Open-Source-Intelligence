"""Match Reddit users to a GitHub ego graph by name and by structure.

Usage:
  python3 suggest_identity_v2.py --reddit-run r2008-v2 --github-user rtomayko [--top 100]
  python3 suggest_identity_v2.py --reddit-run r2008-v2 --github-graph github_multi_ego.graphml [--top 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.analysis import betweenness_centrality, closeness_centrality, louvain_communities, pagerank
from osi.github_graph import MULTI_EGO_LAYER, MULTI_EGO_RUN, fetch_github_graph
from osi.identity_structural import match_by_structure
from osi.store import get_run, list_runs, load_communities, load_graph, load_metrics

_OUTPUT = ROOT / "identity_v4.json"
# Saved runs that already contain a fetched GitHub ego. Read only.
_SAVED_EGO_RUNS = ("cross-v2", "cross-v1")


def _login_of(node, data: dict) -> str:
    label = data.get("label")
    if label:
        return str(label)
    account = data.get("account")
    # The supra-graph overwrites the center's account with its own node id.
    if account and "|" not in str(account):
        return str(account)
    text = str(node)
    if text.startswith("github:"):
        return text.split(":", 1)[1]
    return text


def _ego_from_saved(login: str) -> nx.Graph | None:
    """Pull one already-fetched GitHub neighborhood out of a saved run."""
    folded = login.casefold()
    for run_id in _SAVED_EGO_RUNS:
        meta = get_run(run_id)
        if meta is None:
            continue
        layer = (meta.get("config") or {}).get("layer")
        graph = load_graph(run_id, layer) if layer else None
        if graph is None:
            continue
        center = None
        for node, data in graph.nodes(data=True):
            if data.get("layer") != "github":
                continue
            if _login_of(node, data).casefold() != folded:
                continue
            # Prefer the account the fetch was centered on, not a later mention.
            if data.get("center") or center is None:
                center = node
                if data.get("center"):
                    break
        if center is None:
            continue
        chosen = {center}
        for neighbor in graph.neighbors(center):
            if graph.nodes[neighbor].get("layer") == "github":
                chosen.add(neighbor)
        ego = nx.Graph()
        names = {node: _login_of(node, graph.nodes[node]) for node in chosen}
        ego.add_nodes_from(names.values())
        for left, right, data in graph.edges(data=True):
            if left not in names or right not in names:
                continue
            ego.add_edge(names[left], names[right], weight=float(data.get("weight", 1.0)))
        if ego.number_of_nodes():
            return ego
    return None


def load_github_graph(logins: list[str]) -> nx.Graph:
    """One ego network per login. A saved layer is used before a new API call."""
    combined = nx.Graph()
    for login in logins:
        ego = _ego_from_saved(login)
        if ego is None:
            ego = fetch_github_graph(login)
        # A login can be the center of one ego and a neighbor in another.
        combined = nx.compose(combined, ego)
    return combined


def _metric_bundle(graph: nx.Graph, run_id: str | None) -> dict:
    """Stored scores when the run has them. Otherwise measure this graph."""
    metrics: dict = {}
    nodes = set(graph.nodes)
    if run_id:
        for name in ("pagerank", "betweenness", "closeness"):
            loaded = load_metrics(run_id, name)
            # A score dict from an older, smaller graph cannot fingerprint these nodes.
            if loaded and nodes <= set(loaded):
                metrics[name] = loaded
        community = load_communities(run_id, "louvain")
        if community and nodes <= set(community):
            metrics["community"] = community
    if "pagerank" not in metrics:
        metrics["pagerank"] = pagerank(graph)
    if "betweenness" not in metrics:
        metrics["betweenness"] = betweenness_centrality(graph)
    if "closeness" not in metrics:
        metrics["closeness"] = closeness_centrality(graph)
    if "community" not in metrics:
        metrics["community"] = louvain_communities(graph)
    return metrics


def _name_candidates(reddit_nodes: list, github: nx.Graph) -> list[tuple[str, str]]:
    github_by_fold = {str(node).casefold(): str(node) for node in github.nodes}
    pairs = []
    for node in reddit_nodes:
        github_node = github_by_fold.get(str(node).casefold())
        if github_node is not None:
            pairs.append((str(node), github_node))
    return pairs


def _print_table(title: str, rows: list[dict]) -> None:
    print(title)
    headers = ("reddit_handle", "github_handle", "struct_sim", "combined")
    rendered = [
        (row["reddit"], row["github"], f"{row['struct_sim']:.4f}", f"{row['combined_score']:.4f}")
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for cells in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for cells in rendered:
        print("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))


def _confidence(row: dict, kind: str) -> str:
    if kind == "confirmed":
        return "high"
    if kind == "structural":
        return "medium"
    return "low"


def _write(confirmed: list[dict], structural: list[dict], name_only: list[dict]) -> None:
    payload: dict[str, dict] = {}
    for kind, rows in (("confirmed", confirmed), ("name_only", name_only), ("structural", structural)):
        for row in rows:
            # Name rows keep the Reddit handle. A structural pair must not overwrite that row.
            person_id = row["reddit"] if kind != "structural" else f"{row['reddit']}::{row['github']}"
            payload[person_id] = {
                "github": row["github"],
                "reddit": row["reddit"],
                "confidence": _confidence(row, kind),
                "struct_sim": round(float(row["struct_sim"]), 4),
            }
    _OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_github_graphml(path: Path) -> nx.Graph:
    """Read a saved multi-ego layer. GraphML stores every attribute as text."""
    graph = nx.read_graphml(path)
    for _left, _right, data in graph.edges(data=True):
        if "weight" in data:
            data["weight"] = float(data["weight"])
    return graph


def _metrics_for_github_graph(graph: nx.Graph) -> dict:
    """Use the cached multi-ego scores when this file is that same node set."""
    stored = load_graph(MULTI_EGO_RUN, MULTI_EGO_LAYER)
    if stored is not None and {str(node) for node in stored.nodes} == {str(node) for node in graph.nodes}:
        return _metric_bundle(graph, MULTI_EGO_RUN)
    return _metric_bundle(graph, None)


def suggest(
    reddit_run: str,
    github_users: list[str] | None,
    top: int,
    github_graph: str | None = None,
) -> int:
    meta = get_run(reddit_run)
    if meta is None:
        for saved_id, source, _created_at, _notes in list_runs():
            print(f"{saved_id} {source}")
        return 1
    layer = (meta.get("config") or {}).get("layer")
    reddit = load_graph(reddit_run, layer) if layer else None
    if reddit is None:
        for saved_id, source, _created_at, _notes in list_runs():
            print(f"{saved_id} {source}")
        return 1
    if github_graph:
        path = Path(github_graph)
        if not path.is_file():
            print(f"missing GitHub graph {path}", file=sys.stderr)
            return 1
        github = load_github_graphml(path)
        # A multi-ego file already has edges between neighborhoods. Do not warn about a star.
        github_metrics = _metrics_for_github_graph(github)
    else:
        github = load_github_graph(github_users or [])
        if len(github_users or []) == 1:
            # One fetched neighborhood has no edges except the center's. Scores cannot see the rest of GitHub.
            print(
                f"note: GitHub graph is one ego network around {github_users[0]} "
                f"({github.number_of_nodes()} nodes). Fingerprints only see that local structure.",
                file=sys.stderr,
            )
        github_metrics = _metric_bundle(github, None)
    reddit_metrics = _metric_bundle(reddit, reddit_run)
    # Stored PageRank is already strongest-first, which is the order --top cuts.
    ranked = [node for node in reddit_metrics["pagerank"] if node in reddit][:top]
    candidates = _name_candidates(ranked, github)
    confirmed, structural, name_only = match_by_structure(
        github,
        reddit,
        candidates,
        reddit_nodes=ranked,
        github_metrics=github_metrics,
        reddit_metrics=reddit_metrics,
    )
    _print_table("CONFIRMED", confirmed)
    _print_table("STRUCTURAL_ONLY", structural)
    _print_table("NAME_ONLY", name_only)
    _write(confirmed, structural, name_only)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match Reddit users to a GitHub ego graph by structure.")
    parser.add_argument("--reddit-run", required=True)
    parser.add_argument("--github-user", action="append")
    parser.add_argument("--github-graph")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    # One source for the GitHub layer. The file is the multi-ego graph; the flag is a single login.
    if bool(args.github_user) == bool(args.github_graph):
        parser.error("pass exactly one of --github-user or --github-graph")
    return suggest(args.reddit_run, args.github_user, args.top, args.github_graph)


if __name__ == "__main__":
    raise SystemExit(main())
