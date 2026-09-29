"""Analyze a graph from a file or from a run already in the store.

``--source file`` reads CSV, TSV, JSON, or GraphML. ``--load-run`` and ``--run``
read a saved graph.
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

from osi.loader import load_edge_list, load_graphml  # noqa: E402
from osi.analysis import (  # noqa: E402
    adamic_adar,
    betweenness_centrality,
    closeness_centrality,
    compare_centralities,
    compare_communities,
    cpm_communities,
    cross_reference,
    degree_centrality,
    degree_distribution,
    jaccard,
    leiden_communities,
    louvain_communities,
    network_health,
    pagerank,
    preferential_attachment,
    print_cpm_summary,
    rich_club,
    robustness,
)
from osi.store import (  # noqa: E402
    create_run,
    get_run,
    list_results,
    list_runs,
    load_communities,
    load_graph,
    load_metrics,
    load_result,
    save_communities,
    save_graph,
    save_metrics,
    save_result,
)


def _parse_top(value: str) -> int:
    text = value.strip().removeprefix("top=")
    try:
        number = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use top=N") from error
    if number < 1:
        raise argparse.ArgumentTypeError("top must be at least 1")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a graph from a file or from a saved run.")
    parser.add_argument("--source", choices=("file",), default=None)
    parser.add_argument("--path", help="Edge list or GraphML. Used with --source file.")
    parser.add_argument("--analyze", choices=("all", "centrality", "communities"), default="all")
    parser.add_argument(
        "--compare",
        choices=("all", "centrality", "communities", "cross"),
        help="Print a side-by-side comparison. Omit this flag to keep the original report.",
    )
    parser.add_argument("--out", default="graph.json", help="Node-link JSON output path.")
    parser.add_argument("--link-predict", type=_parse_top, metavar="top=N")
    parser.add_argument("--cpm", action="store_true", help="Print overlapping k-clique communities.")
    parser.add_argument("--save-run", metavar="NAME", help="Save results under this run id.")
    parser.add_argument("--run", metavar="NAME", help="Saved run to analyze. Takes precedence over --source.")
    parser.add_argument("--load-run", metavar="NAME", help="Load a previous run instead of recomputing.")
    parser.add_argument("--list-runs", action="store_true", help="Print saved runs and exit.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the store and compute fresh.")
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Remove nodes at random, by degree, and by betweenness, and print what remains.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print network health, degree-distribution fits, and rich-club coefficients.",
    )
    args = parser.parse_args(argv)
    args.robustness_loaded_latest = False
    if args.list_runs:
        return args
    if args.run:
        args.load_run = args.run
    if args.robustness and args.source is None and not args.load_run:
        if args.no_cache:
            parser.error("pass --load-run with --robustness --no-cache")
        saved = list_runs()
        if not saved:
            parser.error("pass --load-run with --robustness, or save a run first. The store is empty.")
        args.load_run = saved[-1][0]
        args.robustness_loaded_latest = True
    if args.load_run and not args.no_cache:
        return args
    if args.source == "file":
        if not args.path:
            parser.error("--source file requires --path")
        return args
    parser.error("pass --load-run, or --run")
    return args


def community_sizes(membership: dict) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for community in membership.values():
        counts[int(community)] = counts.get(int(community), 0) + 1
    return sorted(counts.items())


def print_rank_list(ranks: dict) -> None:
    print("PageRank top 20:")
    for index, (node, score) in enumerate(list(ranks.items())[:20], start=1):
        print(f"{index}. {node} {score:.6f}")


def print_pagerank(graph: nx.Graph) -> dict[str, dict]:
    # Kept and returned so --save-run does not compute betweenness a second time.
    scores = {
        "degree": degree_centrality(graph),
        "pagerank": pagerank(graph),
        "betweenness": betweenness_centrality(graph),
    }
    print_rank_list(scores["pagerank"])
    return scores


def print_membership(name: str, membership: dict) -> None:
    sizes = community_sizes(membership)
    print(f"{name} communities: {len(sizes)}")
    for community, size in sizes:
        print(f"  community {community}: {size}")


def print_communities(graph: nx.Graph) -> dict[str, dict]:
    saved: dict[str, dict] = {}
    for name, algorithm, membership in (
        ("Louvain", "louvain", louvain_communities(graph)),
        ("Leiden", "leiden", leiden_communities(graph)),
    ):
        saved[algorithm] = membership
        print_membership(name, membership)
    return saved


def print_centrality_table(centralities: dict, top_n: int = 10) -> None:
    # The example table shows these four. Eigenvector stays in the data for callers.
    columns = ["degree", "betweenness", "closeness", "pagerank"]
    print("TOP NODES BY EACH CENTRALITY MEASURE:")
    print("Rank | Degree | Betweenness | Closeness | PageRank")
    for rank in range(1, top_n + 1):
        cells = [str(rank)]
        for name in columns:
            ordered = centralities["order"][name]
            cells.append(str(ordered[rank - 1]) if rank <= len(ordered) else "")
        print(" | ".join(cells))
    if centralities.get("eigenvector_error"):
        print(f"Eigenvector centrality failed: {centralities['eigenvector_error']}")


def print_community_table(communities: dict) -> None:
    print("COMMUNITY ALGORITHMS:")
    print("Method | Communities | Modularity")
    for name, label in (("louvain", "Louvain"), ("leiden", "Leiden"), ("girvan_newman", "Girvan-Newman")):
        block = communities[name]
        if block.get("skipped"):
            print(f"{label} | skipped | {block['message']}")
            continue
        print(f"{label} | {block['count']} | {block['modularity']:.6f}")


def print_comparison(graph: nx.Graph, mode: str) -> None:
    # Cross-reference needs both results, so "all" and "cross" compute both.
    need_centrality = mode in ("all", "centrality", "cross")
    need_communities = mode in ("all", "communities", "cross")
    centralities = compare_centralities(graph) if need_centrality else None
    communities = compare_communities(graph) if need_communities else None
    if mode in ("all", "centrality") and centralities is not None:
        print_centrality_table(centralities)
    if mode in ("all", "communities") and communities is not None:
        print_community_table(communities)
    if mode in ("all", "cross") and centralities is not None and communities is not None:
        cross_reference(graph, communities, centralities)


def print_link_predictions(graph: nx.Graph, top_n: int) -> None:
    """Print the strongest predicted links and whether each pair shares a Louvain community."""
    print(f"link-predict before: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    communities = louvain_communities(graph)
    # Score one method at a time. Holding every method's pair list at once
    # does not fit when tens of millions of pairs share a neighbor.
    for name, scorer in (
        ("adamic_adar", adamic_adar),
        ("jaccard", jaccard),
        ("preferential_attachment", preferential_attachment),
    ):
        rows = scorer(graph)
        print(name)
        for left, right, score in rows[:top_n]:
            left_community = communities.get(left)
            right_community = communities.get(right)
            shared = left_community == right_community
            print(
                f"{score:.6f} {left} community {left_community} "
                f"{right} community {right_community} shared={shared}"
            )
    print(f"link-predict after: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def _robust_cell(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _component_cell(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _robust_rows(results: dict):
    yield ("intact", 0.0, results["baseline"])
    for strategy, rows in results.items():
        if strategy == "baseline":
            continue
        for ratio, stats in rows.items():
            yield (strategy, ratio, stats)


def print_robustness(results: dict) -> None:
    print("strategy  ratio  remaining  largest  components  efficiency")
    for strategy, ratio, stats in _robust_rows(results):
        print(
            f"{strategy}  {ratio:.2f}  {int(stats['remaining'])}  "
            f"{_robust_cell(stats['largest'], 6)}  {_component_cell(stats['components'])}  "
            f"{_robust_cell(stats['efficiency'], 6)}"
        )


# The metrics table is one row per node. Health is one JSON document per run.
_HEALTH_SOURCE = "health"
_HEALTH_KEYS = (
    "assortativity",
    "avg_clustering",
    "transitivity",
    "num_components",
    "avg_degree",
    "max_degree",
)
_FIT_NAMES = ("power_law", "lognormal", "exponential")


def _health_cell(value) -> str:
    if value is None:
        return "na"
    if isinstance(value, str):
        return value
    # bool is a subclass of int. It is not a health value, but keep it from printing as 0 or 1.
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _health_payload(graph: nx.Graph) -> dict:
    # JSON object keys are strings. rich_club's keys are the degree thresholds.
    return {
        "network": network_health(graph),
        "degree_distribution": degree_distribution(graph),
        "rich_club": {str(k): value for k, value in rich_club(graph).items()},
    }


def print_health(run_id: str, payload: dict) -> None:
    network = payload["network"]
    print(f"health {run_id}")
    for key in _HEALTH_KEYS:
        print(f"{key} {_health_cell(network.get(key))}")
    for key, value in payload["rich_club"].items():
        print(f"rich_club {key} {_health_cell(value)}")
    distribution = payload["degree_distribution"]
    print(f"best_fit {_health_cell(distribution.get('best_fit'))}")
    for name in _FIT_NAMES:
        fit = distribution.get(name) or {}
        params = fit.get("params") or {}
        pieces = [name]
        for param in sorted(params):
            pieces.append(f"{param} {_health_cell(params[param])}")
        pieces.append(f"r_squared {_health_cell(fit.get('r_squared'))}")
        print(" ".join(pieces))


def print_health_comparison(rows: list[tuple[str, dict]]) -> None:
    """Print one column per run once at least two runs have health stored."""
    if len(rows) < 2:
        return
    print("health compare " + " ".join(run_id for run_id, _payload in rows))

    def line(label: str, getter) -> None:
        print(label + " " + " ".join(_health_cell(getter(payload)) for _run_id, payload in rows))

    for key in _HEALTH_KEYS:
        # The default argument binds this iteration's key. A bare closure would keep the last one.
        line(key, lambda payload, key=key: payload["network"].get(key))
    club_keys: list[str] = []
    for _run_id, payload in rows:
        for key in payload["rich_club"]:
            if key not in club_keys:
                club_keys.append(key)
    for key in club_keys:
        line(f"rich_club_{key}", lambda payload, key=key: payload["rich_club"].get(key))
    line("best_fit", lambda payload: payload["degree_distribution"].get("best_fit"))
    for name in _FIT_NAMES:
        line(
            f"{name}_r2",
            lambda payload, name=name: (payload["degree_distribution"].get(name) or {}).get("r_squared"),
        )


def _emit_health(run_id: str | None, graph: nx.Graph) -> None:
    # A repeat call for the same run reads the stored document instead of refitting.
    payload = load_result(run_id, _HEALTH_SOURCE) if run_id else None
    if payload is None:
        payload = _health_payload(graph)
        if run_id:
            save_result(run_id, _HEALTH_SOURCE, payload)
    print_health(run_id or "graph", payload)
    # Other runs may already have a health document. Two or more is the comparison table.
    stored = list_results(_HEALTH_SOURCE)
    if run_id and all(saved_id != run_id for saved_id, _payload in stored):
        stored.append((run_id, payload))
    print_health_comparison(stored)


def print_robustness_pair(label_a: str, left: dict, label_b: str, right: dict) -> None:
    print(f"a {label_a}")
    print(f"b {label_b}")
    print("strategy  ratio  rem_a  lcc_a  comp_a  eff_a  rem_b  lcc_b  comp_b  eff_b")
    rows_b = {(strategy, ratio): stats for strategy, ratio, stats in _robust_rows(right)}
    for strategy, ratio, stats in _robust_rows(left):
        other = rows_b[(strategy, ratio)]
        print(
            f"{strategy}  {ratio:.2f}  {int(stats['remaining'])}  {_robust_cell(stats['largest'], 6)}  "
            f"{_component_cell(stats['components'])}  {_robust_cell(stats['efficiency'], 6)}  "
            f"{int(other['remaining'])}  {_robust_cell(other['largest'], 6)}  "
            f"{_component_cell(other['components'])}  {_robust_cell(other['efficiency'], 6)}"
        )


def write_graph(graph: nx.Graph, path: str) -> None:
    payload = nx.node_link_data(graph, edges="links")
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def graph_layer(args: argparse.Namespace) -> str:
    if args.source == "file":
        return "file"
    return "graph"


def _print_saved_runs() -> None:
    for run_id, source, created_at, notes in list_runs():
        print(f"{run_id} {source} {created_at} {notes}")


def _load_saved_graph(run_id: str) -> nx.Graph:
    meta = get_run(run_id)
    if meta is None:
        raise LookupError(f"run {run_id} not found")
    # The layer is recorded with the run so a later command can omit --source.
    layer = (meta.get("config") or {}).get("layer")
    if not layer:
        raise LookupError(f"run {run_id} has no layer")
    graph = load_graph(run_id, layer)
    if graph is None:
        raise LookupError(f"run {run_id} has no graph for {layer}")
    return graph


def _report_from_store(args: argparse.Namespace, graph: nx.Graph) -> None:
    """Print the saved report. Missing pieces are computed from the loaded graph."""
    if args.analyze in ("all", "centrality"):
        ranks = load_metrics(args.load_run, "pagerank")
        if ranks:
            print_rank_list(ranks)
        else:
            print_pagerank(graph)
    if args.analyze in ("all", "communities"):
        louvain = load_communities(args.load_run, "louvain")
        leiden = load_communities(args.load_run, "leiden")
        if louvain and leiden:
            print_membership("Louvain", louvain)
            print_membership("Leiden", leiden)
        else:
            print_communities(graph)


def _eigenvector_scores(graph: nx.Graph) -> dict | None:
    # A disconnected graph has no single leading eigenvector. Skip it instead of failing the save.
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        print("warning: eigenvector skipped: graph is disconnected", file=sys.stderr)
        return None
    try:
        raw = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception as error:
        print(f"warning: eigenvector skipped: {type(error).__name__}: {error}", file=sys.stderr)
        return None
    return dict(sorted(raw.items(), key=lambda item: (-float(item[1]), str(item[0]))))


def _persist_run(args: argparse.Namespace, graph: nx.Graph, centralities: dict | None, communities: dict | None) -> None:
    layer = graph_layer(args)
    source = args.source or "file"
    config = {"source": source, "analyze": args.analyze, "layer": layer, "path": args.path}
    run_id = create_run(source, config, args.save_run, run_id=args.save_run)
    save_graph(run_id, layer, graph)
    scores = dict(centralities or {})
    if "degree" not in scores:
        scores["degree"] = degree_centrality(graph)
    if "pagerank" not in scores:
        scores["pagerank"] = pagerank(graph)
    if "betweenness" not in scores:
        scores["betweenness"] = betweenness_centrality(graph)
    if "closeness" not in scores:
        scores["closeness"] = closeness_centrality(graph)
    eigenvector = _eigenvector_scores(graph)
    if eigenvector is not None:
        scores["eigenvector"] = eigenvector
    for metric, values in scores.items():
        save_metrics(run_id, values, metric=metric)
    for algorithm, membership in (communities or {}).items():
        save_communities(run_id, algorithm, membership)


def _load_source_file(path: str) -> nx.Graph:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".tsv", ".json"}:
        return load_edge_list(path)
    if suffix == ".graphml":
        return load_graphml(path)
    raise ValueError(f"unsupported graph file {suffix or '(no extension)'}; use .csv, .tsv, .json, or .graphml")


def _analyze_fresh(args: argparse.Namespace, graph: nx.Graph) -> None:
    """Print the requested report, then store the graph when --save-run is set."""
    centralities = print_pagerank(graph) if args.analyze in ("all", "centrality") else None
    communities = print_communities(graph) if args.analyze in ("all", "communities") else None
    if not args.save_run:
        return
    _persist_run(args, graph, centralities, communities)
    args.load_run = args.save_run
    print(f"saved run {args.save_run}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.robustness_loaded_latest:
        print(f"using saved run {args.load_run}")
    if args.list_runs:
        _print_saved_runs()
        return 0
    # --run and --load-run win unless --no-cache asked for a fresh file read.
    reading_file = args.source == "file" and not (args.load_run and not args.no_cache)
    if reading_file:
        try:
            graph = _load_source_file(args.path)
        except (OSError, ValueError, json.JSONDecodeError, nx.NetworkXError) as error:
            print(error, file=sys.stderr)
            return 1
        _analyze_fresh(args, graph)
    else:
        try:
            graph = _load_saved_graph(args.load_run)
        except (LookupError, json.JSONDecodeError, OSError, ValueError) as error:
            print(f"run {args.load_run} was not found", file=sys.stderr)
            print(error, file=sys.stderr)
            return 1
        _report_from_store(args, graph)
    if args.robustness:
        degree_scores = load_metrics(args.load_run, "degree") if args.load_run else None
        between_scores = load_metrics(args.load_run, "betweenness") if args.load_run else None
        print_robustness(
            robustness(
                graph,
                degree_scores=degree_scores or None,
                betweenness_scores=between_scores or None,
            )
        )
    if args.health:
        _emit_health(args.load_run or args.save_run, graph)
    if args.compare:
        print_comparison(graph, args.compare)
    write_graph(graph, args.out)
    if args.link_predict:
        print_link_predictions(graph, args.link_predict)
    if args.cpm:
        print(f"cpm before: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print_cpm_summary(cpm_communities(graph))
        print(f"cpm after: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
