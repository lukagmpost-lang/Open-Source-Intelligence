"""Build a public graph, analyze it, and write node-link JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.analysis import (  # noqa: E402
    adamic_adar,
    betweenness_centrality,
    compare_centralities,
    compare_communities,
    closeness_centrality,
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
from viz.interactive import to_interactive_html  # noqa: E402
from layers.reddit_archive import load_reddit_layers  # noqa: E402
from osi.datasets import load_snap_facebook  # noqa: E402
from osi.graph import fetch_github_graph  # noqa: E402
from osi.layers.bluesky import build_bluesky_layer  # noqa: E402
from osi.identity import (  # noqa: E402
    annotate_coverage,
    fetch_github_identities,
    load_identity_map,
    merge_identity_layers,
    normalize_layers,
    warn_missing_handles,
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
    parser = argparse.ArgumentParser(description="Build a public social graph and analyze it.")
    parser.add_argument(
        "--source",
        choices=("github", "snap_facebook", "reddit", "reddit_2012"),
        default=None,
    )
    parser.add_argument("--username", help="Public GitHub login. Required when --source is github.")
    parser.add_argument("--analyze", choices=("all", "centrality", "communities"), default="all")
    parser.add_argument(
        "--compare",
        choices=("all", "centrality", "communities", "cross"),
        help="Print a side-by-side comparison. Omit this flag to keep the original report.",
    )
    parser.add_argument("--out", default="graph.json", help="Node-link JSON output path.")
    # Draw graph.png after the text report. Off unless this flag is present.
    parser.add_argument("--plot", action="store_true", help="Save graph.png after the analysis.")
    # Value is top=N, for example --link-predict top=10.
    parser.add_argument("--link-predict", type=_parse_top, metavar="top=N")
    parser.add_argument("--cpm", action="store_true", help="Print overlapping k-clique communities.")
    parser.add_argument("--interactive", action="store_true", help="Save graph.html.")
    # Interactive HTML only. The analysis above still uses the full graph.
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Interactive node cap, applied after light edges are removed.",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        default=2,
        help="Drop interactive edges lighter than this before the node cap.",
    )
    # The store is optional. Omitting these flags keeps the old one-shot run.
    parser.add_argument("--save-run", metavar="NAME", help="Save results under this run id.")
    parser.add_argument("--run", metavar="NAME", help="Saved run to analyze. Takes precedence over --source.")
    parser.add_argument("--load-run", metavar="NAME", help="Load a previous run instead of recomputing.")
    parser.add_argument("--list-runs", action="store_true", help="Print saved runs and exit.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the store and compute fresh.")
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Remove nodes at random, by degree, and by betweenness, and print what remains. With no --source and no --run, the newest saved run is used.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print network health, degree-distribution fits, and rich-club coefficients. Stored under the source key health.",
    )
    parser.add_argument("--identity", metavar="path.json", help="JSON map from person id to platform handles.")
    parser.add_argument("--layers", help="Comma-separated layer names. Used only with --identity.")
    parser.add_argument(
        "--normalize-layers",
        action="store_true",
        help="Scale each layer's edge weights to the same total before the supra-graph is built.",
    )
    args = parser.parse_args(argv)
    args.robustness_loaded_latest = False
    if args.list_runs:
        return args
    # A named run is the graph to analyze, even when --source is also present.
    if args.run:
        args.load_run = args.run
    # --robustness with no source and no named run uses the store. GitHub is not assumed.
    if args.robustness and args.source is None and not args.load_run and not args.identity:
        if args.no_cache:
            parser.error("pass --source with --robustness --no-cache")
        saved = list_runs()
        if not saved:
            parser.error("pass --source with --robustness, or save a run first. The store is empty.")
        # list_runs is oldest first, so the last row is the newest save.
        args.load_run = saved[-1][0]
        args.robustness_loaded_latest = True
    # Loading does not build a GitHub graph, so a login is not required yet.
    if args.load_run and not args.no_cache:
        return args
    if args.source is None:
        args.source = "github"
    # Identity loads each named layer on its own, so a GitHub login is not required up front.
    if args.identity:
        if args.max_nodes < 1:
            parser.error("--max-nodes must be at least 1")
        return args
    if args.source == "github" and not args.username:
        parser.error("--username is required when --source is github")
    if args.max_nodes < 1:
        parser.error("--max-nodes must be at least 1")
    return args


def _requested_layers(args: argparse.Namespace) -> list[str]:
    if args.layers:
        return [part.strip() for part in args.layers.split(",") if part.strip()]
    # The flag is documented with these two layers when the caller does not list any.
    return ["github", "reddit"]


def _load_named_layer(name: str, args: argparse.Namespace) -> nx.Graph | None:
    """Load one layer. Any failure returns None so the rest of the merge can continue."""
    try:
        if name == "github":
            # Identity mode fetches every mapped login, not the single --username ego graph.
            if args.identity:
                return fetch_github_identities(load_identity_map(args.identity))
            if not args.username:
                return None
            return fetch_github_graph(args.username)
        if name == "reddit":
            if args.identity:
                # Read the saved January 2008 user layer. Do not rebuild it and do not write that run.
                stored = load_graph("r2008-v2", "reddit_user")
                if stored is not None:
                    return stored
            return load_reddit_layers()["reddit_user"]
        if name == "reddit_2012":
            return load_reddit_layers(
                "2012-08",
                user_layer="reddit_2012_user",
                subreddit_layer="reddit_2012_subreddit",
            )["reddit_2012_user"]
        if name == "snap_facebook":
            return load_snap_facebook()
        if name == "bluesky":
            # Public AppView. Only identity-map handles are fetched, and no token is sent.
            if not args.identity:
                return None
            handles = [
                str(accounts["bluesky"])
                for accounts in load_identity_map(args.identity).values()
                if accounts.get("bluesky")
            ]
            return build_bluesky_layer(handles)
    except Exception as error:  # noqa: BLE001 - a bad layer must not abort the merge
        print(f"warning: layer {name} not loaded: {error}", file=sys.stderr)
        return None
    return None


def _graph_from_identity(args: argparse.Namespace) -> nx.Graph:
    identity_map = load_identity_map(args.identity)
    layers: dict[str, nx.Graph | None] = {}
    for name in _requested_layers(args):
        graph = _load_named_layer(name, args)
        if graph is None:
            print(f"warning: layer {name} not loaded", file=sys.stderr)
            continue
        layers[name] = graph
    warn_missing_handles(layers, identity_map)
    if args.normalize_layers:
        # Scale the layers first. merge_identity_layers then calls the existing supra-graph builder.
        layers = normalize_layers(layers)
    merged = merge_identity_layers(layers, identity_map)
    annotate_coverage(merged)
    return merged


def print_identity_summary(graph: nx.Graph) -> None:
    """Print how many mapped people landed on both platforms, plus the supra-graph size."""
    coverage = graph.graph.get("coverage") or {}
    # "both" lists person ids, so its length is the cross-platform count.
    print(f"cross-platform persons {len(coverage.get('both') or [])}")
    print(f"nodes {graph.number_of_nodes()}")
    interlayer = sum(1 for _left, _right, data in graph.edges(data=True) if data.get("kind") == "interlayer")
    print(f"interlayer edges {interlayer}")


def print_identity_membership(graph: nx.Graph, ranks: dict) -> None:
    """Print which layer each top PageRank node came from."""
    print("PageRank top 20 layer membership:")
    for index, (node, score) in enumerate(list(ranks.items())[:20], start=1):
        data = graph.nodes[node] if node in graph else {}
        print(f"{index}. {node} {score:.6f} layer {data.get('layer')} person {data.get('person')}")


def build_graph(args: argparse.Namespace) -> nx.Graph:
    if args.source == "snap_facebook":
        return load_snap_facebook()
    if args.source == "reddit":
        # The user co-participation layer is the graph the report and the plot use.
        return load_reddit_layers()["reddit_user"]
    if args.source == "reddit_2012":
        # August 2012 keeps its own layer names, so the 2008 graph is left as it is.
        return load_reddit_layers(
            "2012-08",
            user_layer="reddit_2012_user",
            subreddit_layer="reddit_2012_subreddit",
        )["reddit_2012_user"]
    return fetch_github_graph(args.username)


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


def save_plot(graph: nx.Graph, path: str = "graph.png") -> None:
    # PageRank sets the dot size and decides which few nodes get a name.
    scores = pagerank(graph)
    # Louvain community ids are the color key. tab20 cycles every 20 groups.
    communities = louvain_communities(graph)
    # Betweenness order is strongest first, so the first five are the bridges.
    bridges = list(betweenness_centrality(graph))[:5]

    # Draw without a window. Agg writes a file on a machine with no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same seed and spring settings every run, so the picture does not jump around.
    positions = nx.spring_layout(graph, k=0.15, iterations=50, seed=42)
    figure, axes = plt.subplots(figsize=(20, 20))

    # Scale scores into pixel areas. 20000 makes hubs large and leaves the rest small.
    node_size = [scores.get(node, 0.0) * 20000 for node in graph.nodes]
    node_color = [communities.get(node, 0) for node in graph.nodes]
    # Faint edges keep the communities visible without a solid gray sheet.
    nx.draw_networkx_edges(graph, positions, ax=axes, alpha=0.15, width=0.3)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.tab20,
        linewidths=0,
    )

    # A red ring on the top 5 betweenness nodes, using the same size as the fill.
    if bridges:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=bridges,
            ax=axes,
            node_size=[scores.get(node, 0.0) * 20000 for node in bridges],
            node_color="none",
            edgecolors="red",
            linewidths=2,
        )

    # Names only for the handful of nodes above 0.003, about the top five on SNAP.
    labels = {node: str(node) for node, score in scores.items() if score > 0.003}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axes, font_size=10)
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("Saved graph.png")


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
    if args.identity:
        return "supra"
    if args.source == "reddit":
        return "reddit_user"
    if args.source == "reddit_2012":
        return "reddit_2012_user"
    if args.source == "snap_facebook":
        return "snap_facebook"
    return "github"


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
    source = "identity" if args.identity else args.source
    config = {
        "source": source,
        "username": args.username,
        "analyze": args.analyze,
        "layer": layer,
        "identity": args.identity,
        "layers": args.layers,
        "normalize_layers": bool(args.normalize_layers),
    }
    # The flag value is the primary key, so a second save with the same name replaces it.
    run_id = create_run(source, config, args.save_run, run_id=args.save_run)
    save_graph(run_id, layer, graph)
    # Always store the standard metrics, including ones this report did not print.
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
    if args.identity:
        # Person ids already classified on the graph: both platforms, or one side only.
        save_result(run_id, "identity", graph.graph.get("coverage") or {})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.robustness_loaded_latest:
        print(f"using saved run {args.load_run}")
    if args.list_runs:
        _print_saved_runs()
        return 0
    loaded = False
    if args.load_run and not args.no_cache:
        try:
            graph = _load_saved_graph(args.load_run)
            loaded = True
        except (LookupError, json.JSONDecodeError, OSError, ValueError) as error:
            # --run names one saved graph. Do not silently rebuild a different source.
            if args.run:
                print(f"run {args.run} was not found", file=sys.stderr)
                return 1
            # A bad or missing --load-run must not stop the command. Rebuild from the source.
            print(f"warning: could not load run {args.load_run}: {error}; computing fresh", file=sys.stderr)
            if args.source == "github" and not args.username:
                raise SystemExit("--username is required when --source is github") from error
            graph = build_graph(args)
    elif args.identity:
        graph = _graph_from_identity(args)
    else:
        graph = build_graph(args)
    centralities = None
    communities = None
    if args.robustness:
        # Saved centralities are the intact-graph ranking. Recomputing betweenness here would repeat that work.
        degree_scores = load_metrics(args.load_run, "degree") if loaded else None
        between_scores = load_metrics(args.load_run, "betweenness") if loaded else None
        print_robustness(
            robustness(
                graph,
                degree_scores=degree_scores or None,
                betweenness_scores=between_scores or None,
            )
        )
    if loaded:
        _report_from_store(args, graph)
    else:
        if args.analyze in ("all", "centrality"):
            centralities = print_pagerank(graph)
        if args.analyze in ("all", "communities"):
            communities = print_communities(graph)
    if args.identity:
        # Communities-only runs have not ranked nodes yet. PageRank is computed here for the membership list.
        scores = (centralities or {}).get("pagerank") if centralities else None
        print_identity_membership(graph, scores or pagerank(graph))
    if args.health:
        # Loaded runs keep their id. A fresh build can still store health when --save-run is set.
        _emit_health(args.load_run or args.save_run, graph)
    if args.compare:
        print_comparison(graph, args.compare)
    if args.save_run and not args.no_cache and not loaded:
        _persist_run(args, graph, centralities, communities)
    write_graph(graph, args.out)
    # Plot last so "Saved graph.png" is the final line of the run.
    if args.plot:
        save_plot(graph)
    if args.link_predict:
        print_link_predictions(graph, args.link_predict)
    if args.cpm:
        print(f"cpm before: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print_cpm_summary(cpm_communities(graph))
        print(f"cpm after: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    if args.interactive:
        to_interactive_html(
            graph,
            max_nodes=args.max_nodes,
            min_edge_weight=args.min_edge_weight,
        )
    if args.identity:
        print_identity_summary(graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
