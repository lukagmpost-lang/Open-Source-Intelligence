"""Six evenly spaced Reddit months, then one joint multislice at omega 0.5.

The months step 16 months from 2005-12 through 2012-08. Graphs are saved as
new runs (reddit-hy-YYYY-MM). Existing runs are not opened for writing.

Leiden uses leidenalg.find_partition_temporal. A parent process reads the
child's VmRSS and kills it above 5.5 GiB. If that peak also passes 5 GiB,
the same graphs are retried as four slices.

Run:
    python3 scripts/reddit_halfyear_multislice.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# 2005-12 is only the subreddit reddit.com, so the eight-subreddit filter is empty there.
# 2006-05 is the earliest month in the even grid that still has those subreddits.
# August 2012 is the last month. Five gaps of 15 months land on these six.
MONTHS = (
    (2006, 5),
    (2007, 8),
    (2008, 11),
    (2010, 2),
    (2011, 5),
    (2012, 8),
)
OMEGA = 0.5
LAYER = "reddit_user"
# VmRSS is KiB. These caps are GiB, matching free -h, not decimal GB.
KILL_RSS_KIB = int(5.5 * 1024 * 1024)
RETRY_RSS_KIB = int(5 * 1024 * 1024)
# Runs already in the store. A slice id must never be one of these.
PROTECTED_RUNS = {
    "r2008-v1",
    "r2012-v2",
    "r2008-v2",
    "cross-v1",
    "cross-v2",
    "github-multi-ego",
    "cross-v4",
    "bluesky-v1",
    "github-v1",
    "snap-v1",
    "github-organic-v1",
    "bluesky-organic-v1",
}
SUMMARY_PATH = ROOT / "results" / "reddit_halfyear_multislice.json"
_CHILD_RESULT = Path("/tmp/reddit-hy-child-result.json")


def month_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def run_id_for(year: int, month: int) -> str:
    run_id = f"reddit-hy-{month_label(year, month)}"
    if run_id in PROTECTED_RUNS:
        raise RuntimeError(f"{run_id} would overwrite an existing run")
    return run_id


def reduce_to_four(months: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep four months, including the first and the last."""
    chosen = list(months)
    if len(chosen) <= 4:
        return chosen
    last = len(chosen) - 1
    # Three equal index steps across the list, rounded, so the ends stay put.
    indexes = []
    for step in range(4):
        index = round(step * last / 3)
        if index not in indexes:
            indexes.append(index)
    return [chosen[index] for index in indexes]


def _rss_kib(pid: int) -> int:
    """Current resident set from /proc, or 0 when the process has already exited."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _mucha_modularity(graphs, assignments, omega: float) -> float:
    """Same expression as networkx_temporal.modularity_multislice, without that package.

    Each slice contributes its Newman modularity scaled by twice its edge weight.
    Consecutive slices add omega once per account that keeps its community id.
    """
    import networkx as nx

    intra = 0.0
    weight_sum = 0.0
    for graph, assignment in zip(graphs, assignments):
        grouped: dict[int, set] = {}
        for node, community in assignment.items():
            grouped.setdefault(int(community), set()).add(node)
        groups = list(grouped.values())
        size = float(graph.size(weight="weight"))
        weight_sum += size
        if size == 0 or not groups:
            continue
        intra += float(nx.community.modularity(graph, groups, weight="weight")) * 2.0 * size
    interslice_edges = 0
    coupling = 0
    for earlier, later, left, right in zip(graphs, graphs[1:], assignments, assignments[1:]):
        shared = set(earlier.nodes) & set(later.nodes)
        interslice_edges += len(shared)
        coupling += sum(1 for node in shared if int(left[node]) == int(right[node]))
    # mu counts each intraslice edge once and each coupling once. The 2*mu denominator is Mucha's.
    mu = weight_sum + float(omega) * interslice_edges
    if mu == 0.0:
        return 0.0
    return (intra + float(omega) * coupling) / (2.0 * mu)


def _load_saved(year: int, month: int):
    from osi.store import get_run, load_graph

    run_id = run_id_for(year, month)
    if get_run(run_id) is None:
        return None
    graph = load_graph(run_id, LAYER)
    if graph is None or graph.number_of_nodes() == 0:
        return None
    return graph


def _build_one(year: int, month: int):
    """Download one month with the DuckDB loader and apply the Reddit thread cap."""
    from layers.reddit_archive import DEFAULT_SUBREDDITS, user_co_participation
    from osi.loaders.reddit_duckdb import load_month
    from osi.store import create_run, save_graph

    saved = _load_saved(year, month)
    if saved is not None:
        print(
            f"reuse {month_label(year, month)} nodes={saved.number_of_nodes()} edges={saved.number_of_edges()}",
            file=sys.stderr,
            flush=True,
        )
        return saved
    print(f"loading {month_label(year, month)}", file=sys.stderr, flush=True)
    frame = load_month(year, month, list(DEFAULT_SUBREDDITS))
    # The archive builder reads link_id. The DuckDB loader renamed that column to thread_id.
    frame = frame.rename(columns={"thread_id": "link_id"})
    # The builder prints the size. Keep that off stdout so the result file stays the only output.
    with contextlib.redirect_stdout(sys.stderr):
        graph = user_co_participation(frame, LAYER)
    del frame
    gc.collect()
    run_id = run_id_for(year, month)
    create_run(
        "reddit_halfyear",
        {
            "layer": LAYER,
            "year": year,
            "month": month,
            "subreddits": list(DEFAULT_SUBREDDITS),
            "thread_cap": 30,
        },
        notes=run_id,
        run_id=run_id,
    )
    save_graph(run_id, LAYER, graph)
    print(
        f"saved {run_id} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} rss_kib={_rss_kib(os.getpid())}",
        file=sys.stderr,
        flush=True,
    )
    return graph


def _leiden(graphs, omega: float):
    """Joint Leiden. Call find_partition_temporal directly so networkx_temporal is not imported."""
    import igraph as ig
    import leidenalg

    from osi.temporal_communities import _assignments_from_membership, _with_weights

    layers = []
    for graph in graphs:
        prepared = _with_weights(graph)
        layer = ig.Graph.from_networkx(prepared)
        # from_networkx keeps the account name on _nx_name. Coupling matches on that id.
        layer.vs["id"] = layer.vs["_nx_name"]
        if "weight" not in layer.es.attributes() and layer.ecount():
            layer.es["weight"] = 1.0
        layers.append(layer)
    print(f"leiden start slices={len(layers)} rss_kib={_rss_kib(os.getpid())}", file=sys.stderr, flush=True)
    membership, _improvement = leidenalg.find_partition_temporal(
        layers,
        leidenalg.ModularityVertexPartition,
        interslice_weight=float(omega),
        vertex_id_attr="id",
        weight_attr="weight",
        # Negative iterations run until a pass stops improving. seed=0 matches the other Leiden calls.
        n_iterations=-1,
        seed=0,
    )
    print(f"leiden done rss_kib={_rss_kib(os.getpid())}", file=sys.stderr, flush=True)
    return _assignments_from_membership(graphs, membership)


def _child(months: list[tuple[int, int]], phase: str) -> int:
    graphs = []
    built = []
    for year, month in months:
        graph = _build_one(year, month)
        if phase == "build":
            # One month at a time. Holding every graph here would add their RSS together.
            built.append(
                {
                    "month": month_label(year, month),
                    "run_id": run_id_for(year, month),
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                }
            )
            del graph
            gc.collect()
        else:
            graphs.append(graph)
    if phase == "build":
        _CHILD_RESULT.write_text(
            json.dumps(
                {
                    "phase": "build",
                    "months": [month_label(year, month) for year, month in months],
                    "snapshots": built,
                }
            ),
            encoding="utf-8",
        )
        return 0
    assignments = _leiden(graphs, OMEGA)
    modularity = _mucha_modularity(graphs, assignments, OMEGA)
    # Assignments are written by the parent after it accepts this slice count.
    # A run that is over the 5 GiB line must not replace a later four-slice save.
    from osi.temporal_communities import community_persistence, mean_persistence

    snapshots = []
    transitions = []
    assignment_payload = {}
    for (year, month), graph, assignment in zip(months, graphs, assignments):
        label = month_label(year, month)
        run_id = run_id_for(year, month)
        snapshots.append(
            {
                "month": label,
                "run_id": run_id,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "communities": len(set(assignment.values())),
            }
        )
        # JSON keys must be strings. Community ids stay ints.
        assignment_payload[label] = {str(node): int(community) for node, community in assignment.items()}
    for index in range(len(assignments) - 1):
        rows = community_persistence(assignments[index], assignments[index + 1])
        shared = sum(row[2] for row in rows)
        transitions.append(
            {
                "earlier": month_label(*months[index]),
                "later": month_label(*months[index + 1]),
                "mean_persistence": mean_persistence(rows),
                "shared_members": shared,
                "communities": len(rows),
                "rows": [
                    {"community": community, "size": size, "shared": shared_count, "fraction": fraction}
                    for community, size, shared_count, fraction in rows
                ],
            }
        )
    _CHILD_RESULT.write_text(
        json.dumps(
            {
                "phase": "leiden",
                "omega": OMEGA,
                "modularity": modularity,
                "months": [month_label(year, month) for year, month in months],
                "snapshots": snapshots,
                "transitions": transitions,
                "assignments": assignment_payload,
            }
        ),
        encoding="utf-8",
    )
    return 0


def _spawn(months: list[tuple[int, int]], phase: str) -> tuple[int, int, dict | None]:
    """Run one child. Return exit code, peak RSS in KiB, and the result dict if it finished."""
    if _CHILD_RESULT.exists():
        _CHILD_RESULT.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--phase",
        phase,
        "--months",
        ",".join(month_label(year, month) for year, month in months),
    ]
    # stderr is inherited so the log shows each month as it finishes. stdout stays empty.
    proc = subprocess.Popen(command, cwd=str(ROOT))
    peak = 0
    killed = False
    while proc.poll() is None:
        rss = _rss_kib(proc.pid)
        if rss > peak:
            peak = rss
        if rss > KILL_RSS_KIB:
            print(f"kill {phase} rss_kib={rss} above {KILL_RSS_KIB}", file=sys.stderr, flush=True)
            proc.send_signal(signal.SIGKILL)
            killed = True
            break
        time.sleep(0.25)
    if killed:
        proc.wait()
        return proc.returncode if proc.returncode is not None else -9, peak, None
    code = proc.returncode if proc.returncode is not None else 1
    payload = None
    if code == 0 and _CHILD_RESULT.is_file():
        payload = json.loads(_CHILD_RESULT.read_text(encoding="utf-8"))
    return code, peak, payload


def _parse_months(text: str) -> list[tuple[int, int]]:
    found = []
    for part in text.split(","):
        year, month = part.split("-")
        found.append((int(year), int(month)))
    return found


def _store_assignments(payload: dict) -> None:
    """Write the accepted membership onto the new slice runs. No other run id is opened."""
    from osi.store import save_communities

    for label, mapping in payload["assignments"].items():
        year, month = (int(part) for part in label.split("-"))
        # JSON object keys are strings. The graph nodes are the same account names.
        assignment = {node: int(community) for node, community in mapping.items()}
        save_communities(run_id_for(year, month), "multislice-omega-0.5", assignment)


def _print_summary(payload: dict, peak: int) -> None:
    print(f"omega {payload['omega']}")
    print(f"modularity {payload['modularity']:.6f}")
    print(f"peak_rss_kib {peak}")
    print(f"peak_rss_gib {peak / 1024 / 1024:.3f}")
    for snapshot in payload["snapshots"]:
        print(
            f"{snapshot['month']} communities {snapshot['communities']} "
            f"nodes {snapshot['nodes']} edges {snapshot['edges']}"
        )
    for transition in payload["transitions"]:
        print(
            f"{transition['earlier']} -> {transition['later']} "
            f"persistence {transition['mean_persistence']:.6f} "
            f"shared {transition['shared_members']} "
            f"communities {transition['communities']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build six Reddit months and run one multislice.")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--phase", choices=("build", "leiden"), default="leiden")
    parser.add_argument("--months", default="")
    args = parser.parse_args(argv)
    if args.child:
        months = _parse_months(args.months) if args.months else list(MONTHS)
        try:
            return _child(months, args.phase)
        except Exception as error:
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return 1
    months = list(MONTHS)
    print(f"months {' '.join(month_label(year, month) for year, month in months)}", flush=True)
    code, build_peak, _built = _spawn(months, "build")
    print(f"build_exit {code} build_peak_rss_kib {build_peak}", flush=True)
    if code != 0:
        print("build failed", file=sys.stderr)
        return code if code > 0 else 1
    code, peak, payload = _spawn(months, "leiden")
    print(f"leiden_exit {code} peak_rss_kib {peak}", flush=True)
    # A kill at 5.5 GiB, or any peak past 5 GiB, drops to four slices and tries once more.
    if payload is None or peak >= RETRY_RSS_KIB:
        smaller = reduce_to_four(months)
        print(
            "retry slices "
            + " ".join(month_label(year, month) for year, month in smaller),
            flush=True,
        )
        code, peak, payload = _spawn(smaller, "leiden")
        print(f"retry_exit {code} peak_rss_kib {peak}", flush=True)
        if payload is None:
            print("multislice failed", file=sys.stderr)
            return code if code and code > 0 else 1
    payload["peak_rss_kib"] = peak
    payload["peak_rss_gib"] = peak / 1024 / 1024
    _store_assignments(payload)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(SUMMARY_PATH)
    _print_summary(payload, peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
