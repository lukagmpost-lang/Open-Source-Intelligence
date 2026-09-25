"""One month of public Reddit comments from the Arctic Shift Parquet archive.

The Hugging Face dataset is https://huggingface.co/datasets/Dk587/arctic.
Its comment shards stop in 2012, so 2024-01 is not available. The default
month is 2008-01, which is the whole month in a single shard.

Run:
    python3 layers/reddit_archive.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import networkx as nx
import pandas as pd

# Repo root on the path so this file can be run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osi.graph import MultiLayerGraph  # noqa: E402

DATASET = "Dk587/arctic"
# Comments are published only for 2005-12 through 2012. One complete month.
DEFAULT_MONTH = "2008-01"
# Eight communities. reddit.com and politics are left out because they
# dominate the month, and nsfw is left out on purpose.
DEFAULT_SUBREDDITS = (
    "programming",
    "science",
    "entertainment",
    "business",
    "gaming",
    "gadgets",
    "sports",
    "netsec",
)
# A thread with hundreds of accounts turns into hundreds of thousands of
# pairs. Skip those so the co-participation graph stays buildable.
THREAD_AUTHOR_CAP = 30
# Removed accounts are not people, so they never become nodes.
DROPPED_AUTHORS = {"[deleted]", "[removed]", ""}
CACHE = ROOT / "data" / "arctic"


def shard_urls(month: str) -> list[str]:
    """List the Parquet shards for one comment month. Does not download them."""
    year, mon = month.split("-")
    listing = (
        "https://huggingface.co/api/datasets/"
        f"{DATASET}/tree/main/data/comments/{year}/{mon}"
    )
    try:
        with urlopen(listing, timeout=60) as response:
            entries = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"No Arctic Shift comment month {month} in {DATASET}. "
            "Comment shards in this dataset run from 2005-12 through 2012, "
            "so 2024-01 is not there. Pass a month such as 2008-01."
        ) from error
    urls = []
    for entry in entries:
        path = entry.get("path") or ""
        if path.endswith(".parquet"):
            urls.append(f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}")
    if not urls:
        raise RuntimeError(f"Month {month} has no Parquet shards.")
    return sorted(urls)


def _download_one(url: str, target: Path) -> Path:
    # Skip a shard that is already cached so 2008 is never fetched again.
    if not target.exists() or target.stat().st_size == 0:
        temporary = target.with_suffix(target.suffix + ".part")
        with urlopen(url, timeout=180) as response:
            temporary.write_bytes(response.read())
        # Replace only after the full shard is on disk, so a failed fetch is retried.
        temporary.replace(target)
    print(f"shard: {target.name} ({target.stat().st_size} bytes)")
    return target


def download_month(month: str = DEFAULT_MONTH, cache: Path = CACHE) -> list[Path]:
    """Download every shard of one month into the local cache."""
    folder = cache / month
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for url in shard_urls(month):
        name = url.rsplit("/", 1)[-1]
        jobs.append((url, folder / name))
    # A month like 2012-08 is dozens of files. A few at a time is faster than one by one.
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda job: _download_one(*job), jobs))
    paths = sorted(paths)
    print(f"download: {len(paths)} shard(s) for {month}")
    return paths


def load_comments(paths: list[Path], subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS) -> pd.DataFrame:
    """Read only the columns the graphs need, then keep the chosen subreddits."""
    frames = []
    chosen = set(subreddits)
    for path in paths:
        frame = pd.read_parquet(path, columns=["author", "subreddit", "link_id"])
        # Row filter happens after the read so a shard without statistics still works.
        frame = frame[frame["subreddit"].isin(chosen)]
        frame = frame[~frame["author"].isin(DROPPED_AUTHORS)]
        frames.append(frame)
    comments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["author", "subreddit", "link_id"])
    print(
        f"comments: {len(comments)} rows, "
        f"{comments['author'].nunique() if len(comments) else 0} users, "
        f"{comments['link_id'].nunique() if len(comments) else 0} threads, "
        f"{comments['subreddit'].nunique() if len(comments) else 0} subreddits"
    )
    return comments


def user_co_participation(comments: pd.DataFrame, layer: str = "reddit_user") -> nx.Graph:
    """Edge weight is the number of threads two accounts both commented on."""
    threads: dict[str, set[str]] = defaultdict(set)
    for link_id, author in zip(comments["link_id"], comments["author"], strict=False):
        # Dropped names are filtered again here so a raw frame cannot leak them.
        if str(author) in DROPPED_AUTHORS:
            continue
        threads[str(link_id)].add(str(author))

    weights: dict[tuple[str, str], int] = defaultdict(int)
    skipped = 0
    for authors in threads.values():
        # One commenter, or a thread over the cap, does not add pairs.
        if len(authors) < 2 or len(authors) > THREAD_AUTHOR_CAP:
            skipped += 1 if len(authors) > THREAD_AUTHOR_CAP else 0
            continue
        for left, right in combinations(sorted(authors), 2):
            weights[(left, right)] += 1

    graph = nx.Graph()
    for (left, right), weight in weights.items():
        graph.add_node(left, layer=layer)
        graph.add_node(right, layer=layer)
        graph.add_edge(left, right, weight=float(weight), layer=layer)
    print(
        f"{layer}: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {skipped} threads skipped over the cap"
    )
    return graph


def subreddit_similarity(comments: pd.DataFrame, layer: str = "reddit_subreddit") -> nx.Graph:
    """Edge weight is how many users commented in both subreddits."""
    membership: dict[str, set[str]] = defaultdict(set)
    for author, subreddit in zip(comments["author"], comments["subreddit"], strict=False):
        if str(author) in DROPPED_AUTHORS:
            continue
        membership[str(author)].add(str(subreddit))

    weights: dict[tuple[str, str], int] = defaultdict(int)
    for subreddits in membership.values():
        if len(subreddits) < 2:
            continue
        for left, right in combinations(sorted(subreddits), 2):
            weights[(left, right)] += 1

    graph = nx.Graph()
    for (left, right), weight in weights.items():
        graph.add_node(left, layer=layer)
        graph.add_node(right, layer=layer)
        graph.add_edge(left, right, weight=float(weight), layer=layer)
    print(f"{layer}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def as_multilayer(
    user_graph: nx.Graph,
    subreddit_graph: nx.Graph,
    user_layer: str = "reddit_user",
    subreddit_layer: str = "reddit_subreddit",
) -> MultiLayerGraph:
    """Copy both NetworkX graphs into the pipeline's multi-layer graph."""
    graph = MultiLayerGraph()
    for node in user_graph.nodes:
        graph.add_node(user_layer, node, node)
    for left, right, data in user_graph.edges(data=True):
        graph.add_edge(f"{user_layer}:{left}", f"{user_layer}:{right}", user_layer, data.get("weight", 1.0))
    for node in subreddit_graph.nodes:
        graph.add_node(subreddit_layer, node, node)
    for left, right, data in subreddit_graph.edges(data=True):
        graph.add_edge(
            f"{subreddit_layer}:{left}",
            f"{subreddit_layer}:{right}",
            subreddit_layer,
            data.get("weight", 1.0),
        )
    print(f"multilayer: {len(graph.nodes)} nodes, {len(graph.edges)} edges, layers {graph.layers()}")
    return graph


def load_reddit_layers(
    month: str = DEFAULT_MONTH,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
    cache: Path = CACHE,
    user_layer: str = "reddit_user",
    subreddit_layer: str = "reddit_subreddit",
) -> dict[str, nx.Graph]:
    """Download one month, filter it, and return the two layer graphs."""
    paths = download_month(month, cache)
    comments = load_comments(paths, subreddits)
    # Names default to the 2008 layers. A later month passes its own names.
    return {
        user_layer: user_co_participation(comments, user_layer),
        subreddit_layer: subreddit_similarity(comments, subreddit_layer),
    }


def _groups(membership: dict) -> list[set]:
    buckets: dict[int, set] = defaultdict(set)
    for node, community in membership.items():
        buckets[int(community)].add(node)
    return list(buckets.values())


# Exact betweenness visits every node as a source. Above this size that takes hours.
EXACT_BETWEENNESS_LIMIT = 10000
# Sampled betweenness draws this many sources. The seed keeps the ranking repeatable.
SAMPLED_BETWEENNESS_SOURCES = 400


def _betweenness(graph: nx.Graph) -> dict:
    from osi.analysis import betweenness_centrality

    node_count = graph.number_of_nodes()
    if node_count <= EXACT_BETWEENNESS_LIMIT:
        return betweenness_centrality(graph)
    print(
        f"betweenness: exact is too slow for {node_count} nodes. "
        f"Using {SAMPLED_BETWEENNESS_SOURCES} random source nodes instead."
    )
    scores = nx.betweenness_centrality(
        graph,
        k=SAMPLED_BETWEENNESS_SOURCES,
        weight="weight",
        seed=42,
    )
    return dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))


def summarize_user_graph(graph: nx.Graph) -> dict:
    """Louvain, Leiden, PageRank, and betweenness for one user layer."""
    from networkx.algorithms.community.quality import modularity

    from osi.analysis import leiden_communities, louvain_communities, pagerank

    louvain = louvain_communities(graph)
    leiden = leiden_communities(graph)
    ranks = pagerank(graph)
    bridges = _betweenness(graph)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "louvain_communities": len(set(louvain.values())),
        "leiden_communities": len(set(leiden.values())),
        "louvain_modularity": modularity(graph, _groups(louvain), weight="weight"),
        "leiden_modularity": modularity(graph, _groups(leiden), weight="weight"),
        "pagerank": list(ranks.items())[:5],
        "betweenness": list(bridges.items())[:5],
    }


def compare_2008_and_2012() -> None:
    """Build both months and print one comparison table. 2008 keeps its layer names."""
    older = load_reddit_layers("2008-01")
    newer = load_reddit_layers("2012-08", user_layer="reddit_2012_user", subreddit_layer="reddit_2012_subreddit")
    as_multilayer(older["reddit_user"], older["reddit_subreddit"])
    as_multilayer(
        newer["reddit_2012_user"],
        newer["reddit_2012_subreddit"],
        user_layer="reddit_2012_user",
        subreddit_layer="reddit_2012_subreddit",
    )
    print("scoring 2008")
    left = summarize_user_graph(older["reddit_user"])
    print("scoring 2012")
    right = summarize_user_graph(newer["reddit_2012_user"])
    print("COMPARISON")
    print(f"{'metric':<24} {'2008':<40} {'2012'}")
    for key in ("nodes", "edges", "louvain_communities", "leiden_communities", "louvain_modularity", "leiden_modularity"):
        print(f"{key:<24} {left[key]!s:<40} {right[key]}")
    print("top 5 PageRank")
    for index in range(5):
        old = left["pagerank"][index] if index < len(left["pagerank"]) else ("", "")
        new = right["pagerank"][index] if index < len(right["pagerank"]) else ("", "")
        print(f"{index + 1:<24} {old[0]} {old[1]:.6f}{'':<20} {new[0]} {new[1]:.6f}")
    print("top 5 betweenness")
    for index in range(5):
        old = left["betweenness"][index] if index < len(left["betweenness"]) else ("", "")
        new = right["betweenness"][index] if index < len(right["betweenness"]) else ("", "")
        print(f"{index + 1:<24} {old[0]} {old[1]:.6f}{'':<20} {new[0]} {new[1]:.6f}")


def main() -> int:
    layers = load_reddit_layers()
    as_multilayer(layers["reddit_user"], layers["reddit_subreddit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
