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


def download_month(month: str = DEFAULT_MONTH, cache: Path = CACHE) -> list[Path]:
    """Download every shard of one month into the local cache."""
    folder = cache / month
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in shard_urls(month):
        name = url.rsplit("/", 1)[-1]
        target = folder / name
        # Keep a shard that is already on disk so a rerun does not fetch it again.
        if not target.exists() or target.stat().st_size == 0:
            with urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        paths.append(target)
        print(f"shard: {target.name} ({target.stat().st_size} bytes)")
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


def user_co_participation(comments: pd.DataFrame) -> nx.Graph:
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
        graph.add_node(left, layer="reddit_user")
        graph.add_node(right, layer="reddit_user")
        graph.add_edge(left, right, weight=float(weight), layer="reddit_user")
    print(
        f"reddit_user: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {skipped} threads skipped over the cap"
    )
    return graph


def subreddit_similarity(comments: pd.DataFrame) -> nx.Graph:
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
        graph.add_node(left, layer="reddit_subreddit")
        graph.add_node(right, layer="reddit_subreddit")
        graph.add_edge(left, right, weight=float(weight), layer="reddit_subreddit")
    print(f"reddit_subreddit: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def as_multilayer(user_graph: nx.Graph, subreddit_graph: nx.Graph) -> MultiLayerGraph:
    """Copy both NetworkX graphs into the pipeline's multi-layer graph."""
    graph = MultiLayerGraph()
    for node in user_graph.nodes:
        graph.add_node("reddit_user", node, node)
    for left, right, data in user_graph.edges(data=True):
        graph.add_edge(f"reddit_user:{left}", f"reddit_user:{right}", "reddit_user", data.get("weight", 1.0))
    for node in subreddit_graph.nodes:
        graph.add_node("reddit_subreddit", node, node)
    for left, right, data in subreddit_graph.edges(data=True):
        graph.add_edge(
            f"reddit_subreddit:{left}",
            f"reddit_subreddit:{right}",
            "reddit_subreddit",
            data.get("weight", 1.0),
        )
    print(f"multilayer: {len(graph.nodes)} nodes, {len(graph.edges)} edges, layers {graph.layers()}")
    return graph


def load_reddit_layers(
    month: str = DEFAULT_MONTH,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
    cache: Path = CACHE,
) -> dict[str, nx.Graph]:
    """Download one month, filter it, and return the two layer graphs."""
    paths = download_month(month, cache)
    comments = load_comments(paths, subreddits)
    return {
        "reddit_user": user_co_participation(comments),
        "reddit_subreddit": subreddit_similarity(comments),
    }


def main() -> int:
    layers = load_reddit_layers()
    as_multilayer(layers["reddit_user"], layers["reddit_subreddit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
