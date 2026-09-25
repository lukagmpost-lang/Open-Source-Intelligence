import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.reddit_archive import as_multilayer, subreddit_similarity, user_co_participation
from osi.analysis import louvain_communities


def _comments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"author": "ada", "subreddit": "programming", "link_id": "t3_a"},
            {"author": "grace", "subreddit": "programming", "link_id": "t3_a"},
            {"author": "ada", "subreddit": "science", "link_id": "t3_b"},
            {"author": "grace", "subreddit": "science", "link_id": "t3_b"},
            {"author": "linus", "subreddit": "science", "link_id": "t3_b"},
            {"author": "[deleted]", "subreddit": "science", "link_id": "t3_b"},
        ]
    )


def test_user_edges_count_shared_threads():
    graph = user_co_participation(_comments())
    assert graph.edges["ada", "grace"]["weight"] == 2.0
    assert graph.edges["ada", "linus"]["weight"] == 1.0
    assert "[deleted]" not in graph


def test_subreddit_edges_count_shared_users():
    graph = subreddit_similarity(_comments())
    assert graph.edges["programming", "science"]["weight"] == 2.0
    assert graph.number_of_nodes() == 2


def test_layers_plug_into_the_multilayer_graph_and_louvain():
    comments = _comments()
    users = user_co_participation(comments)
    topics = subreddit_similarity(comments)
    merged = as_multilayer(users, topics)
    assert "reddit_user" in merged.layers()
    assert "reddit_subreddit" in merged.layers()
    assert "reddit_user:ada" in merged.nodes
    # The analysis functions accept the NetworkX layer directly.
    assert louvain_communities(users)
