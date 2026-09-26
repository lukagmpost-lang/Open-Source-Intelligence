"""Public Reddit comments via the official OAuth API."""

from __future__ import annotations

import os

from osi.graph import Graph
from osi.limits import REDDIT_MAX_ITEMS, get_json


def fetch_reddit(username: str) -> Graph:
    """Load a public profile's recent comments. Requires REDDIT_TOKEN (oauth bearer)."""
    username = username.strip().removeprefix("u/")
    token = os.environ.get("REDDIT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("REDDIT_TOKEN is required; anonymous .json scraping is not used")
    if not username or "/" in username:
        raise ValueError("username must be a single public Reddit name")
    headers = {"Authorization": f"Bearer {token}"}
    about = get_json(f"https://oauth.reddit.com/user/{username}/about", headers)
    data = about.get("data") or {}
    if data.get("is_suspended") or not data.get("name"):
        raise RuntimeError(f"Reddit user {username} is not a public profile")
    graph = Graph()
    root = graph.add_node("reddit", data["name"], data["name"])
    listing = get_json(
        f"https://oauth.reddit.com/user/{username}/comments?limit={REDDIT_MAX_ITEMS}",
        headers,
    )
    children = ((listing.get("data") or {}).get("children")) or []
    subreddits: set[str] = set()
    for child in children[:REDDIT_MAX_ITEMS]:
        body = child.get("data") or {}
        subreddit = body.get("subreddit")
        if not subreddit:
            continue
        subreddits.add(subreddit)
        graph.add_node("reddit", f"r/{subreddit}", f"r/{subreddit}", {"kind": "subreddit"})
        graph.add_edge(root.id, f"reddit:r/{subreddit}", "comment", 1.0)
    return graph
