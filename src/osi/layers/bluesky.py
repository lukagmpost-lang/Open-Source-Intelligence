"""Bluesky follow layer from the public AT Protocol API. No API key."""

from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import quote

import networkx as nx

from osi.limits import get_json

# One request per second. The public AppView allows more, this stays under it.
_PAUSE_SECONDS = 1.0
# public.api.bsky.app is the AppView. api.bsky.app is the fallback when that host returns 403.
_HOSTS = ("https://public.api.bsky.app", "https://api.bsky.app")
_host_index = 0
_last_request = 0.0


def _throttle() -> None:
    global _last_request
    if not _last_request:
        return
    delay = _PAUSE_SECONDS - (time.monotonic() - _last_request)
    if delay > 0:
        time.sleep(delay)


def _mark_request() -> None:
    global _last_request
    _last_request = time.monotonic()


def _get(path: str) -> dict | None:
    """GET one XRPC path. A 403 on the public host switches to the fallback host."""
    global _host_index
    _throttle()
    url = f"{_HOSTS[_host_index]}{path}"
    try:
        payload = get_json(url)
    except (OSError, RuntimeError, ValueError) as error:
        message = str(error)
        # Only the public host's 403 is a reason to try the other host. Other errors stop here.
        if "HTTP 403" in message and _host_index == 0:
            _host_index = 1
            _mark_request()
            print("note: public.api.bsky.app returned 403; using api.bsky.app", file=sys.stderr)
            return _get(path)
        _mark_request()
        print(f"note: bluesky request failed: {message}", file=sys.stderr)
        return None
    _mark_request()
    if not isinstance(payload, dict):
        print(f"note: bluesky request failed: {path} was not a JSON object", file=sys.stderr)
        return None
    return payload


def _profile_fields(payload: dict) -> dict[str, Any]:
    return {
        "handle": payload.get("handle") or "",
        "did": payload.get("did") or "",
        "displayName": payload.get("displayName") or "",
        "description": payload.get("description") or "",
        "followersCount": int(payload.get("followersCount") or 0),
        "followsCount": int(payload.get("followsCount") or 0),
        "postsCount": int(payload.get("postsCount") or 0),
    }


def fetch_profile(handle: str) -> dict[str, Any]:
    """Profile for one actor. An error returns an empty dict."""
    text = str(handle).strip()
    if not text:
        return {}
    payload = _get(f"/xrpc/app.bsky.actor.getProfile?actor={quote(text)}")
    if not payload or not payload.get("handle"):
        print(f"note: bluesky profile {text} unavailable", file=sys.stderr)
        return {}
    return _profile_fields(payload)


def _actor_row(row: dict) -> dict[str, str] | None:
    handle = row.get("handle")
    if not handle:
        return None
    return {
        "handle": str(handle),
        "did": str(row.get("did") or ""),
        "displayName": str(row.get("displayName") or ""),
    }


def _collect(method: str, list_key: str, handle: str, limit: int) -> list[dict]:
    """Page an actor list until limit rows are collected or the cursor runs out."""
    text = str(handle).strip()
    if not text or limit < 1:
        return []
    collected: list[dict] = []
    cursor = None
    while len(collected) < limit:
        # The AppView rejects a limit above 100, so each page asks only for the remainder.
        page_size = min(100, limit - len(collected))
        path = f"/xrpc/{method}?actor={quote(text)}&limit={page_size}"
        if cursor:
            path = f"{path}&cursor={quote(cursor)}"
        payload = _get(path)
        if not payload:
            break
        rows = payload.get(list_key) or []
        if not isinstance(rows, list) or not rows:
            break
        collected.extend(row for row in rows if isinstance(row, dict))
        cursor = payload.get("cursor")
        if not cursor:
            break
    return collected[:limit]


def fetch_follows(handle: str, limit: int = 100) -> list[dict[str, str]]:
    """Accounts this actor follows. An error or a private graph returns []."""
    people = []
    for row in _collect("app.bsky.graph.getFollows", "follows", handle, limit):
        actor = _actor_row(row)
        if actor:
            people.append(actor)
    if not people:
        print(f"note: no bluesky follows for {handle}", file=sys.stderr)
    return people


def fetch_followers(handle: str, limit: int = 100) -> list[dict[str, str]]:
    """Accounts that follow this actor. An error or a private graph returns []."""
    people = []
    for row in _collect("app.bsky.graph.getFollowers", "followers", handle, limit):
        actor = _actor_row(row)
        if actor:
            people.append(actor)
    if not people:
        print(f"note: no bluesky followers for {handle}", file=sys.stderr)
    return people


def fetch_posts(handle: str, limit: int = 50) -> list[dict[str, Any]]:
    """Recent posts: text, time, reply target, likes, and reposts. An error returns []."""
    posts = []
    for row in _collect("app.bsky.feed.getAuthorFeed", "feed", handle, limit):
        post = row.get("post") or {}
        record = post.get("record") or {}
        if not isinstance(post, dict) or not isinstance(record, dict):
            continue
        posts.append(
            {
                "text": record.get("text") or "",
                "createdAt": record.get("createdAt") or "",
                # Absent when the post is not a reply. The parent uri is the reply target.
                "reply": record.get("reply"),
                "likeCount": int(post.get("likeCount") or 0),
                "repostCount": int(post.get("repostCount") or 0),
            }
        )
    if not posts:
        print(f"note: no bluesky posts for {handle}", file=sys.stderr)
    return posts


def _node_id(handle: str) -> str:
    return f"bluesky:{handle}"


def _ensure_node(graph: nx.Graph, handle: str, attrs: dict) -> str:
    node = _node_id(handle)
    if node not in graph:
        graph.add_node(node, platform="bluesky", label=handle, **attrs)
        return node
    # A neighbor row is thinner than a profile. Do not wipe counts already stored on a center.
    for key, value in attrs.items():
        if value not in (None, "") or key not in graph.nodes[node]:
            graph.nodes[node][key] = value
    return node


def _add_follow(graph: nx.Graph, left: str, right: str, direction: str) -> None:
    if left == right:
        return
    if not graph.has_edge(left, right):
        relation = "mutual" if direction == "mutual" else "follow"
        graph.add_edge(left, right, relation=relation, direction=direction, weight=1.0, layer=relation)
        return
    current = graph.edges[left, right]
    # The other list can record the opposite direction of the same follow.
    if current.get("direction") != direction:
        current["direction"] = "mutual"
        current["relation"] = "mutual"
        current["layer"] = "mutual"


def build_bluesky_layer(handles: list[str]) -> nx.Graph:
    """Profiles and follow edges for each handle. Recent posts are stored on the center node."""
    graph = nx.Graph()
    seen: set[str] = set()
    for handle in handles:
        text = str(handle).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        profile = fetch_profile(text)
        if not profile:
            print(f"note: bluesky handle {text} skipped", file=sys.stderr)
            continue
        # Keep the requested spelling as the node id so the identity map still matches.
        center = _ensure_node(graph, text, {**profile, "handle": text, "posts": fetch_posts(text)})
        for person in fetch_follows(text):
            other = _ensure_node(
                graph,
                person["handle"],
                {"handle": person["handle"], "did": person["did"], "displayName": person["displayName"]},
            )
            _add_follow(graph, center, other, "following")
        for person in fetch_followers(text):
            other = _ensure_node(
                graph,
                person["handle"],
                {"handle": person["handle"], "did": person["did"], "displayName": person["displayName"]},
            )
            _add_follow(graph, center, other, "follower")
    return graph
