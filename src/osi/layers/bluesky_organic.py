"""Bluesky co-reply graph.

Two handles share an edge when they both appear in the same post thread.
The weight is the number of threads they share. A thread with more than 30
authors is skipped, the same cap as a Reddit thread and a GitHub repository.

This is not the follow graph. Follow edges stay in osi.layers.bluesky.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import networkx as nx

from osi.identity import load_identity_map
from osi.layers.co_participation import co_participation_graph
from osi.limits import get_json

LAYER = "bluesky_organic"
# One live request per two seconds. Cache hits do not count.
_PAUSE_SECONDS = 2.0
# public.api.bsky.app is the AppView. api.bsky.app is the fallback on HTTP 403.
_HOSTS = ("https://public.api.bsky.app", "https://api.bsky.app")
# One author-feed page per seed. Enough posts to find threads, not a full history.
_FEED_LIMIT = 100
# How far getPostThread nests replies. Deeper replies are outside this view.
_THREAD_DEPTH = 6
_KNOWN_BOTS = {"handle.invalid"}
_ROOT = Path(__file__).resolve().parents[3]
_CACHE = _ROOT / "data" / "bluesky_organic_cache"
_IDENTITY = _ROOT / "identity.json"
_last_request = 0.0


def seed_handles(identity_path: Path | None = None) -> tuple[str, ...]:
    """Bluesky handles recorded in identity.json. Order follows the file."""
    path = identity_path or _IDENTITY
    if not path.is_file():
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for accounts in load_identity_map(path).values():
        if not isinstance(accounts, dict):
            continue
        handle = str(accounts.get("bluesky") or "").strip()
        key = handle.casefold()
        if not handle or key in seen:
            continue
        seen.add(key)
        found.append(handle)
    return tuple(found)


def is_bot(handle: str) -> bool:
    """True for a blank handle, a [bot] suffix, or a known non-person handle."""
    text = str(handle).strip()
    if not text:
        return True
    folded = text.casefold()
    if folded.endswith("[bot]"):
        return True
    return folded in _KNOWN_BOTS


def _cache_file(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_cache(cache_dir: Path, url: str) -> Any | None:
    path = _cache_file(cache_dir, url)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("payload")


def _write_cache(cache_dir: Path, url: str, payload: Any) -> None:
    path = _cache_file(cache_dir, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "payload": payload}), encoding="utf-8")


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


def _cached_get(
    path: str,
    fetch: Callable[..., Any],
    cache_dir: Path,
    pause: bool,
) -> dict | None:
    """GET one XRPC path. A cached host is used as-is. A 403 tries the other host."""
    for index, host in enumerate(_HOSTS):
        url = f"{host}{path}"
        cached = _read_cache(cache_dir, url)
        if cached is not None:
            return cached if isinstance(cached, dict) else None
        if pause:
            _throttle()
        try:
            payload = fetch(url)
        except (OSError, RuntimeError, ValueError) as error:
            if pause:
                _mark_request()
            message = str(error)
            if "HTTP 403" in message and index == 0:
                print("note: public.api.bsky.app returned 403; using api.bsky.app", file=sys.stderr)
                continue
            print(f"note: bluesky request failed: {message}", file=sys.stderr)
            return None
        if pause:
            _mark_request()
        if not isinstance(payload, dict):
            print(f"note: bluesky request failed: {path} was not a JSON object", file=sys.stderr)
            return None
        _write_cache(cache_dir, url, payload)
        return payload
    return None


def thread_root(post: dict) -> str | None:
    """The thread root uri, or the post's own uri when it is not a reply."""
    record = post.get("record") if isinstance(post.get("record"), dict) else {}
    reply = record.get("reply") if isinstance(record, dict) else None
    if isinstance(reply, dict):
        root = reply.get("root") if isinstance(reply.get("root"), dict) else {}
        uri = root.get("uri") if isinstance(root, dict) else None
        if uri:
            return str(uri)
    uri = post.get("uri")
    return str(uri) if uri else None


def authors_in_thread(node: Any, found: set[str] | None = None) -> set[str]:
    """Handles on a thread view, including the parent chain and nested replies."""
    if found is None:
        found = set()
    if not isinstance(node, dict):
        return found
    post = node.get("post") if isinstance(node.get("post"), dict) else {}
    author = post.get("author") if isinstance(post.get("author"), dict) else {}
    handle = author.get("handle") if isinstance(author, dict) else None
    if handle and not is_bot(str(handle)):
        found.add(str(handle))
    parent = node.get("parent")
    if isinstance(parent, dict):
        authors_in_thread(parent, found)
    replies = node.get("replies")
    if isinstance(replies, list):
        for reply in replies:
            authors_in_thread(reply, found)
    return found


def _feed_roots(payload: dict) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for row in payload.get("feed") or []:
        if not isinstance(row, dict):
            continue
        post = row.get("post")
        if not isinstance(post, dict):
            continue
        root = thread_root(post)
        if not root or root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def build_bluesky_organic(
    handles: tuple[str, ...] | None = None,
    fetch: Callable[..., Any] | None = None,
    cache_dir: Path | None = None,
    identity_path: Path | None = None,
) -> nx.Graph:
    """Co-reply graph for the identity.json Bluesky handles. Pass ``fetch`` to avoid the network."""
    seeds = seed_handles(identity_path) if handles is None else handles
    getter = fetch or get_json
    folder = cache_dir or _CACHE
    pause = fetch is None
    roots: list[str] = []
    seen_roots: set[str] = set()
    for handle in seeds:
        text = str(handle).strip()
        if not text:
            continue
        path = f"/xrpc/app.bsky.feed.getAuthorFeed?actor={quote(text)}&limit={_FEED_LIMIT}"
        payload = _cached_get(path, getter, folder, pause)
        if not payload:
            print(f"note: no bluesky posts for {text}", file=sys.stderr)
            continue
        for root in _feed_roots(payload):
            if root in seen_roots:
                continue
            seen_roots.add(root)
            roots.append(root)
    groups: list[list[str]] = []
    failed = 0
    for root in roots:
        path = f"/xrpc/app.bsky.feed.getPostThread?uri={quote(root, safe='')}&depth={_THREAD_DEPTH}"
        print(f"bluesky_organic: thread {root}", file=sys.stderr)
        payload = _cached_get(path, getter, folder, pause)
        thread = payload.get("thread") if isinstance(payload, dict) else None
        if not isinstance(thread, dict):
            failed += 1
            continue
        groups.append(sorted(authors_in_thread(thread), key=str.casefold))
    graph, over, under = co_participation_graph(groups, LAYER)
    used = len(groups) - over - under
    print(
        f"{LAYER}: {len(seeds)} seeds, {len(roots)} threads, {used} threads used, "
        f"{over} skipped over the cap, {under} skipped under 2 authors, {failed} failed, "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    return graph
