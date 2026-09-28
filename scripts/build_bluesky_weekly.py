"""One Bluesky reply graph for each of the last 90 ISO weeks.

Run:
    python3 scripts/build_bluesky_weekly.py

Each week is saved as bluesky-reply-YYYY-WW. An id that already exists is
left as it is. Live requests go to public.api.bsky.app and wait two seconds.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osi.layers.bluesky_organic import (  # noqa: E402
    _THREAD_DEPTH,
    _cached_get,
    authors_in_thread,
    seed_handles,
    thread_root,
)
from osi.layers.co_participation import GROUP_CAP, co_participation_graph  # noqa: E402
from osi.limits import get_json  # noqa: E402
from osi.store import create_run, get_run, save_graph  # noqa: E402

# Ninety ISO weeks, counting the week that contains today.
_WEEKS = 90
# One author-feed page. The AppView rejects a limit above 100.
_FEED_LIMIT = 100
LAYER = "bluesky_reply"
_CACHE = ROOT / "data" / "bluesky_weekly_cache"


def _weeks(now: datetime | None = None) -> list[tuple[int, int, datetime]]:
    """(iso year, iso week, Monday 00:00 UTC) for the last 90 weeks, oldest first."""
    moment = now or datetime.now(timezone.utc)
    # weekday() is 0 on Monday, which is the ISO week boundary.
    monday = (moment - timedelta(days=moment.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    found = []
    for offset in range(_WEEKS - 1, -1, -1):
        start = monday - timedelta(weeks=offset)
        iso = start.isocalendar()
        found.append((iso.year, iso.week, start))
    return found


def _stamp(value: str) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    # fromisoformat does not accept the Z suffix Bluesky sends.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _feed(handle: str, cutoff: datetime) -> tuple[list[dict], bool]:
    """Posts at or after cutoff. The bool is False when the feed never answered."""
    posts: list[dict] = []
    cursor = None
    answered = False
    while True:
        path = f"/xrpc/app.bsky.feed.getAuthorFeed?actor={quote(handle)}&limit={_FEED_LIMIT}"
        if cursor:
            path = f"{path}&cursor={quote(cursor)}"
        payload = _cached_get(path, get_json, _CACHE, True)
        if not payload:
            break
        answered = True
        feed = payload.get("feed") or []
        if not isinstance(feed, list) or not feed:
            break
        older = False
        for row in feed:
            if not isinstance(row, dict):
                continue
            post = row.get("post")
            if not isinstance(post, dict):
                continue
            record = post.get("record") if isinstance(post.get("record"), dict) else {}
            created = _stamp(str(record.get("createdAt") or ""))
            # Pages run newest first, so one older post means the rest of the cursor is older too.
            if created is None or created < cutoff:
                older = True
                continue
            posts.append(post)
        cursor = payload.get("cursor")
        if older or not cursor:
            break
    return posts, answered


def _root_created(thread: dict) -> datetime | None:
    post = thread.get("post") if isinstance(thread.get("post"), dict) else {}
    record = post.get("record") if isinstance(post.get("record"), dict) else {}
    return _stamp(str(record.get("createdAt") or ""))


def main() -> int:
    weeks = _weeks()
    cutoff = weeks[0][2]
    allowed = {(year, week) for year, week, _start in weeks}
    seeds = seed_handles()
    print(f"seeds {len(seeds)} cutoff {cutoff.isoformat()} weeks {len(weeks)}", file=sys.stderr)
    roots: list[str] = []
    seen: set[str] = set()
    answered = False
    for handle in seeds:
        posts, ok = _feed(handle, cutoff)
        answered = answered or ok
        print(f"feed {handle} posts {len(posts)}", file=sys.stderr)
        for post in posts:
            root = thread_root(post)
            if not root or root in seen:
                continue
            seen.add(root)
            roots.append(root)
    if not answered:
        print("error: no author feed could be fetched", file=sys.stderr)
        return 1
    # Authors of one thread, filed under the ISO week of the root post.
    groups: dict[tuple[int, int], list[list[str]]] = {key: [] for key in allowed}
    failed = 0
    for root in roots:
        path = f"/xrpc/app.bsky.feed.getPostThread?uri={quote(root, safe='')}&depth={_THREAD_DEPTH}"
        payload = _cached_get(path, get_json, _CACHE, True)
        thread = payload.get("thread") if isinstance(payload, dict) else None
        if not isinstance(thread, dict):
            failed += 1
            continue
        created = _root_created(thread)
        if created is None:
            failed += 1
            continue
        iso = created.isocalendar()
        key = (iso.year, iso.week)
        # A reply inside the window can point at a root from before the window.
        if key not in groups:
            continue
        groups[key].append(sorted(authors_in_thread(thread), key=str.casefold))
    print(f"threads {len(roots)} failed {failed}", file=sys.stderr)
    for year, week, start in weeks:
        label = f"{year:04d}-{week:02d}"
        run_id = f"bluesky-reply-{label}"
        graph, _over, _under = co_participation_graph(groups[(year, week)], LAYER, GROUP_CAP)
        # A second launch must not replace a week that already finished.
        if get_run(run_id) is None:
            create_run(
                "bluesky_reply",
                {
                    "layer": LAYER,
                    "week": label,
                    "start": start.isoformat(),
                    "depth": _THREAD_DEPTH,
                    "cap": GROUP_CAP,
                    "seeds": list(seeds),
                },
                notes=run_id,
                run_id=run_id,
            )
            save_graph(run_id, LAYER, graph)
        else:
            print(f"note: {run_id} already stored; left unchanged", file=sys.stderr)
        print(f"{label} {graph.number_of_nodes()} {graph.number_of_edges()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
