"""GitHub co-contribution graph.

Two logins share an edge when they both contributed to the same repository.
The weight is the number of repositories they share. A repository with more
than 30 human contributors is skipped, the same cap Reddit uses for a thread.

Every API response is written under data/github_organic_cache. A cache hit
does not sleep and does not call the network. A live call waits so requests
stay two seconds apart.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import networkx as nx

from osi.github_graph import _headers
from osi.layers.co_participation import GROUP_CAP, co_participation_graph
from osi.limits import get_json

# The seven GitHub logins already used as ego seeds.
SEED_LOGINS = (
    "torvalds",
    "antirez",
    "rtomayko",
    "hadley",
    "chromakode",
    "defunkt",
    "mojombo",
)
# Asked for by name even when the owner list would not include them.
NAMED_REPOS = (
    ("torvalds", "linux"),
    ("antirez", "redis"),
    ("rtomayko", "rack"),
    ("hadley", "ggplot2"),
)
LAYER = "github_organic"
# One live request per two seconds. Cache hits do not count.
_PAUSE_SECONDS = 2.0
# Stop a huge account from paging forever. 10 pages is 1,000 owned repos.
_MAX_OWNER_PAGES = 10
# Top contributors page. A full page means the repo is larger than we fetched.
_CONTRIBUTOR_PAGE = 100
# Logins that are automation even when the API type field says User.
_KNOWN_BOTS = {
    "dependabot",
    "dependabot-preview",
    "renovate",
    "renovate-bot",
    "github-actions",
    "pyup-bot",
    "greenkeeper",
    "greenkeeperio-bot",
    "imgbot",
    "allcontributors",
    "codecov",
    "codecov-io",
    "whitesource-bolt",
    "snyk-bot",
}
_ROOT = Path(__file__).resolve().parents[3]
_CACHE = _ROOT / "data" / "github_organic_cache"
_last_request = 0.0


def is_bot(login: str, account_type: str | None = None) -> bool:
    """True for a [bot] suffix, an API Bot, or a known automation login."""
    text = str(login).strip()
    if not text:
        return True
    folded = text.casefold()
    if folded.endswith("[bot]"):
        return True
    if account_type and str(account_type).casefold() == "bot":
        return True
    return folded in _KNOWN_BOTS


def human_logins(accounts: list[Any]) -> list[str]:
    """Contributor logins after bots and anonymous rows are removed."""
    found: list[str] = []
    seen: set[str] = set()
    for account in accounts:
        if not isinstance(account, dict):
            continue
        login = account.get("login")
        if not login or is_bot(str(login), account.get("type")):
            continue
        key = str(login).casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(str(login))
    return found


def _cache_file(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_cache(cache_dir: Path, url: str) -> Any | None:
    path = _cache_file(cache_dir, url)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("payload")


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
    url: str,
    fetch: Callable[..., Any],
    cache_dir: Path,
    pause: bool,
) -> Any:
    """Return the cached JSON, or fetch it and store the body. Failures are not stored."""
    cached = _read_cache(cache_dir, url)
    if cached is not None:
        return cached
    if pause:
        _throttle()
    try:
        payload = fetch(url, _headers())
    finally:
        # The clock starts even when the call fails, so the next live call still waits.
        if pause:
            _mark_request()
    _write_cache(cache_dir, url, payload)
    return payload


def _owner_repo_url(login: str, page: int) -> str:
    return (
        "https://api.github.com/users/"
        f"{login}/repos?type=owner&per_page=100&page={page}&sort=full_name&direction=asc"
    )


def _contributor_url(owner: str, repo: str) -> str:
    return (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/contributors?per_page={_CONTRIBUTOR_PAGE}"
    )


def _list_owned(login: str, fetch: Callable[..., Any], cache_dir: Path, pause: bool) -> list[tuple[str, str, bool]]:
    """Owned repositories as (owner, name, is_fork). A failed page stops that login."""
    found: list[tuple[str, str, bool]] = []
    for page in range(1, _MAX_OWNER_PAGES + 1):
        url = _owner_repo_url(login, page)
        try:
            payload = _cached_get(url, fetch, cache_dir, pause)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"note: skip repo list {login} page {page}: {error}", file=sys.stderr)
            break
        if not isinstance(payload, list) or not payload:
            break
        for repo in payload:
            if not isinstance(repo, dict) or not repo.get("name"):
                continue
            owner = (repo.get("owner") or {}).get("login") or login
            found.append((str(owner), str(repo["name"]), bool(repo.get("fork"))))
        if len(payload) < 100:
            break
        if page == _MAX_OWNER_PAGES:
            print(f"note: {login} owned-repo list stopped at {_MAX_OWNER_PAGES} pages", file=sys.stderr)
    return found


def select_repos(
    owned: list[tuple[str, str, bool]],
    named: tuple[tuple[str, str], ...] = NAMED_REPOS,
) -> list[tuple[str, str]]:
    """Non-fork owned repos, plus the named seeds even if they are forks or unlisted.

    Forks repeat the upstream contributor list, so pairing them would count
    the same people once per fork and burn the request budget.
    """
    chosen: dict[str, tuple[str, str]] = {}
    for owner, name, is_fork in owned:
        if is_fork:
            continue
        chosen[f"{owner}/{name}".casefold()] = (owner, name)
    for owner, name in named:
        chosen.setdefault(f"{owner}/{name}".casefold(), (owner, name))
    return [chosen[key] for key in sorted(chosen)]


def contributor_group(accounts: list[Any]) -> list[str] | None:
    """Human logins to pair, or None when this repo must be skipped as too big.

    None means the fetched page is full (more contributors exist than the top
    100) or more than 30 humans remain after bots are removed. A short page
    with fewer than two humans is returned as a short list so the caller can
    count it as under-size rather than over-cap.
    """
    if len(accounts) >= _CONTRIBUTOR_PAGE:
        return None
    return human_logins(accounts)


def build_github_organic(
    logins: tuple[str, ...] = SEED_LOGINS,
    named_repos: tuple[tuple[str, str], ...] = NAMED_REPOS,
    fetch: Callable[..., Any] | None = None,
    cache_dir: Path | None = None,
) -> nx.Graph:
    """Co-contribution graph for the seed logins. Pass ``fetch`` to avoid the network."""
    getter = fetch or get_json
    folder = cache_dir or _CACHE
    # An injected fetch is a test double. Only the real client waits two seconds.
    pause = fetch is None
    owned: list[tuple[str, str, bool]] = []
    for login in logins:
        owned.extend(_list_owned(login, getter, folder, pause))
    repos = select_repos(owned, named_repos)
    groups: list[list[str]] = []
    failed = 0
    # Full contributor pages are over the cap before pairing. They never enter groups.
    truncated = 0
    over_names: list[str] = []
    for owner, name in repos:
        url = _contributor_url(owner, name)
        print(f"github_organic: contributors {owner}/{name}", file=sys.stderr)
        try:
            payload = _cached_get(url, getter, folder, pause)
        except (OSError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"note: skip {owner}/{name}: {error}", file=sys.stderr)
            continue
        if not isinstance(payload, list):
            failed += 1
            print(f"note: skip {owner}/{name}: contributors were not a list", file=sys.stderr)
            continue
        humans = contributor_group(payload)
        if humans is None:
            truncated += 1
            over_names.append(f"{owner}/{name}")
            continue
        # A short page can still hold more humans than the cap. Name it with the full pages.
        if len(humans) > GROUP_CAP:
            over_names.append(f"{owner}/{name}")
        groups.append(humans)
    graph, over, under = co_participation_graph(groups, LAYER)
    # Groups that became edges. Over-cap and under-2 lists are not used.
    used = len(groups) - over - under
    over += truncated
    print(
        f"{LAYER}: {used} repos used, {over} skipped over the cap, "
        f"{under} skipped under 2 contributors, {failed} failed, "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    if over_names:
        print(f"{LAYER} over cap: {', '.join(over_names)}")
    return graph
