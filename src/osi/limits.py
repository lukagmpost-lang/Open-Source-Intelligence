"""Hard caps so connectors stay inside public API budgets."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "osi-public-graph/0.1"
MAX_BYTES = 2_000_000
TIMEOUT = 20

# Per-call ceilings. These are deliberately below documented platform quotas.
GITHUB_PAGE_SIZE = 100
GITHUB_MAX_PAGES = 2
REDDIT_MAX_ITEMS = 100
STEAM_MAX_FRIENDS = 300
STEAM_MAX_GAMES = 200
SPOTIFY_MAX_PLAYLISTS = 20
SPOTIFY_MAX_TRACKS = 100
TELEGRAM_PAGE_SIZE = 50
TELEGRAM_MAX_PAGES = 2


def get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise RuntimeError(f"response from {url} exceeded {MAX_BYTES} bytes")
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 429:
            time.sleep(min(int(error.headers.get("Retry-After", "1")), 5))
            raise RuntimeError("rate limited by the upstream API; retry later") from error
        detail = error.read(500).decode("utf-8", errors="replace")
        path = url.split("?", 1)[0]
        raise RuntimeError(f"upstream HTTP {error.code} for {path}: {detail}") from None
