"""Profile-displayed Telegram gifts via the Bot API.

Telegram documents that a gift is fetchable by other users only after the
recipient displays it on their profile. Hidden gifts and private channels are
not requested. Gifts with a hidden sender are dropped, because the sender is
not public.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from osi.graph import Graph
from osi.limits import MAX_BYTES, TELEGRAM_MAX_PAGES, TELEGRAM_PAGE_SIZE, TIMEOUT, USER_AGENT


def _public_sender(gift: dict) -> str | None:
    """Return a sender id only when the gift is pinned and the sender is shown."""
    displayed = gift.get("is_saved")
    if displayed is None:
        displayed = gift.get("saved")
    if displayed is not True:
        return None
    if gift.get("is_private") or gift.get("name_hidden") or gift.get("sender_hidden"):
        return None
    sender = gift.get("sender_user") or gift.get("from_user") or {}
    sender_id = sender.get("id") or gift.get("sender_id")
    if sender_id is None:
        return None
    return str(sender_id)


def gifts_to_graph(owner_id: str, owner_label: str, gifts: list[dict]) -> Graph:
    graph = Graph()
    owner = graph.add_node("telegram", owner_id, owner_label or owner_id)
    for gift in gifts:
        sender_id = _public_sender(gift)
        if sender_id is None or sender_id == owner_id:
            continue
        sender = gift.get("sender_user") or gift.get("from_user") or {}
        label = sender.get("username") or sender.get("first_name") or sender_id
        graph.add_node("telegram", sender_id, str(label))
        graph.add_edge(f"telegram:{sender_id}", owner.id, "gift", 1.0, {"public_profile": True})
    return graph


def fetch_telegram_gifts(user_id: str) -> Graph:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or ":" not in token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    user_id = str(user_id).strip()
    if not user_id.lstrip("-").isdigit():
        raise ValueError("user_id must be a numeric Telegram id")
    gifts: list[dict] = []
    offset = ""
    owner_label = user_id
    for _ in range(TELEGRAM_MAX_PAGES):
        body = json.dumps(
            {
                "user_id": int(user_id),
                "offset": offset,
                "limit": TELEGRAM_PAGE_SIZE,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/getUserGifts",
            data=body,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.loads(response.read(MAX_BYTES + 1).decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram HTTP {error.code}: {detail}") from None
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description") or "Telegram rejected getUserGifts")
        result = payload.get("result") or {}
        page = result.get("gifts") or []
        gifts.extend(page)
        offset = result.get("next_offset") or ""
        if not offset or len(page) < TELEGRAM_PAGE_SIZE:
            break
    return gifts_to_graph(user_id, owner_label, gifts)
