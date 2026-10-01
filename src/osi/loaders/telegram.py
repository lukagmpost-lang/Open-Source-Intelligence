"""Telegram JSON export loader for OS-INT."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import networkx as nx


_EDGE_RULES = {"reply", "mention", "temporal"}
_MENTION = re.compile(r"@([\w]+)", re.UNICODE)
_TEMPORAL_WINDOW = timedelta(minutes=5)


def _export_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate / "result.json" if candidate.is_dir() else candidate


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item["text"]
            for item in value
            if isinstance(item, dict)
            and item.get("type") == "plain"
            and isinstance(item.get("text"), str)
        )
    return ""


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _key(value: Any) -> str | None:
    return None if value is None else str(value)


def _add_edge(graph: nx.Graph, left: str, right: str) -> None:
    if left == right:
        return
    if graph.has_edge(left, right):
        graph[left][right]["weight"] += 1
    else:
        graph.add_edge(left, right, weight=1)


def _reply_root(message: dict, by_id: dict[str, dict]) -> tuple[str, str]:
    current = message
    seen: set[str] = set()
    while current["reply_to"] is not None:
        parent_key = current["reply_to"]
        parent = by_id.get(parent_key)
        if parent is None:
            return "missing", parent_key
        if parent["message_key"] in seen:
            break
        seen.add(current["message_key"])
        current = parent
    return "message", current["message_key"]


def _reply_edges(graph: nx.Graph, messages: list[dict], chat_cap: int) -> None:
    by_id = {message["message_key"]: message for message in messages if message["has_id"]}
    conversations: dict[tuple[str, str], list[dict]] = {}
    for message in messages:
        root = _reply_root(message, by_id)
        conversations.setdefault(root, []).append(message)

    for conversation in conversations.values():
        if len({message["sender"] for message in conversation}) > chat_cap:
            continue
        for message in conversation:
            parent = by_id.get(message["reply_to"])
            if parent is not None:
                _add_edge(graph, message["sender"], parent["sender"])


def _mention_edges(graph: nx.Graph, messages: list[dict], chat_cap: int) -> None:
    aliases: dict[str, set[str]] = {}
    for message in messages:
        sender = message["sender"]
        for value in (sender, message["name"]):
            alias = value.casefold().lstrip("@")
            if alias:
                aliases.setdefault(alias, set()).add(sender)
        for token in re.findall(r"[\w]+", message["name"], re.UNICODE):
            aliases.setdefault(token.casefold(), set()).add(sender)

    for message in messages:
        participants = {message["sender"]}
        for match in _MENTION.finditer(message["text"]):
            matches = aliases.get(match.group(1).casefold(), set())
            if len(matches) == 1:
                participants.update(matches)
        if len(participants) > chat_cap:
            continue
        participants = sorted(participants)
        for index, left in enumerate(participants):
            for right in participants[index + 1 :]:
                _add_edge(graph, left, right)


def _temporal_edges(graph: nx.Graph, messages: list[dict], chat_cap: int) -> None:
    dated = sorted(
        (message for message in messages if message["date"] is not None),
        key=lambda message: message["date"],
    )
    conversations: list[list[dict]] = []
    for message in dated:
        if (
            not conversations
            or message["date"] - conversations[-1][-1]["date"] > _TEMPORAL_WINDOW
        ):
            conversations.append([])
        conversations[-1].append(message)

    for conversation in conversations:
        if len({message["sender"] for message in conversation}) > chat_cap:
            continue
        start = 0
        for index, message in enumerate(conversation):
            while message["date"] - conversation[start]["date"] > _TEMPORAL_WINDOW:
                start += 1
            nearby_senders = {item["sender"] for item in conversation[start:index]}
            for sender in nearby_senders:
                _add_edge(graph, message["sender"], sender)


def load_telegram_export(
    path: str | Path,
    edge_rule: str = "reply",
    chat_cap: int = 30,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Load a Telegram chat export into a NetworkX graph.

    ``path`` may point to ``result.json`` or to a folder containing it.
    Reply conversations, message mentions, or five-minute temporal windows
    become edges. Conversations with more than ``chat_cap`` senders are skipped.
    """
    if edge_rule not in _EDGE_RULES:
        raise ValueError(f"Unknown edge rule: {edge_rule}")
    if chat_cap < 1:
        raise ValueError("chat_cap must be at least 1")

    export_path = _export_file(path)
    with export_path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError("Telegram export must be an object with a messages list")

    graph = nx.Graph()
    messages: list[dict] = []
    for index, item in enumerate(data["messages"]):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        sender_value = item.get("from_id")
        if sender_value is None or not str(sender_value).strip():
            continue
        name_value = item.get("from")
        name = str(name_value).strip() if name_value is not None else ""
        sender = name or str(sender_value)
        message_id = _key(item.get("id"))
        message = {
            "sender": sender,
            "name": name or str(sender_value),
            "message_key": message_id if message_id is not None else f"missing-id-{index}",
            "has_id": message_id is not None,
            "reply_to": _key(item.get("reply_to_message_id")),
            "text": _message_text(item.get("text")),
            "date": _timestamp(item.get("date")),
        }
        messages.append(message)
        if graph.has_node(sender):
            if graph.nodes[sender]["name"] is None and name:
                graph.nodes[sender]["name"] = name
        else:
            graph.add_node(sender, user_id=sender_value, name=name_value)

    if edge_rule == "reply":
        _reply_edges(graph, messages, chat_cap)
    elif edge_rule == "mention":
        _mention_edges(graph, messages, chat_cap)
    else:
        _temporal_edges(graph, messages, chat_cap)

    raw_type = str(data.get("type", "private")).casefold()
    if "channel" in raw_type:
        chat_type = "channel"
    elif "group" in raw_type or "supergroup" in raw_type:
        chat_type = "group"
    else:
        chat_type = "private"

    metadata = {
        "source": "telegram",
        "chat_title": data.get("name", ""),
        "chat_type": chat_type,
        "member_count": graph.number_of_nodes(),
        "message_count": len(messages),
    }
    return graph, metadata


def detect_telegram_format(path: str | Path) -> str:
    """Return high, medium, or low confidence that a path is a Telegram export."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "result.json"
    try:
        with candidate.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "low"
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return "low"
    required = {"name", "type", "id", "messages"}
    return "high" if required <= data.keys() else "medium"