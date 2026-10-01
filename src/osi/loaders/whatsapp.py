"""WhatsApp plain-text chat export loader for OS-INT."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx


_DATE = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} [APap][Mm]"
_TIMESTAMP_PREFIX = re.compile(
    rf"^\[(?P<bracket_date>{_DATE})\]\s*(?P<bracket_body>.*)$|"
    rf"^(?P<plain_date>{_DATE})\s+-\s+(?P<plain_body>.*)$"
)
_MENTION = re.compile(r"@([\w]+)", re.UNICODE)
_EDGE_RULES = {"temporal", "sequence", "mention"}


def _parse_date(value: str) -> datetime | None:
    for date_format in ("%m/%d/%y, %I:%M:%S %p", "%m/%d/%Y, %I:%M:%S %p"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            pass
    return None


def _is_system_message(sender: str | None, text: str) -> bool:
    lowered = text.strip().casefold()
    if not lowered:
        return sender is None
    if any(
        phrase in lowered
        for phrase in (
            "messages and calls are end-to-end encrypted",
            "<media omitted>",
            "this message was deleted",
        )
    ):
        return True
    if sender is None:
        return True
    return any(
        re.fullmatch(pattern, lowered)
        for pattern in (
            r".+ added .+",
            r".+ left",
            r".+ changed the subject(?: .*)?",
            r".+ changed this group's icon",
        )
    )


def _parse_export(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    first_line = next((line.strip() for line in raw_lines if line.strip()), "")
    chat_name = path.stem
    if first_line and not _TIMESTAMP_PREFIX.match(first_line):
        chat_name = first_line

    parsed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw_lines:
        match = _TIMESTAMP_PREFIX.match(line)
        if match:
            if current is not None:
                parsed.append(current)
            date_text = match.group("bracket_date") or match.group("plain_date")
            body = match.group("bracket_body") or match.group("plain_body") or ""
            date = _parse_date(date_text)
            sender: str | None = None
            text = body
            if ":" in body:
                possible_sender, text = body.split(":", 1)
                if possible_sender.strip():
                    sender = possible_sender.strip()
            current = {"date": date, "sender": sender, "text": text.strip()}
        elif current is not None:
            continuation = line.strip()
            if continuation:
                current["text"] = f"{current['text']}\n{continuation}" if current["text"] else continuation
    if current is not None:
        parsed.append(current)

    messages = [
        item
        for item in parsed
        if item["date"] is not None
        and item["sender"]
        and not _is_system_message(item["sender"], item["text"])
    ]
    return chat_name, messages


def _add_edge(graph: nx.Graph, left: str, right: str) -> None:
    if left == right:
        return
    if graph.has_edge(left, right):
        graph[left][right]["weight"] += 1
    else:
        graph.add_edge(left, right, weight=1)


def _add_edges(graph: nx.Graph, messages: list[dict], edge_rule: str, window_minutes: float) -> None:
    ordered = sorted(messages, key=lambda item: item["date"])
    if edge_rule == "sequence":
        for previous, current in zip(ordered, ordered[1:]):
            _add_edge(graph, previous["sender"], current["sender"])
        return
    if edge_rule == "mention":
        aliases: dict[str, set[str]] = {}
        for message in ordered:
            sender = message["sender"]
            aliases.setdefault(sender.casefold().lstrip("@"), set()).add(sender)
            for token in re.findall(r"[\w]+", sender, re.UNICODE):
                aliases.setdefault(token.casefold(), set()).add(sender)
        for message in ordered:
            for match in _MENTION.finditer(message["text"]):
                senders = aliases.get(match.group(1).casefold(), set())
                if len(senders) == 1:
                    _add_edge(graph, message["sender"], next(iter(senders)))
        return

    window = timedelta(minutes=window_minutes)
    start = 0
    for index, message in enumerate(ordered):
        while message["date"] - ordered[start]["date"] > window:
            start += 1
        for nearby in ordered[start:index]:
            _add_edge(graph, message["sender"], nearby["sender"])


def load_whatsapp_export(
    path: str | Path,
    edge_rule: str = "temporal",
    window_minutes: float = 5,
    chat_cap: int = 30,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Load a WhatsApp chat export .txt into a NetworkX graph.

    Supports both timestamp-prefix styles commonly produced by WhatsApp.
    A chat with more than ``chat_cap`` unique senders keeps its nodes and
    message counts but does not produce edges.
    """
    if edge_rule not in _EDGE_RULES:
        raise ValueError(f"Unknown edge rule: {edge_rule}")
    if window_minutes < 0:
        raise ValueError("window_minutes must be non-negative")
    if chat_cap < 1:
        raise ValueError("chat_cap must be at least 1")

    export_path = Path(path).expanduser()
    chat_name, messages = _parse_export(export_path)
    graph = nx.Graph()
    for message in messages:
        sender = message["sender"]
        if graph.has_node(sender):
            graph.nodes[sender]["message_count"] += 1
        else:
            graph.add_node(sender, message_count=1)

    if graph.number_of_nodes() <= chat_cap:
        _add_edges(graph, messages, edge_rule, window_minutes)

    dates = [message["date"] for message in messages]
    metadata = {
        "source": "whatsapp",
        "chat_name": chat_name,
        "member_count": graph.number_of_nodes(),
        "message_count": len(messages),
        "date_range": {
            "first": min(dates).isoformat(timespec="seconds") if dates else None,
            "last": max(dates).isoformat(timespec="seconds") if dates else None,
        },
    }
    return graph, metadata


def detect_whatsapp_format(path: str | Path) -> str:
    """Return high, medium, or low confidence that a text file is a WhatsApp export."""
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.suffix.casefold() != ".txt":
        return "low"
    try:
        lines = candidate.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return "low"
    timestamps = [match for line in lines if (match := _TIMESTAMP_PREFIX.match(line))]
    if not timestamps:
        return "low"
    if any(":" in (match.group("bracket_body") or match.group("plain_body") or "") for match in timestamps):
        return "high"
    return "medium"