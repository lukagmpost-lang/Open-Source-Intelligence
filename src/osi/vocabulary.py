"""Source-specific words used in findings and answers."""

from __future__ import annotations

import re
from pathlib import Path


VOCAB = {
    "slack": {
        "person": "employee",
        "people": "employees",
        "group": "team",
        "critical": "single point of failure",
        "bridge": "cross-team contact",
        "isolated": "disconnected employee",
        "hub": "most connected employee",
        "network": "workspace",
        "networks": "workspaces",
    },
    "telegram": {
        "person": "member",
        "people": "members",
        "group": "cell",
        "critical": "central member",
        "bridge": "connector",
        "isolated": "lurker",
        "hub": "most active member",
        "network": "group",
        "networks": "groups",
    },
    "general": {
        "person": "account",
        "people": "accounts",
        "group": "group",
        "critical": "critical node",
        "bridge": "bridge",
        "isolated": "isolated node",
        "hub": "hub",
        "network": "network",
        "networks": "networks",
    },
}


def vocabulary_for(source: str | None) -> dict[str, str]:
    """Return the vocabulary for a source, falling back to general."""
    key = str(source or "").strip().casefold()
    return VOCAB.get(key, VOCAB["general"])


def vocabulary_instruction(source: str | None) -> str:
    """Return the concise source terminology instruction for an answer prompt."""
    vocab = vocabulary_for(source)
    return (
        f"This is a {vocab['network']} network. Refer to people as {vocab['person']}, "
        f"groups as {vocab['group']}."
    )


def get_source(run_id: str) -> str:
    """Resolve a saved run to slack, telegram, csv, or general."""
    try:
        from osi.store import get_run

        metadata = get_run(run_id)
    except Exception:
        return "general"
    if not metadata:
        return "general"

    source = str(metadata.get("source") or "").strip().casefold()
    if source in {"slack", "telegram"}:
        return source
    if source in {"csv", "tsv"}:
        return "csv"
    if source == "file":
        config = metadata.get("config") or {}
        suffix = Path(str(config.get("path") or "")).suffix.casefold()
        if suffix in {".csv", ".tsv"}:
            return "csv"
    return "general"


def _plural(phrase: str) -> str:
    words = phrase.split()
    last = words[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    elif not last.endswith("s"):
        words[-1] = last + "s"
    return " ".join(words)


def apply_vocabulary(text: str, vocab: dict[str, str]) -> str:
    """Replace generic graph terms in one generated sentence."""
    replacements = {
        "isolated accounts": _plural(vocab["isolated"]),
        "isolated account": vocab["isolated"],
        "isolated nodes": _plural(vocab["isolated"]),
        "isolated node": vocab["isolated"],
        "critical nodes": _plural(vocab["critical"]),
        "critical node": vocab["critical"],
        "accounts": vocab["people"],
        "account": vocab["person"],
        "people": vocab["people"],
        "person": vocab["person"],
        "groups": _plural(vocab["group"]),
        "group": vocab["group"],
        "networks": vocab["networks"],
        "network": vocab["network"],
        "hubs": _plural(vocab["hub"]),
        "hub": vocab["hub"],
        "bridges": _plural(vocab["bridge"]),
        "bridge": vocab["bridge"],
    }
    ordered = sorted(replacements, key=len, reverse=True)
    pattern = re.compile(rf"\b(?:{'|'.join(re.escape(word) for word in ordered)})\b", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        value = replacements[match.group(0).casefold()]
        if match.group(0)[0].isupper():
            return value[0].upper() + value[1:]
        return value

    return pattern.sub(replace, text)