"""Extract framing relationships from chat messages."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

from osi.answer import call_llm

DEFAULT_DB = Path.home() / ".osi" / "store.db"

_PROMPT = """Here are messages from a Russian/English community chat.

A FRAMING is a message where the speaker describes or evaluates ANOTHER
person (or group) to an audience, in a way that shapes how the audience
sees the other person. Not every message is a framing.

EXAMPLES of framings:
- "Some residents are making this difficult" -> frames unnamed residents negatively
- "Наташа, вы не правы в этом вопросе" -> frames Наташа negatively
- "Спасибо Стелле за организацию" -> frames Стелла positively
- "Я не понимаю, почему X так делает" -> frames X as unreasonable

NOT framings:
- Logistics questions
- Direct task requests (@Name please do X)
- Factual statements without evaluation

For each framing, return:
- idx: message index (integer, from the list below)
- about: person or people characterized. Use an EXACT participant name, or "unnamed residents".
- audience: "group" if to the whole chat, or an @mentioned participant name.
- valence: "negative" | "positive" | "mixed"
- excerpt: the key phrase (up to 100 chars)

PARTICIPANTS: {participants}

MESSAGES:
{messages}

Reply with ONLY a JSON array. If none, reply [].
Format: [{{"idx": 42, "about": "Name", "audience": "group", "valence": "negative", "excerpt": "..."}}]
"""


def _init_db(db_path=None):
    p = Path(db_path) if db_path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS framings (
            id TEXT PRIMARY KEY,
            from_speaker TEXT NOT NULL,
            about TEXT NOT NULL,
            audience TEXT,
            valence TEXT NOT NULL,
            excerpt TEXT,
            source_index INTEGER NOT NULL,
            stated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_framings_from ON framings(from_speaker);
        CREATE INDEX IF NOT EXISTS idx_framings_about ON framings(about);
    """)
    conn.commit()
    conn.close()
    return p


def _parse(reply):
    reply = reply.strip()
    if reply.startswith("```"):
        parts = reply.split("```")
        if len(parts) >= 2:
            reply = parts[1]
            if reply.startswith("json"):
                reply = reply[4:]
            reply = reply.strip()
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        s, e = reply.find("["), reply.rfind("]")
        if s == -1 or e == -1:
            return []
        try:
            parsed = json.loads(reply[s:e + 1])
        except Exception:
            return []
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def extract_framings(messages, batch=20, limit=None, start=0, sleep=20):
    participants = sorted({m.get("sender", "") for m in messages if m.get("sender")})
    participants_str = ", ".join(participants)
    max_i = len(messages) if limit is None else min(limit, len(messages))

    out = []
    batch_num = 0
    for s in range(start, max_i, batch):
        e = min(s + batch, max_i)
        block = messages[s:e]
        batch_num += 1
        if batch_num > 1 and sleep:
            print("[framing] sleeping " + str(sleep) + "s", file=sys.stderr)
            time.sleep(sleep)
        print("[framing] batch " + str(batch_num) + ": " + str(s) + "-" + str(e), file=sys.stderr)

        lines = []
        for i, m in enumerate(block):
            sender = m.get("sender", "?")
            text = (m.get("text") or "").replace("\n", " ")[:250]
            lines.append(str(s + i) + ": " + sender + ": " + text)
        prompt = _PROMPT.format(participants=participants_str, messages="\n".join(lines))

        try:
            reply = call_llm(prompt, temperature=0.0)
        except Exception as ex:
            print("[framing] LLM error: " + str(ex), file=sys.stderr)
            continue

        items = _parse(reply)
        print("[framing] batch " + str(batch_num) + " returned " + str(len(items)), file=sys.stderr)
        for item in items:
            idx = item.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(messages):
                continue
            msg = messages[idx]
            about = (item.get("about") or "").strip()
            if not about:
                continue
            valence = (item.get("valence") or "neutral").strip().lower()
            if valence not in ("negative", "positive", "mixed"):
                valence = "neutral"
            out.append({
                "id": "f_" + str(idx).zfill(6) + "_" + str(abs(hash(about)) % 10000).zfill(4),
                "from_speaker": msg.get("sender", ""),
                "about": about,
                "audience": (item.get("audience") or "group").strip(),
                "valence": valence,
                "excerpt": (item.get("excerpt") or "")[:120],
                "source_index": idx,
                "stated_at": msg.get("date").isoformat() if hasattr(msg.get("date"), "isoformat") else str(msg.get("date")),
            })
    return out


def save_framings(rows, db_path=None):
    p = _init_db(db_path)
    conn = sqlite3.connect(p)
    n = 0
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO framings (id, from_speaker, about, audience, valence, excerpt, source_index, stated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r["id"], r["from_speaker"], r["about"], r.get("audience"),
             r["valence"], r.get("excerpt"), r["source_index"], r.get("stated_at")),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def load_framings(db_path=None):
    p = Path(db_path) if db_path else DEFAULT_DB
    if not p.exists():
        return []
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM framings ORDER BY source_index").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def framing_summary(db_path=None):
    rows = load_framings(db_path)
    from collections import Counter
    by_from = Counter()
    by_about = Counter()
    neg_from = Counter()
    neg_about = Counter()
    pairs = Counter()
    for r in rows:
        by_from[r["from_speaker"]] += 1
        by_about[r["about"]] += 1
        if r["valence"] == "negative":
            neg_from[r["from_speaker"]] += 1
            neg_about[r["about"]] += 1
        if r["from_speaker"] != r["about"]:
            pairs[(r["from_speaker"], r["about"])] += 1
    return {
        "total": len(rows),
        "top_framers": by_from.most_common(10),
        "top_framed": by_about.most_common(10),
        "top_negative_framers": neg_from.most_common(10),
        "top_negative_framed": neg_about.most_common(10),
        "top_pairs": [{"from": a, "about": b, "count": c} for (a, b), c in pairs.most_common(20)],
    }


if __name__ == "__main__":
    from osi.loaders.whatsapp import _parse_export
    chat = "/Users/2950639/Downloads/_chat.txt"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("parsing " + chat + "...")
    _, messages = _parse_export(chat)
    print("total messages: " + str(len(messages)))
    print("extracting framings from " + str(start) + " to " + str(start + limit) + "...")
    rows = extract_framings(messages, batch=20, start=start, limit=start + limit)
    n = save_framings(rows)
    print("saved " + str(n) + " framings")
    summary = framing_summary()
    print()
    print("total:", summary["total"])
    print("TOP FRAMERS:", summary["top_framers"])
    print("TOP FRAMED:", summary["top_framed"])
    print("TOP NEGATIVE FRAMERS:", summary["top_negative_framers"])
    print("TOP NEGATIVE FRAMED:", summary["top_negative_framed"])
