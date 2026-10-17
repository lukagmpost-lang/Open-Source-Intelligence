"""Commitment ledger for WhatsApp chat exports.

Extracts promises and obligations from chat messages using an LLM, stores
them in SQLite, and provides the ledger for later violation detection.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from osi.answer import call_llm

# Bump this when the prompt changes so cached results are invalidated.
PROMPT_VERSION = "v1"
MODEL_NAME = "openai/gpt-oss-120b"
SLEEP_BETWEEN_CALLS = 32
FAILOVER_MODELS = ["qwen/qwen3.8-27b"]

DEFAULT_DB_PATH = Path.home() / ".osi" / "store.db"


# ---------- storage ----------

def init_db(db_path: Path | None = None) -> Path:
    """Create the commitments table if it doesn't exist. Return the path."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS commitments (
            id TEXT PRIMARY KEY,
            speaker TEXT NOT NULL,
            target TEXT,
            obligation TEXT NOT NULL,
            stated_at TEXT,
            deadline TEXT,
            source_index INTEGER NOT NULL,
            strength TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unresolved',
            violated_at TEXT,
            evidence_index INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_commitments_speaker ON commitments(speaker);
        CREATE INDEX IF NOT EXISTS idx_commitments_target ON commitments(target);
        CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status);
    """)
    conn.commit()
    conn.close()
    return path


def save_commitments(commitments: list[dict], db_path: Path | None = None) -> int:
    """Insert or replace commitments. Returns number written."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    n = 0
    for c in commitments:
        conn.execute(
            """INSERT OR REPLACE INTO commitments
               (id, speaker, target, obligation, stated_at, deadline,
                source_index, strength, status, violated_at, evidence_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c["id"], c["speaker"], c.get("target"),
                c["obligation"], c.get("stated_at"), c.get("deadline"),
                c["source_index"], c["strength"],
                c.get("status", "unresolved"),
                c.get("violated_at"), c.get("evidence_index"),
            ),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def load_commitments(db_path: Path | None = None) -> list[dict]:
    """Load all commitments from the database."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM commitments ORDER BY source_index").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- extraction ----------

def _window_hash(messages: list[dict], start: int, end: int) -> str:
    """Hash of a batch for caching. Includes prompt version and model."""
    parts = [PROMPT_VERSION, MODEL_NAME]
    for i in range(start, end):
        m = messages[i]
        parts.append(f"{m.get('sender','')}|{m.get('text','')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _build_prompt(owned, ctx_before, ctx_after, start_index, participants):
    """Build a prompt to extract commitments AND directives."""
    lines = [
        "You are analyzing a segment of a Russian/English group chat.",
        "",
        "Extract TWO kinds of statements:",
        "",
        "1. PERSONAL COMMITMENT: the SPEAKER promises to do something",
        "   themselves. Examples: 'я пришлю', 'I will call', 'will do',",
        "   'беру на себя', 'I will be there', 'step one I will contact',",
        "   'will keep a watch', 'I will speak to X'.",
        "   Short acknowledgments count: 'ok will do', 'will do',",
        "   'sure, I will handle it' - only if responding to a task.",
        "",
        "2. DIRECTIVE: the SPEAKER tells SOMEONE ELSE to do something.",
        "   Examples: 'отправьте', 'please call', 'make sure',",
        "   'kindly change', 'Please help', 'надо сделать',",
        "   '@name Please ...'. Includes @-mentions with a request.",
        "",
        "DO NOT extract:",
        "  - Collective plans: 'мы планируем', 'мы будем'",
        "  - Proposals: 'давайте сделаем'",
        "  - Votes/agreements: 'согласен', 'agree', 'D411-agree'",
        "  - Opinions: 'поддерживаю', 'мне нравится'",
        "  - Thanks/praise: 'спасибо', 'молодец'",
        "  - Past actions: 'я сделал', 'I spoke to X'",
        "",
        "For each statement return:",
        "  type: 'commitment' or 'directive'",
        "  idx: absolute message index",
        "  target: for commitment, recipient ('group' if whole chat);",
        "          for directive, the directed person (a participant)",
        "  action: 2-8 word phrase describing what should happen",
        "  deadline: ISO date if stated, else null",
        "  strength: 'strong' (definite) or 'medium' (soft intent)",
        "",
        "PARTICIPANTS:",
        ", ".join(participants),
        "",
    ]
    if ctx_before:
        lines.append("CONTEXT BEFORE (do not extract):")
        for m in ctx_before:
            t = m.get("text", "").replace(chr(10), " ")[:150]
            lines.append(f"  {m.get('sender', '?')}: {t}")
        lines.append("")
    lines.append(f"MESSAGES (indices {start_index} to {start_index + len(owned) - 1}):")
    for i, m in enumerate(owned):
        sender = m.get("sender", "?")
        t = m.get("text", "").replace(chr(10), " ")
        lines.append(f"{start_index + i}: {sender}: {t}")
    if ctx_after:
        lines.append("")
        lines.append("CONTEXT AFTER (do not extract):")
        for m in ctx_after:
            t = m.get("text", "").replace(chr(10), " ")[:150]
            lines.append(f"  {m.get('sender', '?')}: {t}")
    lines.extend([
        "",
        "Return ONLY a JSON array. Each element:",
        '{"type": "commitment" or "directive", "idx": int, "target": "name", "action": "short phrase", "deadline": null or ISO date, "strength": "strong" or "medium"}',
        "",
        "If nothing, return []. Be thorough - catch all real commitments and directives.",
    ])
    return chr(10).join(lines)


def _normalize_target(raw: str | None, participants: list[str]) -> str | None:
    """Match a raw target to a participant name (casefolded)."""
    if not raw:
        return None
    if raw.strip().lower() == "group":
        return "group"
    raw_cf = raw.strip().casefold()
    for p in participants:
        if p.casefold() == raw_cf:
            return p
    # partial match: first name
    for p in participants:
        first = p.split()[0].casefold()
        if raw_cf == first or raw_cf.startswith(first) or first.startswith(raw_cf):
            return p
    return None


def _extract_batch(window: list[dict], participants: list[str]) -> list[dict]:
    """Run the LLM on one batch and return commitment dicts (idx-relative)."""
    prompt = _build_prompt(window, participants)
    try:
        reply = call_llm(prompt, temperature=0.0)
    except Exception as e:
        print(f"[commitments] LLM call failed: {e}", file=sys.stderr)
        return []
    reply = reply.strip()
    # Strip markdown fences if present
    if reply.startswith("```"):
        reply = reply.split("```")[1]
        if reply.startswith("json"):
            reply = reply[4:]
        reply = reply.strip()
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        # try to find the first array
        start = reply.find("[")
        end = reply.rfind("]")
        if start == -1 or end == -1:
            print(f"[commitments] bad JSON reply: {reply[:120]}", file=sys.stderr)
            return []
        try:
            parsed = json.loads(reply[start:end+1])
        except Exception:
            print(f"[commitments] unparseable: {reply[:120]}", file=sys.stderr)
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _parse_reply(reply: str) -> list[dict]:
    """Parse the LLM reply into a list of dicts."""
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
        start = reply.find("[")
        end = reply.rfind("]")
        if start == -1 or end == -1:
            print(f"[commitments] bad JSON: {reply[:150]}", file=sys.stderr)
            return []
        try:
            parsed = json.loads(reply[start:end+1])
        except Exception:
            print(f"[commitments] unparseable: {reply[:150]}", file=sys.stderr)
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def extract_commitments(
    messages,
    owned: int = 20,
    context: int = 3,
    limit=None,
):
    """Extract commitments AND directives. Returns a dict with two lists."""
    if not messages:
        return {"commitments": [], "directives": []}

    participants = sorted({m.get("sender", "") for m in messages if m.get("sender")})
    max_i = len(messages) if limit is None else min(limit, len(messages))

    commitments = []
    directives = []
    batch_num = 0

    for start in range(0, max_i, owned):
        end = min(start + owned, max_i)
        owned_block = messages[start:end]
        ctx_before = messages[max(0, start - context):start]
        ctx_after = messages[end:min(max_i, end + context)]

        batch_num += 1
        if batch_num > 1:
            print(f"[commitments] sleeping {SLEEP_BETWEEN_CALLS}s", file=sys.stderr)
            time.sleep(SLEEP_BETWEEN_CALLS)
        print(f"[commitments] batch {batch_num}: owned {start}-{end}", file=sys.stderr)

        prompt = _build_prompt(owned_block, ctx_before, ctx_after, start, participants)
        items = []
        for attempt in range(len(FAILOVER_MODELS)):
            model = FAILOVER_MODELS[attempt % len(FAILOVER_MODELS)]
            try:
                reply = call_llm(prompt, temperature=0.0, model=model)
            except Exception as e:
                print(f"[commitments] {model} failed: {e}", file=sys.stderr)
                reply = ""
            if reply and reply.strip():
                items = _parse_reply(reply)
                if items:
                    break
            if attempt < len(FAILOVER_MODELS) - 1:
                print(f"[commitments] retrying with different model in 5s", file=sys.stderr)
                time.sleep(5)
        print(f"[commitments] batch {batch_num} returned {len(items)} items", file=sys.stderr)

        for item in items:
            idx = item.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(messages):
                continue
            if idx < start or idx >= end:
                continue
            msg = messages[idx]
            kind = (item.get("type") or "").strip().lower()
            if kind not in ("commitment", "directive"):
                continue
            # Reject offers and votes disguised as commitments
            action_lc = (item.get("action") or item.get("obligation") or "").lower()
            reject_markers = [
                "support materially", "поддержать материально",
                "sign up", "ready to sign", "agree",
                "голосовать", "согласны",
            ]
            if any(m in action_lc for m in reject_markers):
                continue
            strength = (item.get("strength") or "").strip().lower()
            if strength not in ("strong", "medium"):
                strength = "medium"
            action = (item.get("action") or item.get("obligation") or "").strip()
            if not action:
                continue
            target = _normalize_target(item.get("target"), participants)
            speaker = msg.get("sender", "")
            if target == speaker:
                target = None
            entry = {
                "id": f"{'c' if kind == 'commitment' else 'd'}_{idx:06d}",
                "speaker": speaker,
                "target": target,
                "action": action,
                "stated_at": msg.get("date").isoformat() if hasattr(msg.get("date"), "isoformat") else str(msg.get("date")),
                "deadline": item.get("deadline"),
                "source_index": idx,
                "strength": strength,
                "status": "unresolved",
            }
            if kind == "commitment":
                commitments.append(entry)
            else:
                directives.append(entry)

    def _dedupe(items):
        out = []
        for item in sorted(items, key=lambda x: x["source_index"]):
            dup = False
            for kept in out:
                if kept["speaker"] != item["speaker"]:
                    continue
                if abs(kept["source_index"] - item["source_index"]) <= 3:
                    if kept["action"].lower() == item["action"].lower():
                        dup = True
                        break
            if not dup:
                out.append(item)
        return out

    return {
        "commitments": _dedupe(commitments),
        "directives": _dedupe(directives),
    }


_VIOLATION_PROMPT = """You are checking whether a commitment was KEPT or BROKEN.

Commitment:
  Speaker: {speaker}
  Target: {target}
  Obligation: {obligation}
  Stated at: {stated_at}

Below are messages that came AFTER this commitment. Look for:
  - FULFILLED evidence: the thing was done, thanked for, confirmed
  - VIOLATED evidence: someone complains it was missed, or later messages
    reference it as still outstanding
  - UNRESOLVED: no clear signal either way

Reply with ONLY a JSON object, nothing else:
{{"status": "fulfilled" | "violated" | "unresolved",
  "evidence_idx": <int or null>,
  "reason": "<one sentence>"}}

MESSAGES:
{messages}
"""


def _format_window(block, start_idx):
    lines = []
    for i, m in enumerate(block):
        sender = m.get("sender", "?")
        text = (m.get("text") or "").replace("\n", " ")[:200]
        lines.append(str(start_idx + i) + ": " + sender + ": " + text)
    return "\n".join(lines)


def _parse_violation(reply):
    import json as _json
    text = (reply or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    try:
        return _json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            return {}
        try:
            return _json.loads(text[s:e+1])
        except Exception:
            return {}


def detect_violations(commitments, messages, window=400, sleep_seconds=20):
    """For each commitment, look forward for evidence it was kept or broken.

    Evidence is recorded with its source index so the gate can later filter
    on 'was there a violation BEFORE this message'.
    """
    import time as _time
    updated = []
    for i, c in enumerate(commitments):
        stated = c.get("source_index", 0)
        slice_start = stated + 1
        slice_end = min(len(messages), stated + window)
        if slice_start >= slice_end:
            c["status"] = c.get("status") or "unresolved"
            updated.append(c)
            continue

        if i > 0 and sleep_seconds:
            print("[ledger] sleeping " + str(sleep_seconds) + "s", file=sys.stderr)
            _time.sleep(sleep_seconds)

        prompt = _VIOLATION_PROMPT.format(
            speaker=c.get("speaker", ""),
            target=c.get("target") or "the group",
            obligation=c.get("obligation", ""),
            stated_at=c.get("stated_at", ""),
            messages=_format_window(messages[slice_start:slice_end], slice_start),
        )
        print("[ledger] checking commitment " + str(i+1) + "/" + str(len(commitments)) +
              " (" + c.get("speaker", "") + " -> " + (c.get("obligation") or "")[:40] + ")",
              file=sys.stderr)

        try:
            reply = call_llm(prompt, temperature=0.0)
        except Exception as e:
            print("[ledger] LLM error: " + str(e), file=sys.stderr)
            c["status"] = c.get("status") or "unresolved"
            updated.append(c)
            continue

        result = _parse_violation(reply)
        status = (result.get("status") or "unresolved").lower()
        if status not in ("fulfilled", "violated", "unresolved"):
            status = "unresolved"
        c["status"] = status
        ev = result.get("evidence_idx")
        if isinstance(ev, int) and 0 <= ev < len(messages):
            c["evidence_index"] = ev
            msg = messages[ev]
            c["violated_at"] = msg.get("date").isoformat() if hasattr(msg.get("date"), "isoformat") else str(msg.get("date"))
        c["violation_reason"] = (result.get("reason") or "")[:200]
        updated.append(c)

    return updated


def commitment_summary(db_path=None):
    """Aggregate: top speakers, top targets, status counts."""
    rows = load_commitments(db_path)
    from collections import Counter
    by_speaker = Counter(r["speaker"] for r in rows)
    by_target = Counter(r.get("target") for r in rows if r.get("target") and r.get("target") != "group")
    status = Counter(r.get("status", "unresolved") for r in rows)
    violated = [r for r in rows if r.get("status") == "violated"]
    return {
        "total": len(rows),
        "by_status": dict(status),
        "top_speakers": by_speaker.most_common(10),
        "top_targets": by_target.most_common(10),
        "violated_examples": [
            {"speaker": r["speaker"], "obligation": r["obligation"],
             "reason": r.get("violation_reason", "")[:120]}
            for r in violated[:5]
        ],
    }


# ---------- self-test ----------

if __name__ == "__main__":
    from osi.loaders.whatsapp import _parse_export

    chat_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/2950639/Downloads/_chat.txt"
    limit_arg = sys.argv[2] if len(sys.argv) > 2 else None
    limit = int(limit_arg) if limit_arg else None

    print(f"Parsing {chat_path}...")
    _, messages = _parse_export(chat_path)
    print(f"Total messages: {len(messages)}")
    if limit:
        print(f"Limiting to first {limit} messages.")
    print()

    result = extract_commitments(messages, limit=limit)
    commitments = result["commitments"]
    directives = result["directives"]

    print()
    print(f"COMMITMENTS: {len(commitments)}")
    for c in commitments:
        date = c["stated_at"][:10] if c["stated_at"] else "?"
        target = c["target"] or "?"
        print(f"  [{c['source_index']}] {date} | {c['speaker']} -> {target} | {c['strength']}")
        print(f"      {c['action']}")

    print()
    print(f"DIRECTIVES: {len(directives)}")
    for d in directives:
        date = d["stated_at"][:10] if d["stated_at"] else "?"
        target = d["target"] or "?"
        print(f"  [{d['source_index']}] {date} | {d['speaker']} -> {target} | {d['strength']}")
        print(f"      {d['action']}")
