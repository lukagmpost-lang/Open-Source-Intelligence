from __future__ import annotations

"""Irony detection pipeline for WhatsApp chat exports.

This module provides functions to widen candidate messages for irony detection,
classify them using an LLM with context, and run a full pipeline from chat export to JSON output.
"""

from osi.answer import call_llm
from osi.loaders.whatsapp import _parse_export
from osi.sentiment import score as sentiment_score
import json
import re
from collections import defaultdict
# Widening markers: words/phrases that suggest irony or violated expectation
WIDENING_MARKERS = [
    # Russian
    "спасибо", "конечно", "ну", "опять", "так и", "finally", "отлично",
    "прекрасно", "замечательно", "молодец", "как всегда", "ясно", "понятно",
    "серьезно", "да ладно", "ничего себе", "ну да", "ага", "разумеется",
    "прикольно", "здорово", "великолепно", "фантастично", "супер", "лучше некуда",
    "как же", "вот поворот", "неожиданно", "удивительно", "как всегда", "ну конечно",
    # English
    "thanks", "sure", "of course", "again", "finally", "great", "wonderful",
    "perfect", "nice", "as always", "right", "ok", "really", "absolutely",
    "definitely", "exactly", "precisely", "certainly", "naturally", "obviously",
    "certainly", "surely", "indeed", "absolutely", "positively", "undoubtedly",
    "without a doubt", "no doubt", "by all means", "for sure", "you bet"
]

# Negation markers: words that suggest a violation
NEGATION_MARKERS = [
    # Russian
    "не", "нет", "никогда", "больше не", "так и не", "опять не",
    # English
    "didn't", "don't", "won't", "never", "still haven't", "again didn't",
    "hasn't", "hadn't", "wouldn't", "couldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "doesn't", "don't", "did not", "do not", "does not",
    "has not", "have not", "had not", "will not", "would not", "could not",
    "should not", "is not", "are not", "was not", "were not"
]

# Irony emojis and markers
IRONY_EMOJI = ["🙃", "😏", "/s", "jk"]

# Scare quote detector: a phrase in quotes preceded by "so-called", "якобы", "так называемый"
SCARE_QUOTE_PATTERNS = [
    r'\bso-called\s+["\'][^"\']+["\']',
    r'\bякобы\s+["\'][^"\']+["\']',
    r'\bтак называемый\s+["\'][^"\']+["\']'
]
def widen_vocabulary(messages: list[dict]) -> set[int]:
    """Widen by vocabulary: WIDENING_MARKERS, NEGATION_MARKERS, IRONY_EMOJI, scare quotes."""
    indices = set()
    for i, msg in enumerate(messages):
        text = msg.get("text", "").lower()
        # Check widening markers
        if any(marker.lower() in text for marker in WIDENING_MARKERS):
            indices.add(i)
            continue
        # Check negation markers
        if any(marker.lower() in text for marker in NEGATION_MARKERS):
            indices.add(i)
            continue
        # Check irony emojis
        if any(emoji in msg.get("text", "") for emoji in IRONY_EMOJI):
            indices.add(i)
            continue
        # Check scare quotes
        for pattern in SCARE_QUOTE_PATTERNS:
            if re.search(pattern, msg.get("text", ""), re.IGNORECASE):
                indices.add(i)
                break
    return indices
def widen_echo(messages: list[dict]) -> set[int]:
    """Widen by echo: 6+ character substring repeated in recent history."""
    indices = set()
    for i in range(1, len(messages)):
        text = messages[i].get("text", "")
        if len(text) < 10:
            continue
        # Look for substrings of length 6+ in the current message
        found = False
        for start in range(len(text) - 5):
            for length in range(6, min(20, len(text) - start) + 1):
                substr = text[start:start+length]
                # Check in the previous 3 messages (if they exist)
                for j in range(max(0, i-3), i):
                    prev_text = messages[j].get("text", "")
                    if substr in prev_text:
                        indices.add(i)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return indices
def widen_ledger_hits(messages: list[dict], commitments: list[dict]) -> set[int]:
    """Widen by ledger hits: mentions speaker/target of violated commitment with obligation keyword."""
    if not commitments:
        return set()
    indices = set()
    # Precompute violated commitments
    violated = [c for c in commitments if c.get("violated", False)]
    for i, msg in enumerate(messages):
        text = msg.get("text", "").lower()
        sender = msg.get("sender", "").lower()
        for c in violated:
            speaker = c.get("speaker", "").lower()
            target = c.get("target", "").lower()
            obligation = c.get("obligation", "").lower()
            # Check if message mentions speaker or target
            if speaker in text or target in text:
                # Check for obligation keyword (simple: any word from obligation in text)
                if obligation and any(word in text for word in obligation.split()):
                    indices.add(i)
                    break
    return indices
def widen_by_baseline(messages: list[dict], profiles: dict) -> set[int]:
    """Widen by baseline: speakers with high irony_rate."""
    if not profiles:
        return set()
    indices = set()
    for i, msg in enumerate(messages):
        sender = msg.get("sender", "")
        if sender in profiles and profiles[sender].get("irony_rate", 0.0) >= 0.3:
            indices.add(i)
    return indices
def widen_candidates(messages: list[dict], commitments=None, profiles=None) -> list[dict]:
    """Combine all widening signals into a union of indices with reasons."""
    if commitments is None:
        commitments = []
    if profiles is None:
        profiles = {}
    
    vocab = widen_vocabulary(messages)
    echo = widen_echo(messages)
    ledger = widen_ledger_hits(messages, commitments)
    baseline = widen_by_baseline(messages, profiles)
    
    all_indices = vocab | echo | ledger | baseline
    candidates = []
    for i in all_indices:
        msg = messages[i]
        reasons = []
        if i in vocab:
            reasons.append("vocabulary")
        if i in echo:
            reasons.append("echo")
        if i in ledger:
            reasons.append("ledger")
        if i in baseline:
            reasons.append("baseline")
        candidates.append({
            "index": i,
            "sender": msg.get("sender", ""),
            "text": msg.get("text", ""),
            "date": msg.get("date"),
            "widened_by": reasons
        })
    return candidates
def build_context(messages: list[dict], i: int, commitments=None, rank_gap=None, profiles=None) -> dict:
    """Assemble context for message index i."""
    if commitments is None:
        commitments = []
    if profiles is None:
        profiles = {}
    
    msg = messages[i]
    context = {
        "message": {
            "sender": msg.get("sender", ""),
            "text": msg.get("text", ""),
            "date": msg.get("date")
        },
        "previous_turns": [],
        "violated_commitments": [],
        "rank_gap": rank_gap,
        "speaker_baseline": profiles.get(msg.get("sender", ""), {}).get("irony_rate")
    }
    # Previous 2 messages (if they exist)
    for j in range(max(0, i-2), i):
        prev = messages[j]
        context["previous_turns"].append({
            "sender": prev.get("sender", ""),
            "text": prev.get("text", ""),
            "date": prev.get("date")
        })
    # Viable commitments involving this sender or target
    sender = msg.get("sender", "")
    for c in commitments:
        if not c.get("violated", False):
            continue
        if c.get("speaker", "") == sender or c.get("target", "") == sender:
            context["violated_commitments"].append(
                f"{c.get('speaker', '')} -> {c.get('target', '')}: {c.get('obligation', '')}"
            )
    return context
def _safe_json_parse(text: str):
    """Try to parse JSON, fallback to extracting first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None
def classify_candidates(candidates: list[dict], messages: list[dict], commitments=None, 
                       rank_gap_fn=None, profiles=None, batch_size=5) -> list[dict]:
    """Classify candidates using LLM with context, one message at a time."""
    if not candidates:
        return []
    if commitments is None:
        commitments = []
    if profiles is None:
        profiles = {}
    
    results = []
    # Process in batches (for progress reporting, but each candidate gets its own prompt)
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        for j, cand in enumerate(batch):
            idx = cand["index"]
            # Build context for this candidate
            rank_gap = rank_gap_fn(idx, messages) if rank_gap_fn else None
            context = build_context(messages, idx, commitments, rank_gap, profiles)
            
            # Build prompt for single message classification
            prompt_lines = [
                "You are analyzing a single message from a group chat.",
                "",
                "[context: message, previous 2 turns, violated commitments, rank gap, speaker baseline]",
                f"Message: {context['message']['sender']}: {context['message']['text']}",
                ""
            ]
            if context["previous_turns"]:
                prompt_lines.append("Previous turns:")
                for turn in context["previous_turns"]:
                    prompt_lines.append(f"  {turn['sender']}: {turn['text']}")
                prompt_lines.append("")
            if context["violated_commitments"]:
                prompt_lines.append("Violated commitments:")
                for vc in context["violated_commitments"]:
                    prompt_lines.append(f"  {vc}")
                prompt_lines.append("")
            if context["rank_gap"]:
                prompt_lines.append(f"Rank gap: {context['rank_gap']}")
                prompt_lines.append("")
            if context["speaker_baseline"] is not None:
                prompt_lines.append(f"Speaker baseline irony rate: {context['speaker_baseline']}")
                prompt_lines.append("")
            prompt_lines.extend([
                "Apply these steps:",
                "1. In one sentence, summarize what's happening in this conversation.",
                "2. Compare the message's tone to the situation. Does the literal meaning agree or contradict the situation?",
                "3. Is this message sincere, ironic, ambiguous, or insufficient context to tell?",
                "",
                "Reply in JSON:",
                "{",
                '  "summary": "...",',
                '  "contrast": "agrees" | "contradicts" | "unclear",',
                '  "label": "sincere" | "ironic" | "ambiguous" | "insufficient",',
                '  "confidence": 0.0-1.0,',
                '  "assumed_listener_belief": "one sentence describing what the sender assumes the receiver already knows"',
                "}"
            ])
            prompt = "\n".join(prompt_lines)
            
            # Try to get and parse LLM response
            parsed = None
            for attempt in range(2):
                try:
                    response = call_llm(prompt, temperature=0.0)
                    parsed = _safe_json_parse(response)
                    if parsed and isinstance(parsed, dict):
                        # Validate expected keys
                        required = ["summary", "contrast", "label", "confidence", "assumed_listener_belief"]
                        if all(k in parsed for k in required):
                            break
                except Exception:
                    pass
                if attempt == 0:
                    # Simpler prompt on retry
                    prompt = (
                        "Is this message sincere, ironic, ambiguous, or insufficient context? "
                        "Reply with JSON: {\"label\": \"sincere|ironic|ambiguous|insufficient\", "
                        "\"confidence\": 0.0-1.0, \"reason\": \"one sentence\"}. "
                        f"Message: {cand['sender']}: {cand['text']}"
                    )
            
            if parsed and isinstance(parsed, dict):
                # Map to expected output format
                result = {
                    "index": cand["index"],
                    "sender": cand["sender"],
                    "text": cand["text"],
                    "date": cand["date"],
                    "widened_by": cand["widened_by"],
                    "label": parsed.get("label", "insufficient"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": parsed.get("assumed_listener_belief", ""),
                    "summary": parsed.get("summary", ""),
                    "contrast": parsed.get("contrast", "unclear")
                }
            else:
                # Parse failure
                result = {
                    "index": cand["index"],
                    "sender": cand["sender"],
                    "text": cand["text"],
                    "date": cand["date"],
                    "widened_by": cand["widened_by"],
                    "label": "insufficient",
                    "confidence": 0.0,
                    "reason": "parse error",
                    "summary": "",
                    "contrast": "unclear"
                }
            results.append(result)
        
        # Progress reporting every 10 candidates
        if (i + len(batch)) % 10 == 0 or (i + len(batch)) >= len(candidates):
            print(f"[irony] classified {min(i+len(batch), len(candidates))}/{len(candidates)}", file=sys.stderr)
    
    return results
def run_pipeline(chat_path: str, output_path: str, commitments=None, profiles=None) -> dict:
    """Run the full irony detection pipeline on a WhatsApp chat export."""
    if commitments is None:
        commitments = []
    if profiles is None:
        profiles = {}
    
    # Parse the chat export
    _, messages = _parse_export(chat_path)
    # Widen candidates
    widened = widen_candidates(messages, commitments, profiles)
    # Classify candidates
    classified = classify_candidates(widened, messages, commitments, None, profiles)
    # Filter to items where label in ("ironic", "ambiguous") and confidence >= 0.4
    filtered = [c for c in classified if c["label"] in ("ironic", "ambiguous") and c["confidence"] >= 0.4]
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    # Print summary
    ironic_high = len([c for c in filtered if c["label"] == "ironic"])
    ambiguous_high = len([c for c in filtered if c["label"] == "ambiguous"])
    print(f"Widened: {len(widened)} messages. Classified: {len(classified)}. "
          f"Ironic high-confidence: {ironic_high}. Ambiguous high-confidence: {ambiguous_high}. "
          f"Written to {output_path}")
    # Return summary
    return {
        "widened": len(widened),
        "classified": len(classified),
        "ironic_high_conf": ironic_high,
        "ambiguous_high_conf": ambiguous_high
    }
