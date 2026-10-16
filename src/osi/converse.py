"""Conversation loop with code execution. Prompt-based tool calling."""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import re
import signal

import networkx as nx

from osi.answer import call_llm


_SYSTEM = """You analyze a chat network by writing Python. Reply in JSON only.

Objects available in your code:
- g: networkx.Graph. Nodes are accounts. Edge attrs: weight, positive_count,
  negative_count, message_count, llm_label.
- messages: list of dicts. Each has date (datetime), sender (str), text (str),
  sentiment (float, -1 to 1).
- framings: list of dicts with keys from_speaker, about, audience, valence,
  excerpt, source_index. Precomputed: messages where one person characterizes
  another to an audience. USE THIS for questions about positioning, blame,
  reputation, or who turns whom against whom.
- irony_candidates: list of dicts with keys index, sender, text, label,
  confidence, reason. Precomputed: messages already flagged as possibly
  ironic. USE THIS for questions about sarcasm, passive aggression, hidden
  hostility.

Allowed imports: networkx (as nx), math, statistics, collections, itertools,
re, json, datetime.

You work iteratively. On each turn, reply with a JSON object, nothing else:

To run code:
{"action": "run_python", "code": "print(g.number_of_nodes())"}

To finish and answer:
{"action": "answer", "answer": "Your 2-5 sentence answer here."}

Use print() in your code to inspect. When you have enough evidence,
switch to the answer action.
"""

_ALLOWED = {"networkx", "math", "statistics", "collections", "itertools",
            "re", "json", "datetime"}


def _safe_import(name, *a, **kw):
    root = name.split(".")[0]
    if root in _ALLOWED:
        return __import__(name, *a, **kw)
    raise ImportError("import not allowed: " + name)


def _restricted_builtins():
    safe = dict(vars(builtins))
    for n in ("open", "exec", "eval", "compile", "input", "__import__"):
        safe.pop(n, None)
    safe["__import__"] = _safe_import
    return safe


def _run(code, graph, messages, extras=None, timeout=20.0):
    buf = io.StringIO()
    extras = extras or {}
    ns = {"__builtins__": _restricted_builtins(), "g": graph,
          "messages": messages, "nx": nx, "result": None}
    ns.update(extras)

    def _to(signum, frame):
        raise TimeoutError("timeout")

    old = signal.signal(signal.SIGALRM, _to)
    signal.alarm(int(timeout))
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception as e:
        return {"ok": False, "output": buf.getvalue()[:3000],
                "error": type(e).__name__ + ": " + str(e)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    out = buf.getvalue()[:3000]
    res = ns.get("result")
    res_str = None
    if res is not None:
        try:
            res_str = json.dumps(res, default=str)[:3000]
        except Exception:
            res_str = str(res)[:3000]
    return {"ok": True, "output": out, "result": res_str}


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_action(reply):
    text = (reply or "").strip()
    # strip code fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    m = _JSON.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_RESOLVE_PROMPT = """Below is a conversation and a new question. Rewrite the new
question so that any pronouns (they, them, those, these, he, she, it) are
replaced with the specific people named in the previous assistant answer.

If the new question has no pronouns, return it unchanged.
If you cannot determine who the pronouns refer to, return it unchanged.

Return ONLY the rewritten question, nothing else.

CONVERSATION:
{conversation}

NEW QUESTION: {question}

REWRITTEN:"""


def _resolve_question(question, history):
    """If the question has pronouns, rewrite it using the prior conversation."""
    if not history:
        return question
    # Build a short view of recent turns
    turns = []
    for h in history[-4:]:
        role = h.get("role", "?")
        content = (h.get("content") or "")[:600]
        turns.append(role.upper() + ": " + content)
    conversation_text = "\n\n".join(turns)
    prompt = _RESOLVE_PROMPT.format(conversation=conversation_text, question=question)
    try:
        rewritten = call_llm(prompt, temperature=0.0).strip()
    except Exception:
        return question
    # Sanity: don't accept an empty or absurdly long response
    if not rewritten or len(rewritten) > 400:
        return question
    # Strip quotes if the model wrapped the answer
    rewritten = rewritten.strip('"\'' )
    return rewritten


def _load_extras():
    extras = {"framings": [], "irony_candidates": []}
    try:
        from osi.framing import load_framings
        extras["framings"] = load_framings()
    except Exception:
        pass
    try:
        import json
        from pathlib import Path
        p = Path.home() / "Downloads" / "irony_candidates.json"
        if p.exists():
            extras["irony_candidates"] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return extras


def converse(question, graph, messages, max_rounds=4, verbose=False, history=None):
    extras = _load_extras()
    prior = history if history is not None else []
    original_question = question
    if prior:
        question = _resolve_question(question, prior)
        if question != original_question:
            print("[resolved] " + question)
    conversation = [{"role": "system", "content": _SYSTEM}]
    for h in prior[-6:]:
        conversation.append(h)
    conversation.append({"role": "user", "content": question})
    history = conversation
    code_history = []

    for round_num in range(max_rounds):
        # Cap history: keep system + original question + last 2 exchanges
        trimmed = [history[0], history[1]] + history[2:][-4:]
        prompt_parts = []
        for h in trimmed:
            prompt_parts.append(h["role"].upper() + ": " + h["content"])
        prompt = "\n\n".join(prompt_parts)
        # Hard cap prompt length
        if len(prompt) > 12000:
            prompt = prompt[-12000:]

        try:
            reply = call_llm(prompt, system=_SYSTEM, temperature=0.0)
        except Exception as e:
            return {"answer": "LLM error: " + str(e),
                    "rounds": round_num + 1, "code_history": code_history}

        action = _parse_action(reply)
        if action is None:
            # LLM didn't produce JSON. Take whatever it said as the answer.
            return {"answer": reply.strip()[:2000],
                    "rounds": round_num + 1, "code_history": code_history}

        kind = action.get("action")
        if kind == "answer":
            answer_text = action.get("answer", "").strip()
            if prior is not None:
                prior.append({"role": "user", "content": question})
                prior.append({"role": "assistant", "content": answer_text})
            return {"answer": answer_text,
                    "rounds": round_num + 1, "code_history": code_history}

        if kind == "run_python":
            code = action.get("code", "")
            if not code:
                history.append({"role": "assistant", "content": reply})
                history.append({"role": "user",
                                "content": "Your JSON had action=run_python but no code. Try again."})
                continue
            outcome = _run(code, graph, messages, extras=extras)
            code_history.append({
                "round": round_num + 1, "code": code,
                "ok": outcome.get("ok"),
                "output": outcome.get("output", "")[:800],
                "error": outcome.get("error"),
            })
            if verbose:
                print("--- round", round_num + 1, "---")
                print(code)
                if outcome.get("output"):
                    print(">>> output:")
                    print(outcome["output"])
                if outcome.get("result"):
                    print(">>> result:")
                    print(outcome["result"])
                if outcome.get("error"):
                    print(">>> error:")
                    print(outcome["error"])
                print()

            history.append({"role": "assistant", "content": reply[:800]})
            result_text = json.dumps({
                "ok": outcome.get("ok"),
                "output": outcome.get("output", "")[:800],
                "result": (outcome.get("result") or "")[:800],
                "error": outcome.get("error"),
            }, default=str)
            history.append({"role": "user", "content": "Result:\n" + result_text})
            continue

        # Unknown action
        history.append({"role": "assistant", "content": reply})
        history.append({"role": "user",
                        "content": "Unknown action. Use run_python or answer."})

    # Out of rounds. Make one dedicated answer call with all the evidence.
    summary_lines = ["Original question: " + question, "", "Evidence gathered:"]
    for entry in code_history:
        summary_lines.append("")
        summary_lines.append("CODE (round " + str(entry["round"]) + "):")
        summary_lines.append(entry["code"][:500])
        if entry.get("output"):
            summary_lines.append("OUTPUT:")
            summary_lines.append(entry["output"][:1500])

    answer_prompt = "\n".join(summary_lines) + "\n\nBased on all of this, answer the original question in 2-5 sentences. Name specific people and quote specific evidence from the outputs. Do not write code."

    try:
        answer_text = call_llm(answer_prompt, temperature=0.0)
    except Exception as e:
        return {"answer": "LLM error on final answer: " + str(e),
                "rounds": max_rounds, "code_history": code_history}

    if prior is not None:
        prior.append({"role": "user", "content": question})
        prior.append({"role": "assistant", "content": answer_text.strip()})
    return {"answer": answer_text.strip(),
            "rounds": max_rounds, "code_history": code_history}


if __name__ == "__main__":
    from osi.loaders.whatsapp import _parse_export
    from osi.store import get_run, load_graph

    meta = get_run("wa-real")
    layer = (meta.get("config") or {}).get("layer", "whatsapp")
    g = load_graph("wa-real", layer)
    _, messages = _parse_export("/Users/2950639/Downloads/_chat.txt")

    print("graph:", g.number_of_nodes(), "nodes;", len(messages), "messages")
    print("type a question, or exit")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit", ""):
            break
        result = converse(q, g, messages, verbose=True)
        print()
        print("ANSWER:", result["answer"])
        print("(rounds:", result["rounds"], ")")
        print()
