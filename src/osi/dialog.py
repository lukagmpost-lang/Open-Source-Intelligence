"""Two-call dialog: plan retrievals, then answer."""
from __future__ import annotations

import json
import re

from osi.answer import call_llm
from osi.context import build_context
from osi.retrieve import execute_all, top_senders


_PLAN_SYSTEM = "You plan data retrieval. Reply with JSON only."

_PLAN_PROMPT = """The user asks about a chat network. Decide what to look at.

Network summary: {summary}

Top senders (name: count): {senders}

If the question is structural (centrality, communities, rankings), reply:
{{"definition": "n/a", "strategies": [], "reasoning": "structural"}}

If the question is about behavior, tone, topics, or content:
1. Write a ONE-SENTENCE definition of the concept. Then write TWO examples
   of things that LOOK like the concept but are NOT it. The counter-examples
   are what make the definition sharp.
   Example — for "gossip":
     definition: "talking about a person who is not present, to a third party,
                  usually with evaluative or negative content."
     NOT gossip 1: "@Name please do X" — direct address, not third-party talk.
     NOT gossip 2: "Спасибо @Name" — thanking someone directly, not gossip.
   Example — for "bridge":
     definition: "an account that connects two groups that otherwise don't talk."
     NOT bridge 1: "a hub many people talk to within one group."
     NOT bridge 2: "a person with a high message count."
2. Choose retrieval strategies that surface messages matching the definition
   but NOT the counter-examples.
Reply:
{{"definition": "...", "not_this": ["...", "..."], "strategies": [{{"kind": "sample_by_sender", "names": ["Alice"], "per_sender": 15}}], "reasoning": "..."}}

Available strategies:
- {{"kind": "sample_by_sender", "names": [...], "per_sender": N}}  — N messages from each
- {{"kind": "filter_by_sender", "name": "X", "limit": N}}  — all messages from one person
- {{"kind": "mentions_others", "limit": N}}  — messages where the sender names another participant (useful for gossip / social dynamics)
- {{"kind": "search", "text": "word", "limit": N}}  — substring match
- {{"kind": "sample_recent", "n": N}}  — last N messages

HINTS:
- "gossip", "talks about people", "who relays" → mentions_others
- "what does X talk about", "what is X's role" → filter_by_sender with limit 100-150
- "who talks the most" → sample_by_sender with per_sender 30-50
- If you retrieve fewer than 50 messages for a behavioral question, increase the limit.

Cap total retrieved at ~60.

Question: {question}

Reply with JSON only."""


_ANSWER_SYSTEM = (
    "You answer questions about a chat network. You may reason from the "
    "retrieved messages the way a person reading the chat would. "
    "Ground your read in specific messages. Never invent a numeric metric."
)

_ANSWER_PROMPT = """User asked: {question}

GRAPH METRICS: {summary}

NODES (top and bottom by pagerank):
{nodes}

COMMUNITIES:
{communities}

SCORED RELATIONSHIPS (EDGE entries, not messages. Do not quote
these as if a person said them.):
{edges}

WORKING DEFINITION (judge only against this):
{definition}

COUNTER-EXAMPLES (things that look similar but do NOT count):
{counter_examples}

RETRIEVED MESSAGES ({n} shown):
{messages}

INSTRUCTIONS:
- If the question is structural (centrality, communities), answer from
  the graph tables.
- If the question is behavioral (gossip, tone, personality, topics),
  read the retrieved messages and give your best judgement, the way a
  person reading the chat would.
- Say "based on the N messages I read" so the reader knows the sample.
- CITE EVIDENCE FROM THE RETRIEVED MESSAGES ONLY. Quote a specific
  message text with its sender. Do NOT cite the graph tables or edge
  labels as if they were messages. If you cannot find a message that
  shows the behavior, say the sample did not contain one.
- Do NOT refuse just because there is no numeric metric for the concept.
  Reading the chat is a valid way to answer.

Answer in 2-5 sentences."""


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _format_messages(msgs: list, cap: int = 200) -> str:
    lines = []
    for m in msgs[:cap]:
        sender = m.get("sender", "?")
        text = (m.get("text") or "").replace("\n", " ")[:180]
        lines.append(f"{sender}: {text}")
    return "\n".join(lines)


def _parse_plan(reply: str) -> dict:
    m = _JSON.search(reply or "")
    if not m:
        return {"strategies": [], "reasoning": "parse failed"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"strategies": [], "reasoning": "parse failed"}


def answer(question: str, graph, messages: list, history=None) -> dict:
    ctx = build_context(graph)
    senders = top_senders(messages, 20)
    senders_str = ", ".join(f"{n} ({c})" for n, c in senders)

    plan_prompt = _PLAN_PROMPT.format(summary=ctx["summary"], senders=senders_str, question=question)
    try:
        plan_reply = call_llm(plan_prompt, system=_PLAN_SYSTEM, temperature=0.0)
        plan = _parse_plan(plan_reply)
    except Exception as e:
        plan = {"strategies": [], "reasoning": "plan failed: " + str(e)}

    strategies = plan.get("strategies") or []
    retrieved = execute_all(strategies, messages, cap=60)

    definition = plan.get("definition", "n/a")
    answer_prompt = _ANSWER_PROMPT.format(
        definition=definition,
        counter_examples="\n".join(f"- {x}" for x in (plan.get("not_this") or ["(none given)"])),
        question=question,
        summary=ctx["summary"],
        nodes=ctx["node_table"],
        communities=ctx["comm_table"],
        edges=ctx["edge_table"],
        n=len(retrieved),
        messages=_format_messages(retrieved, cap=60),
    )
    try:
        reply = call_llm(answer_prompt, system=_ANSWER_SYSTEM, temperature=0.0)
    except Exception as e:
        return {"answer": "LLM error: " + str(e), "plan": plan, "retrieved": len(retrieved)}

    return {"answer": reply.strip(), "plan": plan, "definition": plan.get("definition", "n/a"), "retrieved": len(retrieved)}
