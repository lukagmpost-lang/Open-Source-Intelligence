"""Plan tool calls for a question using the LLM."""
from __future__ import annotations

import json

from osi.answer import call_llm_with_tools
from osi.tools import TOOLS


_PLANNER_SYSTEM = (
    "You are a planning assistant. Read the user's question about a "
    "network graph and decide which tools to call. You may call one or "
    "more tools. Do not answer the question yourself — only plan the "
    "tool calls. Choose arguments carefully: read the question for "
    "words like 'least', 'most', 'weakest', 'strongest' and set "
    "direction accordingly. If the question asks about conflict, use "
    "most_tension. If the question asks for a ranking, use rank_nodes. "
    "If it asks about communities, use list_communities or community_audit."
)


def plan(question: str, history: list[dict] | None = None) -> dict:
    """Return a planning result: {tool_calls: [...], raw_response: str}."""
    messages = [{"role": "system", "content": _PLANNER_SYSTEM}]
    if history:
        for h in history[-4:]:
            messages.append(h)
    messages.append({"role": "user", "content": question})

    try:
        response = call_llm_with_tools(messages, TOOLS, temperature=0.0)
    except Exception as e:
        return {"tool_calls": [], "error": f"{type(e).__name__}: {e}"}

    return response
