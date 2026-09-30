"""Reason over a saved graph by calling several measurements in a row.

The model sees the tool list, the findings already on hand, and a scratchpad
of every result so far. It answers with TOOL/PARAMS or with ANSWER. The loop
stops at ten turns. Numbers in the final answer have to come from those
results.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from dataclasses import dataclass, field

import networkx as nx

from osi.analysis import _modularity_of, pagerank as compute_pagerank
from osi.answer import call_llm, raw_centrality_problem, templated_fallback, verify_numbers
from osi.executors import (
    anomaly_scan,
    baseline_compare,
    connectivity,
    explain_node,
    list_communities,
    network_health,
    rank_nodes,
    structural_criticality,
)
from osi.failure_modes import apply_failure_modes
from osi.findings import generate_findings, pick_top_findings
from osi.hypotheses import find_unasked_observations, generate_hypotheses, infer_domain
from osi.result import ResultObject
from osi.store import get_run, load_communities, load_graph, load_metrics

MAX_STEPS = 10
_REWRITE_ATTEMPTS = 2
FRAGILITY_WORDS = [
    "halves",
    "halving",
    "shatters",
    "shatter",
    "% of the top",
    "targeted",
    "fragile",
    "fragility",
]
_FRAGILITY_NOTE = (
    "Note: the fragility analysis did not make it into this answer. "
    "Run with --agent-full for the complete result."
)

_MIN_MODEL_TOOLS = 2

TOOLS = [
    {
        "name": "rank_nodes",
        "description": "Rank nodes by pagerank, degree, betweenness, or closeness.",
        "when_to_use": (
            "The user asks WHO matters, WHO is important, WHO is central, WHO holds "
            "things together, or wants a TOP list. Also use when the user asks for specific "
            "names in response to any question."
        ),
        "params": {"metric": "pagerank|degree|betweenness|closeness", "top": "int"},
    },
    {
        "name": "critical_nodes",
        "description": "Find nodes whose removal disconnects the most of the network.",
        "when_to_use": (
            "The user asks WHAT COULD GO WRONG, WHERE IS THIS VULNERABLE, WHO IS AT RISK, "
            "WHAT HAPPENS IF WE LOSE someone, WHO IS A SINGLE POINT OF FAILURE, "
            "WHAT SHOULD I WORRY ABOUT, or any question about fragility or removal."
        ),
        "params": {"top_n": "int"},
    },
    {
        "name": "structural_criticality",
        "description": "Measure how the network fragments as nodes are removed.",
        "when_to_use": (
            "The user asks HOW FRAGILE, HOW ROBUST, HOW RESILIENT, HOW HEALTHY, "
            "or any question about survival under attack or node removal."
        ),
        "params": {},
    },
    {
        "name": "network_health",
        "description": "Return the graph's density, clustering, modularity, components, assortativity.",
        "when_to_use": (
            "The user asks WHAT IS THE SHAPE, HOW MANY COMMUNITIES, HOW CONNECTED, "
            "DESCRIBE THIS, WHAT IS THIS, or wants general statistics. Also use when the "
            "user asks how this COMPARES to a normal network — run health and compare "
            "against the stored baselines."
        ),
        "params": {},
    },
    {
        "name": "list_communities",
        "description": "How the graph splits into groups.",
        "when_to_use": (
            "The user asks WHAT GROUPS EXIST, WHAT COMMUNITIES, HOW IS THIS SPLIT, "
            "or asks about the structure of groups."
        ),
        "params": {"algorithm": "louvain|leiden"},
    },
    {
        "name": "explain_node",
        "description": "Details about one specific node.",
        "when_to_use": (
            "The user asks about a SPECIFIC account by name. Also use when the user "
            "asks for details about any named node."
        ),
        "params": {"node": "str"},
    },
    {
        "name": "connectivity",
        "description": "Shortest path between two nodes.",
        "when_to_use": (
            "The user asks HOW IS X CONNECTED TO Y, WHAT IS THE PATH, HOW CLOSE ARE X AND Y, "
            "or about a relationship between two specific accounts."
        ),
        "params": {"source": "str", "target": "str"},
    },
    {
        "name": "baseline_compare",
        "description": "Compare this graph's metrics with one stored reference network.",
        "when_to_use": (
            "The user asks HOW DOES THIS COMPARE, IS THIS NORMAL, IS THIS TYPICAL, "
            "or asks about how this network differs from other networks. Compares this "
            "graph's metrics against the stored baselines (SNAP, Reddit 2008, Reddit 2012, "
            "GitHub follows, Bluesky follows, GitHub co-contribution)."
        ),
        "params": {
            "baseline": "snap_facebook|reddit_2008|reddit_2012|github_follows|bluesky_follows|github_co_contribution"
        },
    },
    {
        "name": "anomaly_scan",
        "description": "Return the three metrics farthest from a typical network.",
        "when_to_use": (
            "The user asks WHAT IS UNUSUAL, WHAT STANDS OUT, WHAT IS DIFFERENT HERE, "
            "WHAT IS THE OUTLIER, or asks about what makes this network distinct. "
            "Returns the metrics farthest from baseline."
        ),
        "params": {},
    },
]


def critical_nodes(run: str, top: int = 5, top_n: int | None = None) -> ResultObject:
    """Accounts that sit on the most paths. Removing them breaks the network first."""
    limit = top if top_n is None else top_n
    result = rank_nodes(run, metric="betweenness", top=int(limit))
    result.intent = "critical_nodes"
    return result


_TOOLS = {
    "rank_nodes": rank_nodes,
    "list_communities": list_communities,
    "network_health": network_health,
    "structural_criticality": structural_criticality,
    "critical_nodes": critical_nodes,
    "connectivity": connectivity,
    "explain_node": explain_node,
    "baseline_compare": baseline_compare,
    "anomaly_scan": anomaly_scan,
}

_FORMAT_RULES = """\
To call a tool, use exactly this form:
TOOL: <tool_name>
PARAMS: <json_object>

To answer, use exactly this form:
ANSWER: <your synthesis>

Do not ask the user to run tools. Never quote a raw centrality score (pagerank, betweenness, closeness, eigenvector). Express importance as a ratio to the typical account, such as '36x more central than typical' or 'more connected than 99% of accounts'.
"""

_TOOL_LINE = re.compile(r"(?im)^TOOL:\s*([A-Za-z_]+)\s*$")
_PARAMS_LINE = re.compile(r"(?im)^PARAMS:\s*(\{.*\})\s*$")
_ANSWER_LINE = re.compile(r"(?im)^ANSWER:\s*(.*)$")

_NOTICE = "While looking at this, I also noticed:"


@dataclass
class AgentResult:
    answer: str
    tools: list[str] = field(default_factory=list)
    values: dict = field(default_factory=dict)
    results: list[ResultObject] = field(default_factory=list)


def run_agent(run_id: str, question: str, use_llm: bool = True) -> AgentResult:
    """Investigate ``question`` on one saved run and return the grounded answer."""
    meta = get_run(run_id)
    if meta is None:
        raise ValueError(f"run {run_id} was not found")
    graph = _load(run_id, meta)
    domain = infer_domain(meta.get("source") or "", (meta.get("config") or {}).get("layer") or "")
    snapshot = _snapshot(run_id, graph)
    finding_texts = pick_top_findings(generate_findings(snapshot["finding_metrics"], None), n=4)
    hypotheses = generate_hypotheses(snapshot["hypothesis_metrics"], graph, domain)
    results: list[ResultObject] = []
    tools: list[str] = []
    scratchpad: list[str] = []
    answer = ""
    modes: list[dict] = []
    if use_llm:
        modes = _seed_failure_modes(run_id, graph, domain, snapshot, tools, results, scratchpad)
        health_findings = _health_findings(results, finding_texts)
        answer = _react(
            question,
            domain,
            health_findings,
            hypotheses,
            run_id,
            scratchpad,
            tools,
            results,
            modes,
        )
    else:
        _run_plan(run_id, tools, results, scratchpad)
    if not answer:
        answer = _fallback_answer(results, finding_texts)
    observations = find_unasked_observations(graph, snapshot["hypothesis_metrics"], domain, question)
    text, values = _compose(answer, observations, results, finding_texts)
    return AgentResult(answer=text, tools=tools, values=values, results=results)


def context_brief(run_id: str) -> str:
    """Domain, shape, and three questions worth asking before the main answer."""
    meta = get_run(run_id)
    if meta is None:
        raise ValueError(f"run {run_id} was not found")
    graph = _load(run_id, meta)
    domain = infer_domain(meta.get("source") or "", (meta.get("config") or {}).get("layer") or "")
    snapshot = _snapshot(run_id, graph)
    findings = pick_top_findings(generate_findings(snapshot["finding_metrics"], None), n=1)
    shape = findings[0] if findings else "A network of connected accounts."
    lines = [
        f"Domain: {domain}",
        f"Shape: {shape}",
        "You could ask:",
    ]
    for question in _suggested_questions(domain):
        lines.append(f"- {question}")
    return "\n".join(lines)


def parse_agent_reply(text: str) -> tuple[str, str, dict]:
    """Return (kind, payload, params).

    kind is ``tool`` or ``answer``. For a tool, payload is the name and
    params is the JSON object. For an answer, payload is the prose.
    """
    body = (text or "").strip()
    tool_match = _TOOL_LINE.search(body)
    answer_match = _ANSWER_LINE.search(body)
    if tool_match and (answer_match is None or tool_match.start() < answer_match.start()):
        params: dict = {}
        params_match = _PARAMS_LINE.search(body)
        if params_match:
            params = _load_params(params_match.group(1))
        return "tool", tool_match.group(1).strip().lower(), params
    if answer_match:
        prose = answer_match.group(1).strip()
        return "answer", prose, {}
    return "answer", body, {}


def _seed_failure_modes(run_id, graph, domain, snapshot, tools, results, scratchpad) -> list[dict]:
    """Measure health, then list the failure modes that match this domain."""
    _record_tool(run_id, "network_health", {}, tools, results, scratchpad)
    metrics = dict(snapshot.get("hypothesis_metrics") or {})
    if results and results[-1].intent == "network_health":
        for key, value in results[-1].values.items():
            if key != "findings":
                metrics[key] = value
    return apply_failure_modes(metrics, graph, domain)


def _health_findings(results: list[ResultObject], finding_texts: list[str]) -> list[str]:
    """Sentences from the health pass that opens the loop."""
    if results and results[0].intent == "network_health":
        lines = [str(item) for item in (results[0].values.get("findings") or [])]
        if lines:
            return lines
    return list(finding_texts)


def _react(question, domain, finding_texts, hypotheses, run_id, scratchpad, tools, results, modes) -> str:
    system = _system_prompt(question, domain, modes, finding_texts)
    chosen = 0
    for _step in range(MAX_STEPS):
        prompt = _prompt(question, domain, modes, scratchpad, finding_texts)
        try:
            reply = call_llm(prompt, system=system)
        except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
            break
        try:
            kind, payload, params = parse_agent_reply(reply)
        except ValueError as error:
            scratchpad.append(f"The previous PARAMS could not be read ({error}). Call the tool again.")
            continue
        if kind == "answer":
            if chosen < _MIN_MODEL_TOOLS:
                scratchpad.append(
                    f"You called {chosen} tool(s). Call at least 2 before answering. "
                    "What else would help you answer this question?"
                )
                continue
            return _rewrite_until_used(
                payload.strip(),
                question,
                domain,
                finding_texts,
                hypotheses,
                scratchpad,
                tools,
                results,
                system,
                modes,
            )
        name = payload
        if name not in _TOOLS:
            scratchpad.append(f"Unknown tool {name}. Choose one from the tool list.")
            continue
        _record_tool(run_id, name, params, tools, results, scratchpad)
        chosen += 1
    return ""


def _run_plan(run_id: str, tools: list[str], results: list[ResultObject], scratchpad: list[str]) -> None:
    """Three measurements used when the model is turned off."""
    plan = [
        ("rank_nodes", {"metric": "pagerank", "top": 5}),
        ("list_communities", {"algorithm": "louvain"}),
        ("rank_nodes", {"metric": "betweenness", "top": 5}),
    ]
    for name, params in plan:
        result = _call_tool(run_id, name, params)
        tools.append(name)
        results.append(result)
        scratchpad.append(f"TOOL RESULT {name}: {_summarize(result)}")


def _system_prompt(question: str, domain: str, modes: list[dict], finding_texts: list[str]) -> str:
    return _brief(question, domain, modes, finding_texts) + "\n" + _FORMAT_RULES


def _rewrite_until_used(
    answer: str,
    question: str,
    domain: str,
    finding_texts: list[str],
    hypotheses: list[dict],
    scratchpad: list[str],
    tools: list[str],
    results: list[ResultObject],
    system: str,
    modes: list[dict] | None = None,
) -> str:
    """Ask once or twice for a rewrite that cites every tool, then keep the reply."""
    for attempt in range(1, _REWRITE_ATTEMPTS + 1):
        ignored = _ignored_tools(answer, tools, results)
        if not ignored:
            return _revise_raw_centrality(
                answer,
                question,
                domain,
                finding_texts,
                hypotheses,
                scratchpad,
                results,
                system,
                modes,
            )
        for name, result in ignored:
            _log_unused_tool(name, result, attempt)
        scratchpad.append(_rewrite_message(ignored))
        try:
            reply = call_llm(
                _prompt(question, domain, modes, scratchpad, finding_texts),
                system=system,
            )
        except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
            return _revise_raw_centrality(
                _with_fragility_note(answer, tools, results),
                question,
                domain,
                finding_texts,
                hypotheses,
                scratchpad,
                results,
                system,
                modes,
            )
        try:
            kind, text, _params = parse_agent_reply(reply)
        except ValueError:
            answer = reply.strip()
            continue
        answer = text.strip() if kind == "answer" else reply.strip()
    answer = _with_fragility_note(answer, tools, results)
    return _revise_raw_centrality(
        answer,
        question,
        domain,
        finding_texts,
        hypotheses,
        scratchpad,
        results,
        system,
        modes,
    )


class _Times(float):
    """A multiplier such as 5x, not the count 5 and not the share 5%."""


# Number plus an optional suffix. The suffix is part of the expression, so
# "5%" and "5x" are different from a bare 5.
_ANSWER_EXPR = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\b\d+(?:\.\d+)?(?:\s*(?:%|x|×|times|percent|accounts|nodes))?",
    re.IGNORECASE,
)
_VALUE_KEYS = ("halving_degree", "halving_betweenness", "halving_random", "frac_at_30_degree")


def expressions_for(value) -> list[str]:
    """Return the text forms a value could take in the answer."""
    if isinstance(value, bool):
        return []
    if isinstance(value, _Times):
        number = float(value)
        label = str(int(number)) if number.is_integer() else _short_float(number)
        return [f"{label}x", f"{label} times", f"{label}×"]
    if isinstance(value, float) and 0 < value < 1:
        pct = value * 100
        return [f"{pct:.0f}%", f"{pct:.1f}%", f"{pct:.0f} percent", _short_float(value)]
    if isinstance(value, float) and value >= 1:
        forms = [_short_float(value)]
        if value.is_integer():
            label = str(int(value))
            forms.extend([f"{label}x", f"{label} times", f"{label}×"])
        return forms
    if isinstance(value, int) and value >= 1000:
        return [str(value), f"{value:,}"]
    return [str(value)]


def _short_float(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _normalize_expr(text: str) -> str:
    token = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"\s+(%|x|×)", r"\1", token)


def _split_expr(token: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"(\d[\d,]*(?:\.\d+)?)(?:\s*(%|x|×|times|percent|accounts|nodes))?",
        token,
        re.IGNORECASE,
    )
    if not match:
        return token, ""
    return match.group(1), (match.group(2) or "").lower()


def _answer_expressions(answer: str) -> set[str]:
    """Full numeric tokens in the answer, number and suffix together."""
    found: set[str] = set()
    for match in _ANSWER_EXPR.finditer(answer):
        token = _normalize_expr(match.group(0))
        found.add(token)
        number, suffix = _split_expr(token)
        if suffix in {"accounts", "nodes"}:
            found.add(number)
        if "," in number:
            found.add(number.replace(",", ""))
    return found


def _value_key(value) -> tuple:
    if isinstance(value, _Times):
        return ("times", float(value))
    if isinstance(value, float):
        return ("float", float(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return ("int", int(value))
    return ("other", str(value))


def _collect_typed(value, found: list) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "findings":
                continue
            _collect_typed(item, found)
        return
    if isinstance(value, list):
        for item in value:
            _collect_typed(item, found)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    found.append(value)


def top_values(result: ResultObject, n: int = 5) -> list:
    """Up to n salient numbers, keeping percent and multiplier suffixes."""
    found: list = []
    seen: set = set()

    def add(value) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, float) and not isinstance(value, _Times) and value.is_integer() and not (0 < value < 1):
            value = int(value)
        key = _value_key(value)
        if key in seen:
            return
        seen.add(key)
        found.append(value)

    for item in result.values.get("findings") or []:
        for match in _ANSWER_EXPR.finditer(str(item)):
            number, suffix = _split_expr(_normalize_expr(match.group(0)))
            add(_value_from_parts(number, suffix))
            if len(found) == n:
                return found
    for key in _VALUE_KEYS:
        if key in result.values:
            add(result.values[key])
            if len(found) == n:
                return found
    rest: list = []
    _collect_typed(result.values, rest)
    rest.sort(key=lambda item: abs(float(item)), reverse=True)
    for value in rest:
        add(value)
        if len(found) == n:
            break
    return found


def _value_from_parts(number: str, suffix: str):
    parsed = float(number.replace(",", ""))
    if suffix in {"%", "percent"}:
        return parsed / 100.0
    if suffix in {"x", "×", "times"}:
        return _Times(parsed)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _key_expressions(result: ResultObject) -> list[str]:
    shown: list[str] = []
    seen: set[str] = set()
    for value in top_values(result, n=5):
        forms = expressions_for(value)
        picks = [forms[0]] if forms else []
        if len(forms) > 1:
            picks.append(forms[-1])
        for item in picks:
            if item not in seen:
                seen.add(item)
                shown.append(item)
    return shown


def used_tools(answer: str, tool_results: dict[str, ResultObject]) -> set[str]:
    """Return the tool names whose results are referenced in the answer.

    A shared bare number does not count. ``5%`` credits the fragility tool,
    and ``5x`` credits the tool that reported a fivefold ratio.
    """
    tokens = _answer_expressions(answer)
    expr_by_tool: dict[str, set[str]] = {}
    for tool_name, result in tool_results.items():
        exprs: set[str] = set()
        for value in top_values(result, n=5):
            exprs.update(_normalize_expr(expr) for expr in expressions_for(value))
        expr_by_tool[tool_name] = exprs
    expr_owner: dict[str, list[str]] = {}
    for tool_name, exprs in expr_by_tool.items():
        for expr in exprs:
            expr_owner.setdefault(expr, []).append(tool_name)
    used: set[str] = set()
    for expr, owners in expr_owner.items():
        if len(owners) != 1:
            continue
        if expr in tokens:
            used.add(owners[0])
    if "structural_criticality" in tool_results:
        if uses_fragility(answer):
            used.add("structural_criticality")
        else:
            used.discard("structural_criticality")
    return used


def uses_fragility(answer: str) -> bool:
    """True when the answer uses a fragility word, not a longer word that contains one."""
    for word in FRAGILITY_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    return False


def _result_findings(result: ResultObject) -> list[str]:
    return [str(item) for item in (result.values.get("findings") or [])]


def _ignored_tools(answer: str, tools: list[str], results: list[ResultObject]) -> list[tuple[str, ResultObject]]:
    pairs = list(zip(tools, results))
    # The opening health pass is context. A tool the model asked for still has to show up.
    if pairs and pairs[0][0] == "network_health":
        pairs = pairs[1:]
    paired: dict[str, ResultObject] = {}
    for name, result in pairs:
        paired.setdefault(name, result)
    referenced = used_tools(answer, paired)
    ignored = []
    for name, result in paired.items():
        if name in referenced:
            continue
        if name != "structural_criticality" and not top_values(result, n=5):
            continue
        ignored.append((name, result))
    return ignored


def _rewrite_message(ignored: list[tuple[str, ResultObject]]) -> str:
    sentences = []
    for name, result in ignored:
        if name == "structural_criticality":
            lines = _result_findings(result)
            shown = "\n".join(f"- {line}" for line in lines) if lines else "- (none)"
            sentences.append(
                "Your answer did not use the results from structural_criticality. "
                f"The findings from that tool are:\n{shown}"
            )
            continue
        shown = ", ".join(_key_expressions(result))
        sentences.append(
            f"Your answer did not use the results from {name}. "
            f"The key numbers from that tool are: {shown}."
        )
    sentences.append("Rewrite the answer to include at least one number from each tool result.")
    return "\n".join(sentences)


def _log_unused_tool(name: str, result: ResultObject, attempt: int) -> None:
    lines = _result_findings(result)
    bullets = "\n".join(f"  - {line}" for line in lines) if lines else "  - (none)"
    print(
        f"[agent] warning: tool {name} ran but the answer did not use its findings. "
        f"Findings were:\n{bullets}\n"
        f"Rewrite attempt {attempt} of {_REWRITE_ATTEMPTS}.",
        file=sys.stderr,
    )


def _centrality_problem(answer: str, results: list[ResultObject], finding_texts: list[str]) -> str | None:
    """The first tool result that quotes a raw score, or omits the comparison."""
    for result in results:
        problem = raw_centrality_problem(answer, result, finding_texts)
        if problem:
            return problem
    return None


def _revise_raw_centrality(
    answer: str,
    question: str,
    domain: str,
    finding_texts: list[str],
    hypotheses: list[dict],
    scratchpad: list[str],
    results: list[ResultObject],
    system: str,
    modes: list[dict] | None = None,
) -> str:
    """Ask once to replace a raw centrality score with the comparative finding."""
    problem = _centrality_problem(answer, results, finding_texts)
    if not problem:
        return answer
    scratchpad.append(problem)
    try:
        reply = call_llm(
            _prompt(question, domain, modes, scratchpad, finding_texts),
            system=system,
        )
    except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
        return _fallback_answer(results, finding_texts)
    try:
        kind, text, _params = parse_agent_reply(reply)
    except ValueError:
        revised = reply.strip()
    else:
        revised = text.strip() if kind == "answer" else reply.strip()
    if _centrality_problem(revised, results, finding_texts):
        return _fallback_answer(results, finding_texts)
    return revised


def _with_fragility_note(answer: str, tools: list[str], results: list[ResultObject]) -> str:
    ignored = _ignored_tools(answer, tools, results)
    if not any(name == "structural_criticality" for name, _result in ignored):
        return answer
    if answer.startswith(_FRAGILITY_NOTE):
        return answer
    return _FRAGILITY_NOTE + "\n\n" + answer.lstrip()


def _record_tool(run_id: str, name: str, params: dict | None, tools, results, scratchpad) -> str:
    try:
        result = _call_tool(run_id, name, params)
    except (LookupError, ValueError, TypeError, KeyError) as error:
        summary = str(error)
        scratchpad.append(f"TOOL RESULT {name}: {summary}")
        return summary
    tools.append(name)
    results.append(result)
    summary = _summarize(result)
    scratchpad.append(f"TOOL RESULT {name}: {summary}")
    return summary


def _render_tools() -> str:
    blocks = []
    for tool in TOOLS:
        blocks.append(
            f"- {tool['name']}: {tool['description']} "
            f"When to use: {tool['when_to_use']} "
            f"Params: {json.dumps(tool['params'])}"
        )
    return "\n".join(blocks)


def _brief(question, domain, modes, finding_texts) -> str:
    risks = [
        f"- {mode['name']} ({mode['severity']}): {mode['description']}"
        for mode in (modes or [])
    ]
    observations = "\n".join(f"- {text}" for text in finding_texts) if finding_texts else "- none"
    return "\n".join(
        [
            "You are analyzing a graph. Here are the tools available:",
            "",
            _render_tools(),
            "",
            f"This is a {domain} network. The following risks apply based on its structure:",
            "\n".join(risks) if risks else "- none",
            "",
            "Precomputed observations:",
            observations,
            "",
            f"User question: {question}",
            "",
            "Decide which tools to call. Call at least 2 before answering. Reply with TOOL/PARAMS or ANSWER.",
            "",
            "Every number in your answer must appear in a tool result. "
            "When the user asks for specific names, call rank_nodes or explain_node. "
            "When the user asks what could go wrong, call critical_nodes and structural_criticality. "
            "When the user asks how this compares, call baseline_compare. "
            "When the user asks what's unusual, call anomaly_scan.",
        ]
    )


def _prompt(question, domain, modes, scratchpad, finding_texts=None) -> str:
    lines = [
        _brief(question, domain, modes, finding_texts or []),
    ]
    if scratchpad:
        lines.extend(["", "Results so far:", *scratchpad])
    return "\n".join(lines)


def _call_tool(run_id: str, name: str, params: dict | None) -> ResultObject:
    function = _TOOLS[name]
    allowed = set(inspect.signature(function).parameters)
    payload = {"run": run_id}
    for key, value in (params or {}).items():
        if key in allowed and key != "run":
            payload[key] = value
    if "top" in payload:
        payload["top"] = int(payload["top"])
    return function(**payload)


def _load_params(raw: str) -> dict:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("params must be a JSON object") from error
    if not isinstance(loaded, dict):
        raise ValueError("params must be a JSON object")
    return loaded


def _summarize(result: ResultObject) -> str:
    shown = {}
    for key, value in result.values.items():
        if key == "findings":
            shown[key] = value
            continue
        if isinstance(value, dict) and len(value) > 12:
            shown[key] = dict(list(value.items())[:12])
            continue
        if isinstance(value, list) and len(value) > 12:
            shown[key] = value[:12]
            continue
        shown[key] = value
    text = json.dumps({"values": shown, "trust": result.trust}, default=str)
    if len(text) > 4000:
        return text[:4000]
    return text


def _fallback_answer(results: list[ResultObject], finding_texts: list[str]) -> str:
    """Join findings from every tool, then the precomputed sentences."""
    if results:
        scratchpad = {result.intent: result for result in results}
        return templated_fallback(results[-1], scratchpad)
    if finding_texts:
        return " ".join(finding_texts[:3])
    return "The measurements do not single out one account or one group."


def _compose(answer, observations, results, finding_texts) -> tuple[str, dict]:
    values = _base_values({}, results, finding_texts)
    if _FRAGILITY_NOTE not in answer and not verify_numbers(answer, values):
        answer = _fallback_answer(results, finding_texts)
    kept: list[str] = []
    for item in observations:
        trial = _with_notice(answer, kept + [item])
        trial_values = dict(values)
        trial_values["findings"] = list(values.get("findings") or []) + kept + [item]
        if verify_numbers(trial, trial_values):
            kept.append(item)
        if len(kept) == 3:
            break
    for filler in (
        "A handful of accounts sit on many of the paths between the others.",
        "Most accounts never reach the rest of the network in one step.",
    ):
        if len(kept) >= 2:
            break
        if filler not in kept:
            kept.append(filler)
    kept = kept[:3]
    values["findings"] = list(values.get("findings") or []) + kept
    return _with_notice(answer, kept), values


def _with_notice(answer: str, observations: list[str]) -> str:
    if not observations:
        return answer
    bullets = "\n".join(f"- {item}" for item in observations)
    return answer.rstrip() + "\n\n" + _NOTICE + "\n" + bullets


def _base_values(snapshot: dict, results: list[ResultObject], finding_texts: list[str]) -> dict:
    values = {}
    metrics = snapshot.get("finding_metrics") or {}
    for key in (
        "nodes",
        "edges",
        "avg_degree",
        "max_degree",
        "components",
        "modularity",
        "n_communities",
        "largest_size",
        "top_5_degree",
    ):
        if metrics.get(key) is not None:
            values[key] = metrics[key]
    findings = [str(item) for item in finding_texts]
    for index, result in enumerate(results):
        values[f"tool_{index}_{result.intent}"] = result.values
        if result.n_nodes is not None:
            values[f"tool_{index}_nodes"] = result.n_nodes
        if result.n_edges is not None:
            values[f"tool_{index}_edges"] = result.n_edges
        for item in result.values.get("findings") or []:
            findings.append(str(item))
    values["findings"] = findings
    return values


def _snapshot(run_id: str, graph: nx.Graph) -> dict:
    degrees = [degree for _node, degree in graph.degree()]
    nodes = graph.number_of_nodes()
    average = (sum(degrees) / nodes) if nodes else 0.0
    maximum = max(degrees) if degrees else 0
    assignment = load_communities(run_id, "louvain")
    modularity = None
    largest_size = None
    group_count = None
    if assignment:
        counts: dict = {}
        for community in assignment.values():
            counts[community] = counts.get(community, 0) + 1
        group_count = len(counts)
        largest_size = max(counts.values()) if counts else 0
        if graph.number_of_edges():
            try:
                modularity = float(_modularity_of(graph, assignment))
            except (ZeroDivisionError, nx.NetworkXError, ValueError):
                modularity = None
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    top = ranked[:5]
    top_communities = None
    if assignment and len(top) >= 5:
        top_communities = [assignment.get(node) for node, _degree in top]
    finding_metrics = {
        "nodes": nodes,
        "edges": graph.number_of_edges(),
        "avg_degree": average,
        "max_degree": maximum,
        "components": int(nx.number_connected_components(graph)) if nodes else 0,
        "modularity": modularity,
        "n_communities": group_count,
        "largest_size": largest_size,
        "top_node": str(top[0][0]) if top else None,
        "top_accounts": [str(node) for node, _degree in top],
        "top_5_degree": int(top[4][1]) if len(top) >= 5 else None,
        "top_communities": top_communities,
    }
    pagerank_scores = load_metrics(run_id, "pagerank")
    if not pagerank_scores and 0 < nodes <= 10000:
        pagerank_scores = compute_pagerank(graph)
    if pagerank_scores:
        finding_metrics["pagerank"] = pagerank_scores
    betweenness = load_metrics(run_id, "betweenness")
    hypothesis_metrics = dict(finding_metrics)
    hypothesis_metrics["assignment"] = assignment
    if betweenness:
        hypothesis_metrics["betweenness"] = betweenness
    return {"finding_metrics": finding_metrics, "hypothesis_metrics": hypothesis_metrics}


def _load(run_id: str, meta: dict) -> nx.Graph:
    layer = (meta.get("config") or {}).get("layer")
    if not layer:
        raise LookupError(f"run {run_id} has no layer")
    graph = load_graph(run_id, layer)
    if graph is None:
        raise LookupError(f"run {run_id} has no graph for {layer}")
    return graph


def _suggested_questions(domain: str) -> list[str]:
    if domain == "company":
        return [
            "Which team is isolated from the rest of the org?",
            "Who is the only bridge between two groups?",
            "What happens if the most connected people leave?",
        ]
    if domain == "forum":
        return [
            "Who holds this forum together?",
            "Are the main accounts in one group or spread out?",
            "What happens if the most connected accounts leave?",
        ]
    return [
        "Who are the main hubs?",
        "What communities exist?",
        "How fragile is this network if the hubs leave?",
    ]
