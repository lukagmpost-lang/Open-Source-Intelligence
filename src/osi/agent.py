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
from dataclasses import dataclass, field

import networkx as nx

from osi.analysis import _modularity_of
from osi.answer import call_llm, verify_numbers
from osi.executors import (
    connectivity,
    explain_node,
    list_communities,
    network_health,
    rank_nodes,
    structural_criticality,
)
from osi.findings import generate_findings, pick_top_findings
from osi.hypotheses import find_unasked_observations, generate_hypotheses, infer_domain
from osi.result import ResultObject
from osi.store import get_run, load_communities, load_graph, load_metrics

MAX_STEPS = 10

WORRY_QUESTIONS = ("worry", "worried", "vulnerable", "fragile", "at risk", "should i worry")
HEALTH_QUESTIONS = ("healthy", "health", "shape", "structure", "how is")
IMPORTANT_QUESTIONS = ("important", "central", "key", "hubs", "who matters")
COMMUNITY_QUESTIONS = ("communities", "groups", "clusters")

_REQUIRED_BY_TYPE = (
    (WORRY_QUESTIONS, ("critical_nodes", "structural_criticality")),
    (HEALTH_QUESTIONS, ("network_health", "list_communities")),
    (IMPORTANT_QUESTIONS, ("rank_nodes", "explain_node")),
    (COMMUNITY_QUESTIONS, ("list_communities", "explain_node")),
)

_AUTO_PARAMS = {
    "critical_nodes": {"top_n": 10},
    "structural_criticality": {},
    "network_health": {},
    "list_communities": {"algorithm": "louvain"},
    "rank_nodes": {"metric": "pagerank", "top": 10},
}


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
}

_TOOL_DESCRIPTIONS = """\
- rank_nodes: metric is pagerank, degree, betweenness, or closeness. top is an integer.
- list_communities: algorithm is louvain or leiden.
- network_health: no parameters.
- structural_criticality: no parameters.
- critical_nodes: accounts that hold the network together. top_n is an integer.
- connectivity: source and target are account names.
- explain_node: node is an account name. For a community question, pass one account from the largest community.
"""

_AGENT_SYSTEM_TEMPLATE = """\
You are an agent analyzing a graph. You drive the analysis by 
requesting tool calls. You do not wait for tools to be run — 
YOU request them.

To request a tool, respond with EXACTLY this format:

TOOL: <tool_name>
PARAMS: <json_object>

For example:
TOOL: critical_nodes
PARAMS: {"top_n": 10}

Your response will be parsed. The tool will run automatically. 
You will see the result in the next message. Then you can request 
another tool or provide your final answer.

To give your final answer, respond with:

ANSWER: <your synthesis>

Do not ask the user to run tools. Do not wait for tools. Request 
them yourself with the TOOL/PARAMS format.

Available tools:
{tool_descriptions}

Precomputed findings about this graph:
{findings}

User question:
{question}

Rules:
- Call at least 2 tools before answering.
- For worry questions, call critical_nodes and structural_criticality.
- Every number in your answer must appear in a tool result.
- Keep the answer to 3-5 sentences.

EXAMPLE CONVERSATION:

User question: "what should I be worried about"

Your response:
TOOL: critical_nodes
PARAMS: {"top_n": 10}

[Tool runs. You see the result in the next message.]

Your response:
TOOL: structural_criticality
PARAMS: {}

[Tool runs. You see the result.]

Your response:
ANSWER: akdas has 759 connections, 22x the average. Removing 
it disconnects 342 other accounts. The network halves after 
removing 25% of the top accounts. The top 5 hubs sit in the 
same community and hold the network together. How sure: stable.

END EXAMPLE

Now respond to the actual question.
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
    if use_llm:
        answer = _react(question, domain, finding_texts, hypotheses, run_id, scratchpad, tools, results)
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


def _react(question, domain, finding_texts, hypotheses, run_id, scratchpad, tools, results) -> str:
    system = _system_prompt(question, finding_texts)
    for _step in range(MAX_STEPS):
        prompt = _prompt(question, domain, finding_texts, hypotheses, scratchpad)
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
            called = _called_tools(scratchpad)
            missing = [name for name in _required_tools(question) if name not in called]
            if not missing:
                return payload.strip()
            rows = []
            for name in missing:
                summary = _auto_run(run_id, name, results, tools, scratchpad)
                rows.append(f"{name} → {summary}")
            scratchpad.append(_auto_note(rows))
            try:
                follow = call_llm(_prompt(question, domain, finding_texts, hypotheses, scratchpad), system=system)
            except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
                break
            try:
                follow_kind, follow_text, _params = parse_agent_reply(follow)
            except ValueError:
                return follow.strip()
            if follow_kind == "answer":
                return follow_text.strip()
            return follow.strip()
        name = payload
        if name not in _TOOLS:
            scratchpad.append(f"Unknown tool {name}. Choose one from the tool list.")
            continue
        _record_tool(run_id, name, params, tools, results, scratchpad)
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


def _system_prompt(question: str, finding_texts: list[str]) -> str:
    findings = "\n".join(f"- {text}" for text in finding_texts) if finding_texts else "None yet."
    return (
        _AGENT_SYSTEM_TEMPLATE.replace("{tool_descriptions}", _TOOL_DESCRIPTIONS.strip())
        .replace("{findings}", findings)
        .replace("{question}", question)
    )


def _required_tools(question: str) -> list[str]:
    """Tools the question type must call before an answer is accepted."""
    text = question.casefold()
    for phrases, required in _REQUIRED_BY_TYPE:
        if any(phrase in text for phrase in phrases):
            return list(required)
    return []


def _called_tools(scratchpad: list[str]) -> set[str]:
    """Tool names that already have a result line in the scratchpad."""
    called = set()
    for line in scratchpad:
        if not line.startswith("TOOL RESULT "):
            continue
        called.add(line.split(":", 1)[0].removeprefix("TOOL RESULT ").strip())
    return called


def _auto_params(name: str, results: list[ResultObject]) -> dict:
    params = dict(_AUTO_PARAMS.get(name, {}))
    if name == "explain_node":
        params["node"] = _named_account(results)
    return params


def _named_account(results: list[ResultObject]) -> str:
    for result in results:
        for key in result.values:
            if key != "findings":
                return str(key)
    return ""


def _auto_run(run_id: str, name: str, results, tools, scratchpad) -> str:
    summary = _record_tool(run_id, name, _auto_params(name, results), tools, results, scratchpad)
    return summary


def _auto_note(rows: list[str]) -> str:
    lines = [
        "The following required tools were run automatically because you did not call them:",
        *rows,
        "",
        "Now write your final answer using all the results in the scratchpad.",
    ]
    return "\n".join(lines)


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


def _prompt(question, domain, finding_texts, hypotheses, scratchpad) -> str:
    lines = [
        f"Domain: {domain}",
        f"Question: {question}",
        "Findings already measured:",
    ]
    if finding_texts:
        lines.extend(f"- {text}" for text in finding_texts)
    else:
        lines.append("- none yet")
    lines.append("Hypotheses worth testing:")
    if hypotheses:
        lines.extend(f"- {item['hypothesis']}" for item in hypotheses)
    else:
        lines.append("- none yet")
    lines.append("Scratchpad:")
    if scratchpad:
        lines.extend(scratchpad)
    else:
        lines.append("(empty)")
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
    lines: list[str] = []
    for result in results:
        for item in result.values.get("findings") or []:
            sentence = str(item)
            if sentence not in lines:
                lines.append(sentence)
            if len(lines) == 3:
                break
        if len(lines) == 3:
            break
    if lines:
        return " ".join(lines)
    if finding_texts:
        return " ".join(finding_texts[:3])
    return "The measurements do not single out one account or one group."


def _compose(answer, observations, results, finding_texts) -> tuple[str, dict]:
    values = _base_values({}, results, [])
    if not verify_numbers(answer, values):
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
