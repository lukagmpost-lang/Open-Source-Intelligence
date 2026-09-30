import re

from osi.agent import _TOOLS, parse_agent_reply, run_agent
from osi.answer import verify_numbers
from osi.ask import ask, main
from osi.result import ResultObject


def test_parse_agent_reply_reads_a_tool_call_and_an_answer():
    kind, name, params = parse_agent_reply('TOOL: rank_nodes\nPARAMS: {"metric": "degree", "top": 5}')
    assert kind == "tool"
    assert name == "rank_nodes"
    assert params == {"metric": "degree", "top": 5}
    kind, prose, _params = parse_agent_reply("ANSWER: The hubs sit in different groups.")
    assert kind == "answer"
    assert prose == "The hubs sit in different groups."


def test_system_prompt_names_the_question_and_the_tool_minimum(monkeypatch):
    seen: dict[str, str] = {}

    def too_soon(prompt, **kwargs):
        seen["system"] = kwargs.get("system") or ""
        return "ANSWER: too soon"

    monkeypatch.setattr("osi.agent.call_llm", too_soon)
    run_agent("simple-v1", "what should I be worried about", use_llm=True)
    system = seen["system"]
    assert "what should I be worried about" in system
    assert "YOU request them" in system
    assert "Do not ask the user to run tools" in system
    assert "Call at least 2 tools before answering." in system
    assert "EXAMPLE CONVERSATION" in system
    assert "TOOL: critical_nodes" in system
    assert "TOOL: structural_criticality" in system
    assert "Now respond to the actual question." in system


def _decimal(text: str) -> str:
    match = re.search(r"\d+\.\d+", text)
    assert match, text
    return match.group(0)


def _tool_section(prompt: str, name: str) -> str:
    marker = f"TOOL RESULT {name}:"
    start = prompt.index(marker)
    return prompt[start : start + 1500]


def test_only_critical_nodes_auto_runs_structural_criticality(monkeypatch):
    prompts: list[str] = []

    def only_critical_nodes(prompt, **kwargs):
        prompts.append(prompt)
        if "run automatically" in prompt:
            bridge = _decimal(_tool_section(prompt, "critical_nodes"))
            fragile = _decimal(_tool_section(prompt, "structural_criticality"))
            return f"ANSWER: The bridge score is {bridge} and the remaining share is {fragile}."
        if "TOOL RESULT critical_nodes" not in prompt:
            return 'TOOL: critical_nodes\nPARAMS: {"top_n": 10}'
        return "ANSWER: One account sits on too many paths."

    monkeypatch.setattr("osi.agent.call_llm", only_critical_nodes)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    follow_up = next(prompt for prompt in prompts if "run automatically" in prompt)
    assert "structural_criticality →" in follow_up
    assert "TOOL RESULT structural_criticality:" in follow_up
    assert "Now write your final answer using all the results in the scratchpad." in follow_up
    assert all("Your answer was rejected" not in prompt for prompt in prompts)
    assert result.tools == ["critical_nodes", "structural_criticality"]
    assert "The bridge score is" in result.answer
    assert "the remaining share is" in result.answer
    assert verify_numbers(result.answer, result.values)


def test_an_early_answer_auto_runs_every_required_tool(monkeypatch):
    prompts: list[str] = []

    def answer_immediately(prompt, **kwargs):
        prompts.append(prompt)
        if "run automatically" not in prompt:
            return "ANSWER: Nothing stands out yet."
        bridge = _decimal(_tool_section(prompt, "critical_nodes"))
        fragile = _decimal(_tool_section(prompt, "structural_criticality"))
        return f"ANSWER: The bridge score is {bridge} and the remaining share is {fragile}."

    monkeypatch.setattr("osi.agent.call_llm", answer_immediately)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    assert result.tools == ["critical_nodes", "structural_criticality"]
    assert "critical_nodes →" in prompts[1]
    assert "structural_criticality →" in prompts[1]
    assert all("Your answer was rejected" not in prompt for prompt in prompts)
    assert verify_numbers(result.answer, result.values)


def test_the_loop_stops_after_ten_tool_calls(monkeypatch):
    def always_tool(prompt, **kwargs):
        return 'TOOL: rank_nodes\nPARAMS: {"metric": "degree", "top": 3}'

    monkeypatch.setattr("osi.agent.call_llm", always_tool)
    result = run_agent("simple-v1", "describe the links", use_llm=True)
    assert len(result.tools) == 10
    assert result.tools == ["rank_nodes"] * 10
    assert "While looking at this, I also noticed:" in result.answer


def test_nothing_is_auto_run_when_the_required_tools_were_called(monkeypatch):
    prompts: list[str] = []

    def both_tools(prompt, **kwargs):
        prompts.append(prompt)
        if "TOOL RESULT structural_criticality" in prompt:
            return "ANSWER: The hubs hold this network together."
        if "TOOL RESULT critical_nodes" in prompt:
            return "TOOL: structural_criticality\nPARAMS: {}"
        return 'TOOL: critical_nodes\nPARAMS: {"top_n": 10}'

    monkeypatch.setattr("osi.agent.call_llm", both_tools)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    assert result.tools == ["critical_nodes", "structural_criticality"]
    assert all("run automatically" not in prompt for prompt in prompts)
    assert all("Your answer was rejected" not in prompt for prompt in prompts)
    assert "The hubs hold this network together." in result.answer


def test_worry_question_on_reddit_2008_uses_three_tools(monkeypatch):
    def _stub(run, **kwargs):
        return ResultObject(
            intent="structural_criticality",
            params={"run": run},
            values={"largest": 0.42, "findings": ["A stub measurement."]},
            method="exact",
            sample_size=None,
            trust="stable",
            caveats=[],
            runtime_ms=0,
        )

    monkeypatch.setitem(_TOOLS, "structural_criticality", _stub)
    monkeypatch.setitem(_TOOLS, "critical_nodes", _stub)
    state = {"n": 0}

    def scripted(prompt, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return 'TOOL: rank_nodes\nPARAMS: {"metric": "pagerank", "top": 5}'
        if state["n"] == 2:
            return 'TOOL: list_communities\nPARAMS: {"algorithm": "louvain"}'
        if state["n"] == 3:
            return 'TOOL: explain_node\nPARAMS: {"node": "akdas"}'
        match = re.search(r'"largest_size":\s*(\d+)', prompt)
        assert match, prompt
        return f"ANSWER: The largest group contains {match.group(1)} accounts."

    monkeypatch.setattr("osi.agent.call_llm", scripted)
    result = run_agent("r2008-v2", "what should I be worried about", use_llm=True)
    assert result.tools[:3] == ["rank_nodes", "list_communities", "explain_node"]
    assert len(result.tools) >= 3
    assert verify_numbers(result.answer, result.values)
    section = result.answer.split("While looking at this, I also noticed:", 1)[1]
    bullets = [line for line in section.splitlines() if line.strip().startswith("- ")]
    assert len(bullets) >= 2


def test_ask_agent_flag_returns_the_agent_answer(monkeypatch):
    monkeypatch.setattr(
        "osi.ask.run_agent",
        lambda run_id, question, use_llm=True: type(
            "Outcome",
            (),
            {"answer": "Checked three measurements.\n\nWhile looking at this, I also noticed:\n- one\n- two", "tools": ["a", "b", "c"]},
        )(),
    )
    text = ask("simple-v1", "what should I be worried about", use_agent=True, use_cache=False)
    assert "While looking at this, I also noticed:" in text
    assert "- one" in text


def test_context_flag_prints_the_brief_first(capsys):
    main(["--run", "simple-v1", "--context", "--no-llm", "top 3 by pagerank"])
    printed = capsys.readouterr().out
    assert printed.startswith("Domain:")
    assert "Shape:" in printed
    assert "You could ask:" in printed
    assert printed.index("Domain:") < printed.index("The most central accounts are ")
