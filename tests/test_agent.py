import re

from osi.agent import _TOOLS, expressions_for, parse_agent_reply, run_agent, used_tools, uses_fragility
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
            return (
                f"ANSWER: The bridge score is {bridge} and the remaining share is {fragile}. "
                "akdas is more central than a typical account."
            )
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
            return "ANSWER: The hubs hold this network together. They are more central than typical."
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
    assert {"rank_nodes", "list_communities", "explain_node"} <= set(result.tools)
    assert len(result.tools) >= 3
    assert verify_numbers(result.answer, result.values)
    section = result.answer.split("While looking at this, I also noticed:", 1)[1]
    bullets = [line for line in section.splitlines() if line.strip().startswith("- ")]
    assert len(bullets) >= 2


def _stub_result(intent: str, values: dict) -> ResultObject:
    return ResultObject(
        intent=intent,
        params={},
        values=values,
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=[],
        runtime_ms=0,
    )


def _install_worry_stubs(monkeypatch):
    monkeypatch.setitem(
        _TOOLS,
        "critical_nodes",
        lambda run, **kwargs: _stub_result(
            "critical_nodes",
            {"hub": 0.91, "findings": ["The bridge score is 0.91."]},
        ),
    )
    monkeypatch.setitem(
        _TOOLS,
        "structural_criticality",
        lambda run, **kwargs: _stub_result(
            "structural_criticality",
            {
                "largest": 0.08,
                "findings": [
                    "The network is extremely fragile: removing just 5% of the top accounts by degree halves it."
                ],
            },
        ),
    )


def test_an_answer_that_uses_only_critical_nodes_asks_for_a_rewrite(monkeypatch):
    prompts: list[str] = []
    _install_worry_stubs(monkeypatch)

    def only_the_bridge(prompt, **kwargs):
        prompts.append(prompt)
        if "did not use the results" in prompt:
            return "ANSWER: The bridge score is 0.91 and removing 5% of the top accounts halves the network."
        if "TOOL RESULT structural_criticality" in prompt:
            return "ANSWER: The bridge score is 0.91."
        if "TOOL RESULT critical_nodes" in prompt:
            return "TOOL: structural_criticality\nPARAMS: {}"
        return 'TOOL: critical_nodes\nPARAMS: {"top_n": 5}'

    monkeypatch.setattr("osi.agent.call_llm", only_the_bridge)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    rewrite = next(prompt for prompt in prompts if "did not use the results" in prompt)
    assert "structural_criticality" in rewrite
    assert "The network is extremely fragile: removing just 5% of the top accounts by degree halves it." in rewrite
    assert "Rewrite the answer to include at least one number from each tool result." in rewrite
    assert "0.91" in result.answer
    assert "5%" in result.answer
    assert verify_numbers(result.answer, result.values)


def test_an_answer_that_uses_both_tools_is_accepted(monkeypatch):
    prompts: list[str] = []
    _install_worry_stubs(monkeypatch)

    def both(prompt, **kwargs):
        prompts.append(prompt)
        if "TOOL RESULT structural_criticality" in prompt:
            return "ANSWER: The bridge score is 0.91 and removing 5% of the top accounts halves the network."
        if "TOOL RESULT critical_nodes" in prompt:
            return "TOOL: structural_criticality\nPARAMS: {}"
        return 'TOOL: critical_nodes\nPARAMS: {"top_n": 5}'

    monkeypatch.setattr("osi.agent.call_llm", both)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    assert all("did not use the results" not in prompt for prompt in prompts)
    assert "0.91" in result.answer
    assert "5%" in result.answer
    assert verify_numbers(result.answer, result.values)


def test_after_two_rewrites_the_answer_is_accepted_anyway(monkeypatch, capsys):
    prompts: list[str] = []
    _install_worry_stubs(monkeypatch)

    def still_ignores_fragility(prompt, **kwargs):
        prompts.append(prompt)
        if "TOOL RESULT critical_nodes" not in prompt:
            return 'TOOL: critical_nodes\nPARAMS: {"top_n": 5}'
        if "TOOL RESULT structural_criticality" not in prompt:
            return "TOOL: structural_criticality\nPARAMS: {}"
        return "ANSWER: The bridge score is 0.91."

    monkeypatch.setattr("osi.agent.call_llm", still_ignores_fragility)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    rewrites = [prompt for prompt in prompts if "did not use the results from structural_criticality" in prompt]
    assert len(rewrites) == 2
    finding = "The network is extremely fragile: removing just 5% of the top accounts by degree halves it."
    assert finding in rewrites[0]
    assert result.answer.startswith(
        "Note: the fragility analysis did not make it into this answer. "
        "Run with --agent-full for the complete result."
    )
    assert "The bridge score is 0.91." in result.answer
    assert "5%" not in result.answer.split("While looking at this", 1)[0]
    logged = capsys.readouterr().err
    assert "[agent] warning: tool structural_criticality ran but the answer did not use its findings." in logged
    assert finding in logged
    assert "Rewrite attempt 1 of 2." in logged
    assert "Rewrite attempt 2 of 2." in logged
    assert verify_numbers(result.answer, result.values)


def _tool_results():
    return {
        "structural_criticality": _stub_result(
            "structural_criticality",
            {"halving_degree": 0.05, "findings": ["Removing 5% halves the network."]},
        ),
        "critical_nodes": _stub_result(
            "critical_nodes",
            {"removed": 342, "findings": ["342 accounts become unreachable."]},
        ),
        "rank_nodes": _stub_result(
            "rank_nodes",
            {"akdas": 759, "findings": ["akdas has 759 connections."]},
        ),
    }


def test_fragmented_does_not_count_as_fragile():
    assert uses_fragility("This network is far more fragmented than typical.") is False
    assert "structural_criticality" not in used_tools(
        "This network is far more fragmented than typical.",
        _tool_results(),
    )
    assert uses_fragility("The network is fragile.") is True


def test_connection_count_does_not_use_structural_criticality():
    results = _tool_results()
    assert "structural_criticality" not in used_tools("759 connections", results)


def test_halves_the_network_uses_structural_criticality():
    results = _tool_results()
    assert "structural_criticality" in used_tools("halves the network", results)


def test_percent_expression_uses_fragility_and_multiplier_does_not():
    results = _tool_results()
    assert used_tools("removing 5% halves the network", results) == {"structural_criticality"}
    assert "structural_criticality" not in used_tools("5x more connected", results)
    assert used_tools("342 accounts become unreachable", results) == {"critical_nodes"}
    assert used_tools("759 connections", results) == {"rank_nodes"}


def test_a_missing_unique_expression_is_not_marked_used():
    results = _tool_results()
    used = used_tools("759 connections", results)
    assert "structural_criticality" not in used
    assert "critical_nodes" not in used
    shared = {
        "structural_criticality": _stub_result("structural_criticality", {"halving_degree": 0.05, "also": 5}),
        "rank_nodes": _stub_result("rank_nodes", {"score": 5}),
    }
    assert used_tools("the score is 5", shared) == set()
    assert used_tools("removing 5% halves the network", shared) == {"structural_criticality"}


def test_expressions_keep_the_suffix_with_the_number():
    assert "5%" in expressions_for(0.05)
    assert "0.05" in expressions_for(0.05)
    assert "5 percent" in expressions_for(0.05)
    assert "5x" not in expressions_for(0.05)
    assert "5x" in expressions_for(5.0)
    assert "5 times" in expressions_for(5.0)
    assert "5×" in expressions_for(5.0)
    assert expressions_for(342) == ["342"]
    assert expressions_for(759) == ["759"]
    assert expressions_for(1210) == ["1210", "1,210"]


def test_rewrite_fires_when_the_fragility_expression_is_missing(monkeypatch):
    prompts: list[str] = []
    _install_worry_stubs(monkeypatch)

    def cites_a_multiplier(prompt, **kwargs):
        prompts.append(prompt)
        if "did not use the results" in prompt:
            return "ANSWER: The bridge score is 0.91 and removing 5% halves the network."
        if "TOOL RESULT structural_criticality" in prompt:
            return "ANSWER: The bridge score is 0.91. Accounts are 5x more connected."
        if "TOOL RESULT critical_nodes" in prompt:
            return "TOOL: structural_criticality\nPARAMS: {}"
        return 'TOOL: critical_nodes\nPARAMS: {"top_n": 5}'

    monkeypatch.setattr("osi.agent.call_llm", cites_a_multiplier)
    result = run_agent("simple-v1", "what should I be worried about", use_llm=True)
    rewrite = next(prompt for prompt in prompts if "did not use the results" in prompt)
    assert "structural_criticality" in rewrite
    assert "5%" in rewrite
    assert "5%" in result.answer
    assert verify_numbers(result.answer, result.values)


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
