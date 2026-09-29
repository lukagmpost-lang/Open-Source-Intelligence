import re

from osi.agent import parse_agent_reply, run_agent
from osi.answer import verify_numbers
from osi.ask import ask, main


def test_parse_agent_reply_reads_a_tool_call_and_an_answer():
    kind, name, params = parse_agent_reply('TOOL: rank_nodes\nPARAMS: {"metric": "degree", "top": 5}')
    assert kind == "tool"
    assert name == "rank_nodes"
    assert params == {"metric": "degree", "top": 5}
    kind, prose, _params = parse_agent_reply("ANSWER: The hubs sit in different groups.")
    assert kind == "answer"
    assert prose == "The hubs sit in different groups."


def test_the_loop_stops_after_eight_tool_calls(monkeypatch):
    def always_tool(prompt, **kwargs):
        return 'TOOL: rank_nodes\nPARAMS: {"metric": "degree", "top": 3}'

    monkeypatch.setattr("osi.agent.call_llm", always_tool)
    result = run_agent("simple-v1", "who matters here", use_llm=True)
    assert len(result.tools) == 8
    assert result.tools == ["rank_nodes"] * 8
    assert "While looking at this, I also noticed:" in result.answer


def test_worry_question_on_reddit_2008_uses_three_tools(monkeypatch):
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
