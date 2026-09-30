import pytest

from osi.ask import ask, main


def test_ask_returns_a_string_for_top_pagerank(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt, **kwargs: "Alice is more central than the other accounts.",
    )
    text = ask("simple-v1", "top 3 by pagerank", use_cache=False)
    assert isinstance(text, str)
    assert text.startswith("Alice is more central than the other accounts.")
    assert "0.415481" not in text


def test_ask_returns_a_helpful_message_for_an_unsupported_question():
    text = ask("simple-v1", "   ")
    assert "I don't know how to answer that" in text
    assert "top 10 by pagerank" in text
    assert "0.415481" not in text


def test_ask_explains_an_open_question_from_the_graph(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt, **kwargs: "Alice, Bob, and Carol form a triangle, and Dave is only tied to Alice.",
    )
    text = ask("simple-v1", "why is the graph shaped this way", use_cache=False)
    assert "triangle" in text
    assert "Dave" in text
    assert "I don't know how to answer that" not in text


def test_ask_missing_run_raises_value_error():
    with pytest.raises(ValueError, match="was not found"):
        ask("no-such-run-ask", "top 3 by pagerank")


def test_ask_without_llm_returns_the_template(monkeypatch):
    def fail(prompt, **kwargs):
        raise AssertionError("call_llm should not run")

    monkeypatch.setattr("osi.answer.call_llm", fail)
    text = ask("simple-v1", "top 3 by pagerank", use_llm=False)
    assert text.startswith("The most central accounts are ")
    assert "alice" in text
    assert "How sure: stable." in text


def test_no_cache_flag_skips_a_stored_answer(monkeypatch, capsys):
    seen: dict[str, bool] = {}

    def fake_execute(run_id, question, use_llm, use_cache=True):
        seen["use_cache"] = use_cache
        return "Alice at 0.415481.", {
            "intent": "rank_nodes",
            "params": {"run": run_id},
            "method": "exact",
            "trust": "stable",
        }

    monkeypatch.setattr("osi.ask.get_run", lambda run_id: {"id": run_id})
    monkeypatch.setattr("osi.ask._execute", fake_execute)
    main(["--run", "simple-v1", "--no-cache", "top 3 by pagerank"])
    assert seen["use_cache"] is False
    assert "Alice at 0.415481." in capsys.readouterr().out


def test_interactive_loop_terminates_on_exit(monkeypatch, capsys):
    prompts: list[str] = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if len(prompts) > 2:
            raise AssertionError("interactive loop did not stop")
        return "exit"

    monkeypatch.setattr("builtins.input", fake_input)
    main(["--run", "simple-v1"])
    assert prompts == ["> "]
    assert "Query used:" not in capsys.readouterr().out
