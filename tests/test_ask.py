import pytest

from osi.ask import ask, main


def test_ask_returns_a_string_for_top_pagerank(monkeypatch):
    monkeypatch.setattr(
        "osi.answer.call_llm",
        lambda prompt: "Alice leads PageRank at 0.415481.",
    )
    text = ask("simple-v1", "top 3 by pagerank")
    assert isinstance(text, str)
    assert "0.415481" in text


def test_ask_returns_a_helpful_message_for_an_unsupported_question():
    text = ask("simple-v1", "what is the weather")
    assert "I don't know how to answer that" in text
    assert "top 10 by pagerank" in text
    assert "0.415481" not in text


def test_ask_missing_run_raises_value_error():
    with pytest.raises(ValueError, match="was not found"):
        ask("no-such-run-ask", "top 3 by pagerank")


def test_ask_without_llm_returns_the_template(monkeypatch):
    def fail(prompt):
        raise AssertionError("call_llm should not run")

    monkeypatch.setattr("osi.answer.call_llm", fail)
    text = ask("simple-v1", "top 3 by pagerank", use_llm=False)
    assert text.startswith("Top 3 by pagerank:")
    assert "0.415481" in text
    assert "trust is stable" in text


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
