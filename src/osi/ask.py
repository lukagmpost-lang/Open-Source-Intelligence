"""Ask one saved graph a question.

The command routes the question, runs the matching executor, and prints
the answer plus the query that produced it.
"""

from __future__ import annotations

import argparse
import sys

from osi.answer import write_answer
from osi.executors import (
    connectivity,
    discuss,
    explain_node,
    list_communities,
    network_health,
    rank_nodes,
    structural_criticality,
)
from osi.router import route
from osi.store import get_run

EXECUTORS = {
    "rank_nodes": rank_nodes,
    "list_communities": list_communities,
    "network_health": network_health,
    "structural_criticality": structural_criticality,
    "connectivity": connectivity,
    "explain_node": explain_node,
    "discuss": discuss,
}

_UNSUPPORTED = (
    "I don't know how to answer that. "
    "Try 'top 10 by pagerank', 'communities', 'how healthy', "
    "'how fragile', 'path from A to B', or 'tell me about NAME'."
)


def ask(run_id: str, question: str, use_llm: bool = True, *, use_cache: bool = True) -> str:
    """Full pipeline: route → executor → write_answer.

    Raises ValueError if run_id doesn't exist in the store.
    ``use_cache=False`` skips a stored answer and does not write a new one.
    """
    text, _query = _execute(run_id, question, use_llm, use_cache=use_cache)
    return text


def _execute(run_id: str, question: str, use_llm: bool, use_cache: bool = True) -> tuple[str, dict]:
    if get_run(run_id) is None:
        raise ValueError(f"run {run_id} was not found")
    routed = route(question, run_id)
    intent = routed["intent"]
    params = dict(routed["params"])
    if intent not in EXECUTORS:
        return _UNSUPPORTED, {
            "intent": intent,
            "params": params,
            "method": "none",
            "trust": "none",
        }
    result = EXECUTORS[intent](**params)
    text = write_answer(result, use_llm=use_llm, question=question, use_cache=use_cache)
    return text, {
        "intent": result.intent,
        "params": result.params,
        "method": result.method,
        "trust": result.trust,
    }


def _print_report(text: str, query: dict) -> None:
    print(text)
    print()
    print("Query used:")
    print(f"  intent: {query['intent']}")
    print(f"  params: {query['params']}")
    print(f"  method: {query['method']}")
    print(f"  trust: {query['trust']}")


def _emit(run_id: str, question: str, use_llm: bool, use_cache: bool = True) -> int:
    try:
        text, query = _execute(run_id, question, use_llm, use_cache=use_cache)
    except (ValueError, LookupError) as error:
        print(error, file=sys.stderr)
        return 1
    _print_report(text, query)
    return 0


def _interactive(run_id: str, use_llm: bool, use_cache: bool = True) -> int:
    while True:
        try:
            question = input("> ")
        except EOFError:
            print()
            return 0
        if question.strip().casefold() in {"exit", "quit"}:
            return 0
        if not question.strip():
            continue
        try:
            text, query = _execute(run_id, question, use_llm, use_cache=use_cache)
        except LookupError as error:
            print(error, file=sys.stderr)
            continue
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        _print_report(text, query)
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    python3 -m osi.ask --run RUN_ID "question"
    python3 -m osi.ask --run RUN_ID              (interactive)
    python3 -m osi.ask --run RUN_ID --no-llm "q"
    python3 -m osi.ask --run RUN_ID --no-cache "q"

    Prints the answer, a blank line, then a 'Query used:' section
    with intent, params, method, and trust.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m osi.ask",
        description="Ask a saved graph a question.",
    )
    parser.add_argument("--run", required=True, help="saved run id")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="print the template instead of calling a model",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="call the model even when this question was answered before",
    )
    parser.add_argument("question", nargs="?", help="question; omit this to start an interactive session")
    args = parser.parse_args(argv)
    use_llm = not args.no_llm
    use_cache = not args.no_cache
    if get_run(args.run) is None:
        print(f"run {args.run} was not found", file=sys.stderr)
        raise SystemExit(1)
    if args.question:
        code = _emit(args.run, args.question, use_llm, use_cache=use_cache)
        if code:
            raise SystemExit(code)
        return
    code = _interactive(args.run, use_llm, use_cache=use_cache)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
