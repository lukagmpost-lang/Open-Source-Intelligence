"""Interactive REPL for the OSI conversation system."""
from __future__ import annotations

import sys

from osi.converse import converse


def main():
    from osi.loaders.whatsapp import _parse_export
    from osi.store import get_run, load_graph

    print("loading wa-real graph...")
    meta = get_run("wa-real")
    if meta is None:
        print("no wa-real run found. Upload a chat first.")
        return 1
    layer = (meta.get("config") or {}).get("layer", "whatsapp")
    g = load_graph("wa-real", layer)
    if g is None:
        print("no graph for wa-real")
        return 1

    print("parsing messages...")
    _, messages = _parse_export("/Users/2950639/Downloads/_chat.txt")

    print("graph:", g.number_of_nodes(), "nodes,", g.number_of_edges(), "edges")
    print("messages:", len(messages))
    print()
    print("Ask a question. Follow-ups remember the previous answer.")
    print("Type 'exit' to quit, 'verbose' to toggle code display.")
    print()

    conversation = []
    verbose = False

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q in ("exit", "quit"):
            break
        if q == "verbose":
            verbose = not verbose
            print("verbose:", verbose)
            continue
        if q == "history":
            for h in conversation:
                print(" ", h["role"], ":", h["content"][:120])
            continue

        result = converse(q, g, messages, verbose=verbose, history=conversation)
        print()
        print(result["answer"])
        print()
        print("(rounds:", result["rounds"], "| history size:", len(conversation), ")")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
