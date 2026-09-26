"""A small public-shaped example used by the local viewer."""

from osi.graph import Graph


def sample_graph() -> Graph:
    graph = Graph()
    for login in ("ada", "grace", "linus", "guido", "knuth", "richie"):
        graph.add_node("github", login, login)
    for name in ("ada", "grace"):
        graph.add_node("reddit", name, name)
    graph.add_node("telegram", "100", "ada")
    graph.add_node("telegram", "200", "grace")
    graph.add_node("reddit", "r/python", "r/python", {"kind": "subreddit"})

    graph.add_edge("github:ada", "github:grace", "mutual", 1)
    graph.add_edge("github:ada", "github:linus", "mutual", 1)
    graph.add_edge("github:grace", "github:linus", "mutual", 1)
    graph.add_edge("github:guido", "github:knuth", "mutual", 1)
    graph.add_edge("github:guido", "github:richie", "mutual", 1)
    graph.add_edge("github:knuth", "github:richie", "mutual", 1)
    graph.add_edge("github:linus", "github:guido", "follow", 1)
    graph.add_edge("reddit:ada", "reddit:r/python", "comment", 2)
    graph.add_edge("reddit:grace", "reddit:r/python", "comment", 1)
    graph.add_edge("telegram:200", "telegram:100", "gift", 1)
    graph.add_edge("github:ada", "reddit:ada", "same_as", 1, {"asserted_by": "input"})
    graph.add_edge("github:grace", "reddit:grace", "same_as", 1, {"asserted_by": "input"})
    graph.add_edge("reddit:ada", "telegram:100", "same_as", 1, {"asserted_by": "input"})
    return graph
