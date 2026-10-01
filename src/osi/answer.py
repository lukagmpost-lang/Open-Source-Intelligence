"""Turn a ResultObject into a sentence.

``call_llm`` posts to an OpenAI-compatible chat endpoint. Groq is the
default. Another provider uses the same request and response shape; only
the base URL, model, and Authorization header change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from osi.result import ResultObject
from osi.store import get_cached_answer, put_cached_answer
from osi.vocabulary import get_source, vocabulary_instruction

_ROOT = Path(__file__).resolve().parents[2]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# "10 communities" is not this phrase. The zero has to be its own number.
_ZERO_COMMUNITIES = re.compile(r"\b0\s+communities\b", re.IGNORECASE)

# Groq free tier when .env does not choose a provider.
_DEFAULT_PROVIDER = "groq"
_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _dotenv_path() -> Path:
    return _ROOT / ".env"


def _load_dotenv() -> None:
    """Fill empty variables from .env. A value already in the environment wins."""
    path = _dotenv_path()
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def llm_settings() -> dict[str, str]:
    """Provider, base URL, model, and API key. Groq is the default."""
    _load_dotenv()
    return {
        "provider": os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().casefold(),
        "base_url": os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL).strip(),
        "model": os.environ.get("LLM_MODEL", _DEFAULT_MODEL).strip(),
        "api_key": os.environ.get("LLM_API_KEY", "").strip(),
    }


_DOMAIN_ANALOGIES = {
    "forum": "a small town with distinct neighborhoods",
    "company": "an office where different teams work on different floors",
    "community": "a neighborhood where people know each other by face",
    "activist": "a network of safe houses connected by trusted couriers",
    "general": "a group of people connected by shared interests",
    "social": "a neighborhood where people know each other by face",
}

_DOMAIN_LIST = (
    "   - forum: a small town with distinct neighborhoods\n"
    "   - company: an office where different teams work on different floors\n"
    "   - community: a neighborhood where people know each other by face\n"
    "   - activist: a network of safe houses connected by trusted couriers\n"
    "   - general: a group of people connected by shared interests"
)

WRITING_RULES = (
    "Write 3-5 sentences that a non-expert could read and immediately understand.\n"
    "\n"
    "Rules:\n"
    "- Lead with what happened, not what the numbers are.\n"
    "- Use one analogy per answer. Compare the network to something concrete: "
    "a small town, an office, a party, a neighborhood, a broadcast tower.\n"
    "- End with what this means for someone who uses or runs this network. "
    'Not "the network is fragile" but "if the top accounts leave, most people '
    'lose their connection to each other."\n'
    '- Do not use metric names. No "modularity", "clustering", "assortativity", '
    '"components", "density".\n'
    "- Do not quote raw numbers unless the number itself is the point. "
    '"1,890 connections" is fine. "541x more pieces" is fine. "0.02494" is not.\n'
    "- Every number must appear in a tool result.\n"
    "- Never quote a raw centrality score (pagerank, betweenness, closeness, "
    "eigenvector). Express importance as a ratio to the typical account: "
    "'36x more central than typical' or 'more connected than 99% of accounts'."
)

ANSWER_EXAMPLE = (
    "Example of a good answer for a Reddit 2012 network:\n"
    "\n"
    '"Reddit in 2012 was a completely different animal from what it was in 2008. '
    "It went from a small, tightly-knit community to a sprawling platform with over "
    "a thousand separate groups. The biggest account, CosmicBard, has 1,890 connections "
    "— that's like knowing everyone in a small town. Meanwhile the platform itself is "
    "shattered into pieces: 541× more disconnected fragments than a normal network. "
    "Most forums have one cohesive community. This one has over a thousand, which means "
    "most users never interact with each other. The platform isn't a community anymore "
    "— it's a collection of micro-communities sharing a URL.\"\n"
    "\n"
    "Write in this style."
)


def domain_context(domain: str | None) -> str:
    """The concrete picture the writer should use for this kind of network."""
    key = (domain or "general").strip().casefold() or "general"
    analogy = _DOMAIN_ANALOGIES.get(key, _DOMAIN_ANALOGIES["general"])
    label = key if key in _DOMAIN_ANALOGIES else "general"
    return (
        f"This is a {label} network. A {label} is like {analogy}:\n"
        f"{_DOMAIN_LIST}\n"
        "\n"
        "Use this frame when you write. When you describe a hub, "
        'say what it would mean in this frame: "one person who knows everyone in the small town."'
    )


def answer_system_prompt(domain: str | None = "general", source: str | None = "general") -> str:
    """System prompt for a plain-language answer about one network."""
    return "\n\n".join(
        [WRITING_RULES, vocabulary_instruction(source), domain_context(domain), ANSWER_EXAMPLE]
    )


def findings_system_prompt(
    domain: str | None = "general", source: str | None = "general"
) -> str:
    """System prompt when the executor already wrote the findings."""
    preface = (
        "You receive findings about a network. Rewrite them as flowing prose. "
        "Do not invent numbers. Do not add metrics."
    )
    return preface + "\n\n" + answer_system_prompt(domain, source)


SYSTEM_PROMPT = answer_system_prompt("general")
FINDINGS_SYSTEM_PROMPT = findings_system_prompt("general")

# Every completion uses the same sampling settings. Seed is omitted after a
# provider rejects it, and the result then carries NONDETERMINISTIC_CAVEAT.
LLM_TEMPERATURE = 0
LLM_SEED = 42
NONDETERMINISTIC_CAVEAT = "Non-deterministic: this answer may vary between runs."
_SEED_SUPPORTED = True


def llm_seed_supported() -> bool:
    """False after the provider rejects the seed field."""
    return _SEED_SUPPORTED


def note_determinism(result: ResultObject) -> None:
    """Record that repeated runs can differ when the provider has no seed."""
    if _SEED_SUPPORTED or NONDETERMINISTIC_CAVEAT in result.caveats:
        return
    result.caveats.append(NONDETERMINISTIC_CAVEAT)


def _disable_seed() -> None:
    global _SEED_SUPPORTED
    _SEED_SUPPORTED = False


def _seed_rejected(error: BaseException) -> bool:
    text = str(error).lower()
    return ("400" in text or "422" in text) and "seed" in text


def _chat_request(
    prompt: str,
    settings: dict[str, str],
    system: str | None = None,
    *,
    temperature: float = LLM_TEMPERATURE,
    seed: int | None = LLM_SEED,
) -> urllib.request.Request:
    url = settings["base_url"].rstrip("/") + "/chat/completions"
    body: dict = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if seed is not None and _SEED_SUPPORTED:
        body["seed"] = seed
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        # Cloudflare rejects urllib's default client signature with error 1010.
        "User-Agent": "osi/0.1",
    }
    # Hosted providers require a bearer token. Local Ollama does not.
    if settings["provider"] != "ollama":
        if not settings["api_key"]:
            raise RuntimeError(
                "LLM_API_KEY is not set. Get a free Groq key at "
                "https://console.groq.com/keys and add it to .env as "
                "LLM_API_KEY=your_key_here"
            )
        headers["Authorization"] = f"Bearer {settings['api_key']}"
    return urllib.request.Request(url, data=payload, headers=headers, method="POST")


def _complete(
    prompt: str,
    settings: dict[str, str],
    system: str | None,
    temperature: float,
    seed: int | None,
) -> str:
    request = _chat_request(prompt, settings, system, temperature=temperature, seed=seed)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: {error.code} {detail}") from error
    return str(body["choices"][0]["message"]["content"]).strip()


def call_llm(
    prompt: str,
    *,
    run_id: str = "",
    n_nodes: int | None = None,
    n_edges: int | None = None,
    question: str = "",
    system: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    seed: int | None = LLM_SEED,
) -> str:
    """Send the system rules plus one user message and return the assistant text.

    When a run is known, the user message starts with the network's name and size.
    Temperature is 0. A seed is sent when the provider accepts one.
    """
    if run_id or question or n_nodes is not None or n_edges is not None:
        context = (
            f'Context: this result is from a network called "{run_id}" '
            f'with {n_nodes} nodes and {n_edges} edges. The question was: "{question}".'
        )
        prompt = context + "\n\n" + prompt
    settings = llm_settings()
    actual_seed = seed if _SEED_SUPPORTED else None
    try:
        return _complete(prompt, settings, system, temperature, actual_seed)
    except RuntimeError as error:
        if actual_seed is None or not _seed_rejected(error):
            raise
        _disable_seed()
        return _complete(prompt, settings, system, temperature, None)


def _format_values(values: dict) -> str:
    parts: list[str] = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key} {value:.6f}")
        elif isinstance(value, dict):
            parts.append(f"{key} ({_format_values(value)})")
        else:
            parts.append(f"{key} {value}")
    return ", ".join(parts)


def _communities_text(values: dict) -> str:
    """Name each community field so an id of 0 is not read as a count."""
    count = int(values["n_communities"])
    noun = "community" if count == 1 else "communities"
    sizes = ", ".join(str(size) for size in values["sizes"]) or "none"
    modularity = values.get("modularity")
    if modularity is None:
        mod = "not available"
    else:
        mod = f"{float(modularity):.6f}"
    largest = values.get("largest_community")
    largest_text = "none" if largest is None else str(largest)
    return (
        f"n_communities is {count} ({noun}). "
        f"modularity is {mod}. "
        f"sizes are {sizes}. "
        f"largest_community is {largest_text}. "
        f"largest_size is {values['largest_size']}"
    )


def _score_list(scores: dict) -> str:
    parts = []
    for key, value in scores.items():
        if isinstance(value, float):
            parts.append(f"{key} {value:.6f}")
        else:
            parts.append(f"{key} {value}")
    return ", ".join(parts)


def _num(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "unknown"
    return str(value)


def _discuss_notes(values: dict) -> str:
    """A readable brief. The model answers from this, and it is also the fallback."""
    nodes = values.get("nodes", 0)
    edges = values.get("edges", 0)
    components = values.get("components", 0)
    component_word = "component" if components == 1 else "components"
    sentences = [
        f"The graph has {nodes} nodes and {edges} edges in {components} {component_word}.",
        (
            f"Density is {_num(values.get('density'))}, with {values.get('triangles')} triangles. "
            f"Average clustering is {_num(values.get('clustering'))}. "
            f"Average degree is {_num(values.get('avg_degree'))} and the highest degree is {_num(values.get('max_degree'))}."
        ),
        (
            f"Louvain finds {values.get('communities')} communities "
            f"(modularity {_num(values.get('modularity'))}, largest community {values.get('largest_community')} nodes)."
        ),
    ]
    ranks = values.get("top_pagerank") or {}
    if ranks:
        sentences.append(f"Highest PageRank: {_score_list(ranks)}.")
    links = []
    for row in values.get("structure") or []:
        neighbors = row.get("neighbors") or []
        if not neighbors:
            links.append(f"{row.get('node')} (degree {row.get('degree')}) has no neighbors")
            continue
        joined = ", ".join(
            f"{item.get('node')} (weight {item.get('weight')})" for item in neighbors
        )
        links.append(f"{row.get('node')} (degree {row.get('degree')}) links to {joined}")
    if links:
        sentences.append("Adjacency: " + "; ".join(links) + ".")
    return " ".join(sentences)


def _and_list(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _trust_reason(result: ResultObject) -> str:
    if result.trust == "unstable":
        return "unstable. A small change in the network can rearrange this"
    if result.trust == "moderate":
        return "only moderate, because this is not a complete reading"
    if result.trust == "random":
        return "low, because this looks close to chance"
    return "stable"


_METRIC_WORDS = {
    "pagerank": "PageRank, a score of how important an account is",
    "degree": "how many connections they have",
    "betweenness": "betweenness, meaning how often they sit on paths between others",
    "closeness": "closeness, meaning how near they are to everyone else",
}


def _rank_template(result: ResultObject) -> str:
    metric = str(result.params.get("metric") or "importance")
    metric_text = _METRIC_WORDS.get(metric, metric)
    names = [str(node) for node in result.values]
    top_node = names[0] if names else "nobody"
    top_value = _num(next(iter(result.values.values()))) if result.values else "unknown"
    return (
        f"The most important nodes by {metric_text} are {_and_list(names)}. "
        f"The top one is {top_node} at {top_value}. "
        f"How sure: {_trust_reason(result)}."
    )


def _community_template(result: ResultObject) -> str:
    count = result.values.get("n_communities", 0)
    largest = result.values.get("largest_size", 0)
    return (
        f"This network has {count} communities. "
        f"The largest contains {largest} nodes. "
        f"How sure: {_trust_reason(result)}."
    )


def _connection_count(result: ResultObject, degree) -> str:
    if isinstance(degree, bool) or not isinstance(degree, (int, float)):
        return str(degree)
    if isinstance(degree, float) and result.n_nodes and result.n_nodes > 1 and not float(degree).is_integer():
        return str(int(round(float(degree) * (result.n_nodes - 1))))
    if isinstance(degree, float) and float(degree).is_integer():
        return str(int(degree))
    return str(degree)


def _neighbors_text(neighbors) -> str:
    names = []
    for item in neighbors or []:
        if isinstance(item, dict):
            names.append(str(item.get("node")))
        else:
            names.append(str(item))
    return _and_list(names)


def _explain_template(result: ResultObject) -> str:
    values = result.values
    node = result.params.get("node", "This account")
    degree = _connection_count(result, values.get("degree"))
    rank = _num(values.get("pagerank"))
    community = values.get("community")
    return (
        f"{node} is connected to {degree} others, has a PageRank of {rank}, "
        f"and sits in community {community}. "
        f"Their strongest connections are {_neighbors_text(values.get('neighbors'))}. "
        f"How sure: {_trust_reason(result)}."
    )


_CENTRALITY_METRICS = {"pagerank", "betweenness", "closeness", "eigenvector"}
_COMPARATIVE_PHRASES = ("more central", "x times", "compared to", "more connected")


def _centrality_scores(result: ResultObject) -> list[float]:
    """Floats that are pagerank, betweenness, closeness, or eigenvector."""
    metric = str(result.params.get("metric") or "").lower()
    scores: list[float] = []

    def take(value) -> None:
        if isinstance(value, bool) or not isinstance(value, float):
            return
        if 0 < value <= 1:
            scores.append(value)

    def walk(value, under_metric: bool) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "findings":
                    continue
                name = str(key).lower()
                named = name in _CENTRALITY_METRICS
                if named and isinstance(item, (int, float)) and not isinstance(item, bool):
                    take(float(item))
                    continue
                walk(item, under_metric or named)
            return
        if isinstance(value, list):
            for item in value:
                walk(item, under_metric)
            return
        if under_metric or metric in _CENTRALITY_METRICS:
            take(value)

    walk(result.values, False)
    return scores


def _ratio_sentence(result: ResultObject, extra_findings: list[str] | None = None) -> str:
    texts = [str(item) for item in (result.values.get("findings") or [])]
    texts.extend(str(item) for item in (extra_findings or []))
    for text in texts:
        lowered = text.lower()
        if "more central" in lowered or "more connected" in lowered:
            return text
    return "express importance as a ratio to the typical account"


def _has_comparison(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _COMPARATIVE_PHRASES)


def _quoted_centrality(answer: str, scores: list[float]) -> str | None:
    for match in _NUMBER.finditer(answer):
        token = match.group(0)
        if "." not in token:
            continue
        if _close_to_stored(token, float(token), scores):
            return token
    return None


def raw_centrality_problem(
    answer: str,
    result: ResultObject,
    extra_findings: list[str] | None = None,
) -> str | None:
    """Reject a raw centrality decimal, or an answer that never makes the comparison.

    Returns the rewrite instruction, or None when the answer is fine.
    """
    scores = _centrality_scores(result)
    if not scores:
        return None
    quoted = _quoted_centrality(answer, scores)
    sentence = _ratio_sentence(result, extra_findings)
    if quoted is not None and not _has_comparison(answer):
        return (
            f"Your answer quotes the raw centrality score {quoted}. "
            f"Replace it with the comparative ratio from the findings: '{sentence}'. "
            "Do not quote raw metric values."
        )
    if quoted is None and not _has_comparison(answer):
        return (
            "Your answer is missing centrality. "
            f"Replace it with the comparative ratio from the findings: '{sentence}'. "
            "Do not quote raw metric values."
        )
    return None


def _findings_template(result: ResultObject) -> str:
    findings = [str(item) for item in result.values.get("findings") or []]
    return " ".join(findings) + f" How sure: {_trust_reason(result)}."


def templated_fallback(result: ResultObject, scratchpad: dict | None = None) -> str:
    """Plain-language sentence used when the model is off or its reply is rejected.

    An agent scratchpad contributes findings from every tool, not only the last one.
    """
    if scratchpad:
        findings = []
        for _tool_name, tool_result in scratchpad.items():
            findings.extend(tool_result.values.get("findings", []))
        unique = list(dict.fromkeys(findings))
        return " ".join(unique[:6]) + f" How sure: {result.trust}."
    findings = result.values.get("findings")
    if (
        isinstance(findings, list)
        and findings
        and result.intent in {"network_health", "rank_nodes", "list_communities", "discuss"}
    ):
        return _findings_template(result)
    if result.intent == "rank_nodes" and result.params.get("metric"):
        return _rank_template(result)
    if result.intent == "list_communities":
        return _community_template(result)
    if result.intent == "explain_node":
        return _explain_template(result)
    return _plain(result)


def _plain(result: ResultObject) -> str:
    metric = result.params.get("metric")
    if result.intent == "discuss":
        head = ""
        body = _discuss_notes(result.values)
    elif result.intent == "list_communities":
        head = "list_communities:"
        body = _communities_text(result.values)
    elif result.intent == "rank_nodes" and metric:
        head = f"Top {result.params.get('top', len(result.values))} by {metric}:"
        body = _format_values(result.values)
    else:
        head = f"{result.intent}:"
        body = _format_values(result.values)
    trust = f"This result is {result.method} and trust is {result.trust}."
    caveat = " ".join(result.caveats)
    if result.intent == "discuss":
        return " ".join(part for part in (body, trust, caveat) if part)
    return " ".join(part for part in (head, body + ".", trust, caveat) if part)


def _walk_values(value, numbers: list, keys: list) -> None:
    """Collect numeric values and keys. Strings are not scanned."""
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(key)
            _walk_values(item, numbers, keys)
        return
    if isinstance(value, list):
        for item in value:
            _walk_values(item, numbers, keys)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numbers.append(value)


def _is_whole(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _exactly(number: float, candidate) -> bool:
    """True when candidate is the same number, not a nearby one."""
    if isinstance(candidate, bool):
        return False
    if isinstance(candidate, (int, float)):
        return number == candidate
    if isinstance(candidate, str) and _NUMBER.fullmatch(candidate):
        return number == float(candidate)
    return False


def _close_to_stored(token: str, number: float, stored: list) -> bool:
    """Match a written decimal to a stored score at the precision it uses.

    ``0.415`` is three digits, so the allowance is 1e-3.
    ``0.4155`` is four digits, so the allowance is 1e-4.
    A longer writing is tighter: ``0.999999`` is not 1.
    Whole stored numbers (degree, community size, a score of exactly 1)
    match only by equality.
    """
    if "." not in token:
        return False
    places = len(token.split(".", 1)[1])
    tolerance = 10 ** (-places)
    for value in stored:
        if isinstance(value, bool):
            continue
        if _is_whole(value):
            if number == value:
                return True
            continue
        if abs(number - float(value)) < tolerance:
            return True
    return False


def verify_numbers(text: str, values: dict) -> bool:
    """Extract every number from text. For each number N:

    - If N is a written decimal within one unit of its last digit of any
      value in ``values``, ok. ``|0.415 - 0.415481| < 1e-3`` and
      ``|0.4155 - 0.415481| < 1e-4``. A full-precision miss such as
      0.999999 against 0.415481 fails. Integer counts match only exactly.
    - Else if N equals ``len(values)`` or any key in ``values``, ok.
    - Else, fail.

    Numbers are matched by their numeric value, not by string.
    This function only sees ``values``, so a run id or other params
    cannot satisfy the check.
    """
    stored: list = []
    keys: list = []
    _walk_values(values, stored, keys)
    findings = values.get("findings")
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, str):
                for match in _NUMBER.finditer(item):
                    stored.append(float(match.group(0)))
    count = len(values)
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        number = float(token)
        if _close_to_stored(token, number, stored):
            continue
        if "." not in token and any(_is_whole(value) and number == value for value in stored):
            continue
        if number == count or any(_exactly(number, key) for key in keys):
            continue
        return False
    return True


def _numbers_match(prose: str, result: ResultObject) -> bool:
    """Every number in the reply has to come from result.values.

    A discuss answer may explain the shape without quoting a score.
    Any digit it does use still has to be one of the measured values.
    """
    if not _NUMBER.search(prose):
        if result.intent == "discuss":
            return True
        findings = result.values.get("findings")
        return isinstance(findings, list) and bool(findings)
    return verify_numbers(prose, result.values)


def _misreads_one_community(prose: str, result: ResultObject) -> bool:
    """A community id of 0 is not a count of zero communities."""
    if result.intent != "list_communities":
        return False
    if result.values.get("n_communities") != 1:
        return False
    return _ZERO_COMMUNITIES.search(prose) is not None


_SENTENCE_BREAK = re.compile(r"[.!?]+|\n+")
_JARGON = ("modularity", "betweenness", "assortativity", "transitivity", "rich-club")
_DEFINITION = re.compile(
    r"\b(?:means|meaning|defined|that is|i\.e\.|which is|which are)\b|\([^)]+\)",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]


def _jargon_hits(text: str) -> int:
    """How many specialist terms appear, counting each term once."""
    lowered = text.lower()
    hits = 0
    for word in _JARGON:
        pattern = r"\brich-club\b" if word == "rich-club" else rf"\b{word}\b"
        if re.search(pattern, lowered):
            hits += 1
    return hits


def _periphery_is_defined(text: str) -> bool:
    """core-periphery and periphery need a definition in a neighboring sentence."""
    sentences = _sentences(text)
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        if "core-periphery" not in lowered and re.search(r"\bperiphery\b", lowered) is None:
            continue
        window = " ".join(sentences[max(0, index - 1) : index + 2])
        if _DEFINITION.search(window) is None:
            return False
    return True


def verify_style(text: str) -> bool:
    """True when the reply is plain enough to show a non-specialist.

    Rejects a "The graph is" preamble, undefined periphery language,
    three or more specialist terms, and replies longer than six sentences.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"the graph is\b", stripped, re.IGNORECASE):
        return False
    if len(_sentences(stripped)) > 6:
        return False
    if _jargon_hits(stripped) >= 3:
        return False
    return _periphery_is_defined(stripped)


def _size(result: ResultObject, field: str, value_key: str) -> int:
    raw = getattr(result, field)
    if raw is None:
        raw = result.values.get(value_key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return int(raw)


def _result_json(result: ResultObject) -> str:
    payload = {
        "intent": result.intent,
        "params": result.params,
        "values": result.values,
        "method": result.method,
        "trust": result.trust,
        "caveats": result.caveats,
    }
    return json.dumps(payload, default=str)


_RECITED_METRICS = (
    "density",
    "modularity",
    "clustering",
    "triangles",
    "betweenness",
    "assortativity",
    "degree distribution",
    "power law",
)


def _recites_metrics(text: str) -> bool:
    """True when the reply names a metric instead of the network."""
    lowered = text.lower()
    if re.search(r"\bcomponents?\b", lowered):
        return True
    for word in _RECITED_METRICS:
        if " " in word:
            if word in lowered:
                return True
            continue
        if re.search(rf"\b{word}\b", lowered):
            return True
    return False


def answer_cache_key(run_id: str, question: str, values: dict) -> str:
    """sha256 of the run, the question, and the sorted result values."""
    material = run_id + question + str(sorted(values.items()))
    return hashlib.sha256(material.encode()).hexdigest()


def _cache_key(result: ResultObject, question: str | None) -> tuple[str, str] | None:
    """Return (key, run id) when this answer can be stored. Otherwise None."""
    if question is None:
        return None
    run_id = result.params.get("run")
    if not isinstance(run_id, str) or not run_id:
        return None
    try:
        key = answer_cache_key(run_id, question, result.values)
    except TypeError:
        return None
    return key, run_id


def run_domain(result: ResultObject) -> str:
    """Forum, company, social, or general, from the saved run. Otherwise general."""
    run_id = result.params.get("run")
    if not isinstance(run_id, str) or not run_id:
        return "general"
    try:
        from osi.hypotheses import infer_domain
        from osi.store import get_run

        meta = get_run(run_id)
    except (LookupError, OSError, ValueError):
        return "general"
    if not meta:
        return "general"
    config = meta.get("config") or {}
    return infer_domain(meta.get("source") or "", config.get("layer") or "")


def write_answer(
    result: ResultObject,
    use_llm: bool = False,
    *,
    question: str | None = None,
    use_cache: bool = True,
) -> str:
    """Plain sentence from the result. A model may rephrase it, not change the numbers.

    When ``use_llm`` and ``use_cache`` are set and the question is known, a
    repeated run, question, and result returns the stored sentence and does
    not call the model. A sentence is stored only after the number check and
    the style check both accept it. ``use_cache=False`` skips both the lookup
    and the store.
    """
    text = templated_fallback(result)
    if not use_llm:
        return text
    cached_as = _cache_key(result, question) if use_cache else None
    if cached_as is not None:
        key, _run_id = cached_as
        stored = get_cached_answer(key)
        if stored is not None:
            note_determinism(result)
            return stored
    asked = question if question is not None else str(result.params.get("question") or "")
    run_id = result.params.get("run")
    if not isinstance(run_id, str):
        run_id = ""
    domain = run_domain(result)
    source = get_source(run_id)
    findings = result.values.get("findings")
    send_findings = (
        isinstance(findings, list)
        and bool(findings)
        and result.intent in {"network_health", "rank_nodes", "list_communities", "discuss"}
    )

    def _request(prompt: str) -> str | None:
        try:
            if send_findings:
                return call_llm(prompt, system=findings_system_prompt(domain, source))
            return call_llm(
                prompt,
                run_id=run_id,
                n_nodes=_size(result, "n_nodes", "nodes"),
                n_edges=_size(result, "n_edges", "edges"),
                question=asked,
                system=answer_system_prompt(domain, source),
            )
        except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
            return None

    def _acceptable(prose: str | None) -> bool:
        return bool(
            prose
            and _numbers_match(prose, result)
            and not _misreads_one_community(prose, result)
            and verify_style(prose)
            and not _recites_metrics(prose)
        )

    if send_findings:
        finding_text = "\n".join(str(item) for item in findings)
        prose = _request(finding_text)
        if prose and _recites_metrics(prose):
            prose = _request(
                finding_text
                + "\n\nRewrite these findings in plain English. "
                + "Do not use the words density, modularity, clustering, component, "
                + "triangles, betweenness, assortativity, degree distribution, or power law."
            )
    else:
        finding_text = ""
        prose = _request(_result_json(result))
    problem = raw_centrality_problem(prose or "", result)
    if problem:
        prose = _request(problem if not finding_text else finding_text + "\n\n" + problem)
    if not _acceptable(prose) or raw_centrality_problem(prose or "", result):
        note_determinism(result)
        return text
    assert prose is not None
    note_determinism(result)
    missing = [caveat for caveat in result.caveats if caveat not in prose]
    if missing:
        prose = prose.rstrip() + " " + " ".join(missing)
    if send_findings and "How sure:" not in prose:
        prose = prose.rstrip() + f" How sure: {_trust_reason(result)}."
    if cached_as is not None:
        key, run_id = cached_as
        put_cached_answer(key, prose, run_id)
    return prose
