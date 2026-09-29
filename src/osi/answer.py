"""Turn a ResultObject into a sentence.

``call_llm`` posts to an OpenAI-compatible chat endpoint. Ollama is the
default. A hosted provider such as Groq uses the same request and response
shape; only the base URL, model, and Authorization header change.
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

_ROOT = Path(__file__).resolve().parents[2]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# "10 communities" is not this phrase. The zero has to be its own number.
_ZERO_COMMUNITIES = re.compile(r"\b0\s+communities\b", re.IGNORECASE)

# Local Ollama when .env does not choose a provider.
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "phi4-mini"


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
    """Provider, base URL, model, and API key. Ollama is the default."""
    _load_dotenv()
    return {
        "provider": os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().casefold(),
        "base_url": os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL).strip(),
        "model": os.environ.get("LLM_MODEL", _DEFAULT_MODEL).strip(),
        "api_key": os.environ.get("LLM_API_KEY", "").strip(),
    }


def _chat_request(prompt: str, settings: dict[str, str]) -> urllib.request.Request:
    url = settings["base_url"].rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": settings["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # Hosted providers require a bearer token. Local Ollama does not.
    if settings["provider"] != "ollama":
        if not settings["api_key"]:
            raise RuntimeError("LLM_API_KEY is required when LLM_PROVIDER is not ollama")
        headers["Authorization"] = f"Bearer {settings['api_key']}"
    return urllib.request.Request(url, data=payload, headers=headers, method="POST")


def call_llm(prompt: str) -> str:
    """Send one user message and return the assistant text."""
    request = _chat_request(prompt, llm_settings())
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: {error.code} {detail}") from error
    return str(body["choices"][0]["message"]["content"]).strip()


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


def _plain(result: ResultObject) -> str:
    metric = result.params.get("metric")
    if result.intent == "list_communities":
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
    """Every number in the reply has to come from result.values."""
    if not _NUMBER.search(prose):
        return False
    return verify_numbers(prose, result.values)


def _misreads_one_community(prose: str, result: ResultObject) -> bool:
    """A community id of 0 is not a count of zero communities."""
    if result.intent != "list_communities":
        return False
    if result.values.get("n_communities") != 1:
        return False
    return _ZERO_COMMUNITIES.search(prose) is not None


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
    not call the model. A sentence is stored only after the number check
    accepts it. ``use_cache=False`` skips both the lookup and the store.
    """
    text = _plain(result)
    if not use_llm:
        return text
    cached_as = _cache_key(result, question) if use_cache else None
    if cached_as is not None:
        key, _run_id = cached_as
        stored = get_cached_answer(key)
        if stored is not None:
            return stored
    prompt = (
        "Rewrite this graph result in one or two sentences. "
        "Use only the numbers written below. Do not invent scores, counts, or names.\n\n"
        + text
    )
    try:
        prose = call_llm(prompt)
    except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
        return text
    if not prose or not _numbers_match(prose, result) or _misreads_one_community(prose, result):
        return text
    missing = [caveat for caveat in result.caveats if caveat not in prose]
    if missing:
        prose = prose.rstrip() + " " + " ".join(missing)
    if cached_as is not None:
        key, run_id = cached_as
        put_cached_answer(key, prose, run_id)
    return prose
