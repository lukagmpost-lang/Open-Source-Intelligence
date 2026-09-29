"""Turn a ResultObject into a sentence.

``call_llm`` posts to an OpenAI-compatible chat endpoint. Ollama is the
default. A hosted provider such as Groq uses the same request and response
shape; only the base URL, model, and Authorization header change.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from osi.result import ResultObject

_ROOT = Path(__file__).resolve().parents[2]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

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


def _plain(result: ResultObject) -> str:
    metric = result.params.get("metric")
    if result.intent == "rank_nodes" and metric:
        head = f"Top {result.params.get('top', len(result.values))} by {metric}:"
    else:
        head = f"{result.intent}:"
    body = _format_values(result.values)
    trust = f"This result is {result.method} and trust is {result.trust}."
    caveat = " ".join(result.caveats)
    return " ".join(part for part in (head, body + ".", trust, caveat) if part)


def _collect_numbers(value, found: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        found.append(float(value))
        return
    if isinstance(value, str):
        found.extend(float(item) for item in _NUMBER.findall(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, found)
        return
    if isinstance(value, list):
        for item in value:
            _collect_numbers(item, found)


def _numbers_match(prose: str, result: ResultObject) -> bool:
    """Every number in the reply has to be one of the stored numbers."""
    found = [float(item) for item in _NUMBER.findall(prose)]
    if not found:
        return False
    allowed: list[float] = []
    _collect_numbers(result.to_dict(), allowed)
    for number in found:
        if not any(abs(number - candidate) <= max(0.005, abs(candidate) * 0.05) for candidate in allowed):
            return False
    return True


def write_answer(result: ResultObject, use_llm: bool = False) -> str:
    """Plain sentence from the result. A model may rephrase it, not change the numbers."""
    text = _plain(result)
    if not use_llm:
        return text
    prompt = (
        "Rewrite this graph result in one or two sentences. "
        "Use only the numbers written below. Do not invent scores, counts, or names.\n\n"
        + text
    )
    try:
        prose = call_llm(prompt)
    except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):
        return text
    if not prose or not _numbers_match(prose, result):
        return text
    missing = [caveat for caveat in result.caveats if caveat not in prose]
    if missing:
        return prose.rstrip() + " " + " ".join(missing)
    return prose
