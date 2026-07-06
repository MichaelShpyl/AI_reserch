"""Backend A: real commercial LLM backends for the commercial-vs-local comparison (pipeline
component 4). The core research contribution is comparing a commercial LLM against the local
open model on the SAME question-generation task and the SAME discrimination measure.

ATU cannot provide API access or credits (Aoife Hill, 25 June 2026), so the plan agreed with the
project owner is: use a FREE-TIER commercial API first (Google Gemini via AI Studio), and fall
back to a small self-funded spend on Claude or GPT if needed.

Every backend exposes the same `chat_json(system, user) -> dict` interface as the local
OllamaBackend, so the rest of the pipeline does not change. Keys are read from environment
variables; a local `.env` is loaded if present (and `.env` is gitignored, so keys never get
committed):

  Gemini     GEMINI_API_KEY  or GOOGLE_API_KEY     free tier: https://aistudio.google.com/apikey
  Anthropic  ANTHROPIC_API_KEY                      self-funded
  OpenAI     OPENAI_API_KEY                         self-funded

  python src/question_gen/generate_questions.py --backend commercial --provider gemini --id 3108a
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

# Best-effort .env loading so a key dropped in a local .env is picked up without exporting it.
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

# Default models per provider. Gemini Flash is free-tier and fast; the Anthropic default follows
# the claude-api guidance (claude-opus-4-8), and GPT defaults to a cheap model. All overridable
# with --model so cost stays in the project owner's control.
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o-mini",
}

_RETRYABLE = {429, 500, 502, 503, 504}


def _extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating ```json fences and surrounding prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(t[i:j + 1])
        raise ValueError(f"No JSON object found in model reply: {text[:200]!r}")


def _post_with_retry(url: str, headers: dict, body: dict, timeout: int = 120,
                     retries: int = 3) -> dict:
    """POST JSON with simple exponential backoff on rate-limit and server errors (free tiers
    return 429 readily)."""
    last = None
    for attempt in range(retries):
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in _RETRYABLE and attempt < retries - 1:
            wait = float(r.headers.get("retry-after", 2 ** (attempt + 1)))
            time.sleep(min(wait, 30))
            last = r
            continue
        raise RuntimeError(f"{url} returned {r.status_code}: {r.text[:300]}")
    raise RuntimeError(
        f"{url} failed after {retries} attempts: "
        f"{last.status_code if last is not None else '?'} {last.text[:300] if last is not None else ''}")


def _require_key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(
        f"No API key found. Set one of {', '.join(names)} (in the environment or a local .env). "
        f"Free Gemini key: https://aistudio.google.com/apikey")


# ---------------- providers ----------------
class GeminiBackend:
    """Google Gemini via the AI Studio REST API. Free tier with modest rate limits, which suits a
    research-scale comparison. Not Anthropic, so plain requests rather than an SDK keeps the
    dependency footprint small (relevant on the 8 GB laptop)."""

    def __init__(self, model: str = DEFAULT_MODELS["gemini"], temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.name = f"commercial:gemini:{model}"
        self._key = _require_key("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def chat_json(self, system: str, user: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "responseMimeType": "application/json"},
        }
        data = _post_with_retry(url, {"x-goog-api-key": self._key,
                                      "content-type": "application/json"}, body)
        if "error" in data:
            raise RuntimeError(f"Gemini error: {data['error']}")
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data)[:300]}")
        return _extract_json(text)


class AnthropicBackend:
    """Anthropic Claude via the official `anthropic` Python SDK (per the claude-api guidance, the
    SDK is the right surface for Python). Self-funded path. The SDK reads ANTHROPIC_API_KEY and
    retries 429/5xx itself."""

    def __init__(self, model: str = DEFAULT_MODELS["anthropic"], temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.name = f"commercial:anthropic:{model}"
        _require_key("ANTHROPIC_API_KEY")
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError("The anthropic SDK is needed for the Claude backend: "
                               "pip install anthropic") from e

    def chat_json(self, system: str, user: str) -> dict:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model, max_tokens=2048,
            system=system + " Reply with JSON only, no prose, no code fences.",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _extract_json(text)


class OpenAIBackend:
    """OpenAI GPT via the Chat Completions REST API (non-Anthropic, so plain requests). Self-funded
    alternative; JSON mode forces a parseable object."""

    def __init__(self, model: str = DEFAULT_MODELS["openai"], temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.name = f"commercial:openai:{model}"
        self._key = _require_key("OPENAI_API_KEY")

    def chat_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model, "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        data = _post_with_retry("https://api.openai.com/v1/chat/completions",
                                {"Authorization": f"Bearer {self._key}",
                                 "content-type": "application/json"}, body)
        return _extract_json(data["choices"][0]["message"]["content"])


_PROVIDERS = {"gemini": GeminiBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}


def make_commercial_backend(provider: str = "gemini", model: str | None = None,
                            temperature: float = 0.2):
    """Build a commercial backend by provider name. `model=None` uses the provider default."""
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {list(_PROVIDERS)}")
    return _PROVIDERS[provider](model or DEFAULT_MODELS[provider], temperature)
