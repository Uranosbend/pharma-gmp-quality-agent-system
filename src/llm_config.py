"""Free-tier LLM factory, live model discovery, 404 fallback, and HTTP 429 retry resilience."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PROVIDER_CATALOG: Dict[str, Dict[str, Any]] = {
    "groq": {
        "label": "Groq Free (Recommended)",
        "default_model": "openai/gpt-oss-120b",
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        "default_pacing_sec": 0.3,
        "env_keys": ["GROQ_API_KEY"],
        "requires_key": True,
    },
    "gemini": {
        "label": "Google Gemini Free (AI Studio)",
        "default_model": "gemini-3.5-flash-lite",
        "models": [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
        ],
        "default_pacing_sec": 1.0,
        "env_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "requires_key": True,
    },
    "ollama": {
        "label": "Ollama Local (Offline)",
        "default_model": "qwen2.5:7b",
        "models": [
            "qwen2.5:7b",
            "llama3.1:8b",
            "mistral:7b",
        ],
        "default_pacing_sec": 0.0,
        "env_keys": ["OLLAMA_BASE_URL"],
        "requires_key": False,
    },
}


def resolve_api_key(provider: str, sidebar_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Resolve credentials cleanly: Sidebar -> st.secrets -> .env/os.environ."""
    load_dotenv(override=False)

    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip(), "Sidebar Input"

    cfg = PROVIDER_CATALOG.get(provider, {})
    env_keys: List[str] = cfg.get("env_keys", [])

    if provider == "ollama":
        url = os.getenv("OLLAMA_BASE_URL", "").strip()
        if not url:
            try:
                import streamlit as st

                url = str(st.secrets.get("OLLAMA_BASE_URL", "")).strip()
            except Exception:
                url = ""
        return (url or "http://localhost:11434"), "Local Endpoint"

    # 1. Check Streamlit Community Cloud / local .streamlit/secrets.toml safely
    try:
        import streamlit as st

        for k in env_keys:
            try:
                val = st.secrets.get(k)
            except Exception:
                val = None
            if val and str(val).strip():
                return str(val).strip(), f"Streamlit Secrets ({k})"
    except Exception:
        pass

    # 2. Check .env / OS environment variables
    for k in env_keys:
        val = os.getenv(k, "").strip()
        if val:
            return val, f".env / Env ({k})"

    return None, "Missing"


def detect_key_provider_mismatch(provider: str, api_key: Optional[str]) -> Optional[str]:
    """Warn immediately if an API key prefix clearly belongs to a different provider."""
    if not api_key:
        return None
    k = api_key.strip()
    if provider == "groq" and k.startswith("AIza"):
        return "This key starts with 'AIza' (Google Gemini format). Switch Provider to Google Gemini Free or enter a Groq key ('gsk_...')."
    if provider == "gemini" and k.startswith("gsk_"):
        return "This key starts with 'gsk_' (Groq format). Switch Provider to Groq Free or enter a Google AI Studio key ('AIza...')."
    if k.startswith("sk-"):
        return "This key starts with 'sk-' (OpenAI/Anthropic format). Enter a free Groq ('gsk_...') or Google AI Studio ('AIza...') key."
    return None


def _sort_models_by_preference(discovered: List[str], preferred: List[str]) -> List[str]:
    """Order discovered accessible models so proven tool-calling defaults come first."""
    rank = {name: i for i, name in enumerate(preferred)}
    unique = list(dict.fromkeys(discovered))
    return sorted(unique, key=lambda m: (rank.get(m, 999), m))


def fetch_live_models(provider: str, api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """Query the provider's live model catalog for the supplied API key/endpoint."""
    fallback = list(PROVIDER_CATALOG.get(provider, {}).get("models", []))
    resolved_key, _ = resolve_api_key(provider, api_key)

    if provider == "groq":
        if not resolved_key:
            return fallback, "No Groq API key supplied."
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {resolved_key}"},
                timeout=6,
            )
            if resp.status_code != 200:
                return fallback, f"Groq /models returned HTTP {resp.status_code}"
            data = resp.json().get("data", [])
            exclude_markers = ("whisper", "guard", "tts", "safeguard", "embed", "rerank", "distil", "orpheus", "allam")
            chat_models = [
                str(item.get("id"))
                for item in data
                if item.get("id")
                and item.get("active", True) is not False
                and not any(m in str(item.get("id")).lower() for m in exclude_markers)
            ]
            if not chat_models:
                return fallback, "No active chat models returned by Groq."
            return _sort_models_by_preference(chat_models, fallback), None
        except Exception as exc:
            return fallback, f"Could not query Groq models: {exc}"

    if provider == "gemini":
        if not resolved_key:
            return fallback, "No Gemini API key supplied."
        try:
            from google import genai

            client = genai.Client(api_key=resolved_key)
            exclude_markers = ("embed", "imagen", "veo", "tts", "aqa", "1.0", "1.5", "2.5", "robotics", "computer-use")
            discovered: List[str] = []
            for m in client.models.list():
                name = str(getattr(m, "name", "") or "").replace("models/", "")
                actions = [str(a).lower() for a in (getattr(m, "supported_actions", None) or [])]
                if not name:
                    continue
                if actions and not any("generatecontent" in a for a in actions):
                    continue
                low = name.lower()
                if "gemini" not in low or any(ex in low for ex in exclude_markers):
                    continue
                discovered.append(name)
            if not discovered:
                return fallback, "No active Gemini generateContent models discovered."
            return _sort_models_by_preference(discovered, fallback), None
        except Exception as exc:
            return fallback, f"Could not query Gemini models: {exc}"

    if provider == "ollama":
        base_url = (resolved_key or "http://localhost:11434").rstrip("/")
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return fallback, f"Ollama /api/tags returned HTTP {resp.status_code}"
            names = [str(m.get("name")) for m in resp.json().get("models", []) if m.get("name")]
            if not names:
                return fallback, "No local Ollama models pulled yet."
            return _sort_models_by_preference(names, fallback), None
        except Exception as exc:
            return fallback, f"Ollama not reachable at {base_url}: {exc}"

    return fallback, None


def is_model_not_found_error(exc: Exception) -> bool:
    """Detect HTTP 404 / model_not_found / unsupported model errors."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "model_not_found",
        "does not exist or you do not have access",
        "does not exist",
        "not found for api version",
        "is not supported for generatecontent",
        "not_found",
        "error code: 404",
        "404 not found",
        "status: 404",
    )
    return any(m in msg for m in markers)


def resolve_fallback_model(
    provider: str,
    failed_model: str,
    api_key: Optional[str] = None,
    tried_models: Optional[List[str]] = None,
) -> Optional[str]:
    """Find an accessible fallback model when the selected model returns 404."""
    excluded = {failed_model} | set(tried_models or [])
    live_models, _ = fetch_live_models(provider, api_key)
    for m in live_models:
        if m not in excluded:
            return m
    for m in PROVIDER_CATALOG.get(provider, {}).get("models", []):
        if m not in excluded:
            return m
    return None


def get_chat_model(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
) -> Any:
    """Instantiate a LangChain chat model for Groq, Gemini, or Ollama."""
    resolved_key, _ = resolve_api_key(provider, api_key)

    if provider == "groq":
        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it in Streamlit Secrets, .env, or the sidebar."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=resolved_key,
            temperature=temperature,
            max_retries=0,
        )

    if provider == "gemini":
        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY / GEMINI_API_KEY is missing. Add it in Streamlit Secrets, .env, or the sidebar."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=resolved_key,
            temperature=temperature,
            max_retries=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = resolved_key or "http://localhost:11434"
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _is_transient_rate_or_network_error(exc: Exception) -> bool:
    """Detect HTTP 429, quota throttling, or transient gateway/timeout errors."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "429",
        "rate limit",
        "ratelimit",
        "rate_limit",
        "resourceexhausted",
        "resource_exhausted",
        "quota",
        "too many requests",
        "503",
        "502",
        "500",
        "overloaded",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
    )
    return any(m in msg for m in transient_markers)


def invoke_with_retry(
    callable_fn: Callable[..., Any],
    *args: Any,
    pacing_delay: float = 0.3,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Execute an LLM call with configurable pacing and exponential backoff on 429s."""
    if pacing_delay > 0:
        time.sleep(pacing_delay)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_transient_rate_or_network_error(exc):
                raise
            sleep_sec = min(base_delay * (2**attempt) + random.uniform(0.1, 0.75), 20.0)
            time.sleep(sleep_sec)

    assert last_exc is not None
    raise last_exc


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from raw LLM text or fenced code blocks."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
    return json.loads(cleaned)


def invoke_structured_decision(
    llm: Any,
    schema_cls: Type[T],
    messages: List[Any],
    pacing_delay: float = 0.3,
) -> T:
    """Obtain a Pydantic-validated structured output with JSON text fallback."""
    try:
        structured_llm = llm.with_structured_output(schema_cls)
        result = invoke_with_retry(structured_llm.invoke, messages, pacing_delay=pacing_delay)
        if isinstance(result, schema_cls):
            return result
        if isinstance(result, dict):
            return schema_cls.model_validate(result)
    except Exception as first_err:
        if _is_transient_rate_or_network_error(first_err) or is_model_not_found_error(first_err):
            raise
        raw_msg = invoke_with_retry(llm.invoke, messages, pacing_delay=pacing_delay)
        content = getattr(raw_msg, "content", str(raw_msg))
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", part) if isinstance(part, dict) else part) for part in content
            )
        data = extract_json_object(str(content))
        return schema_cls.model_validate(data)

    raise ValueError(f"Could not validate structured response into {schema_cls.__name__}")
