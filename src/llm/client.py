"""Groq LLM client with caching, multi-key rotation, and rate-limit handling.

Per research/decisions_and_critique.md D6: primary LLM is LLaMA-3.1-8B-Instant via
Groq's free tier. The free tier caps tokens-per-day (TPD) at 500K, which our
experiment matrix exceeds. We support rotation across up to six API keys; when
the current key returns a TPD error, the client switches to the next available
key automatically.

NOTE: cycling separate free-tier accounts is against Groq's terms of service.
Use this at your own risk; the alternative is upgrading one account's Dev Tier
(actual cost for the full matrix is < $1).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.llm.cache import CacheEntry, CacheKey, JsonlCache

load_dotenv()


# --- pinned model identifiers (per experiment_matrix.md §0) ------------------

MODEL_FAST = "llama-3.1-8b-instant"           # default; 500K TPD per key on free tier
MODEL_HEAVY = "llama-3.3-70b-versatile"       # hard cases; 100K TPD per key

# --- rate limit -------------------------------------------------------------
# Groq free-tier RPM = 30, so 2.0 s/call gives 30 calls/min; 2.2 s for safety.
MIN_INTERVAL_SECONDS = 2.2

# Max distinct keys we look for in env (GROQ_API_KEY_1 .. GROQ_API_KEY_16).
MAX_KEYS = 16


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cached: bool


def _load_api_keys() -> list[str]:
    """Discover available Groq keys from the environment.

    Looks for GROQ_API_KEY_1 .. GROQ_API_KEY_{MAX_KEYS} (rotation set) and the
    plain GROQ_API_KEY (single-key fallback). Deduplicates, ignores placeholders
    and empty values, preserves the order GROQ_API_KEY_1 → ... → GROQ_API_KEY.
    """
    keys: list[str] = []

    def _add(k: str | None) -> None:
        if k is None:
            return
        k = k.strip()
        if not k or k == "gsk_REPLACE_ME":
            return
        if k not in keys:
            keys.append(k)

    for i in range(1, MAX_KEYS + 1):
        _add(os.environ.get(f"GROQ_API_KEY_{i}"))
    _add(os.environ.get("GROQ_API_KEY"))
    return keys


def _is_tpd_error(err: Exception) -> bool:
    """Return True if the exception is a Groq daily-token-quota error."""
    msg = str(err).lower()
    return "tokens per day" in msg or "tpd" in msg


class GroqClient:
    """Thin wrapper over the official `groq` SDK with caching, retries, and
    automatic key rotation across multiple Groq accounts.

    Use `cached_complete` for everything that should hit the cache (i.e.
    deterministic experiments). The cache key includes seed and temperature; it
    is independent of which API key produced the response.
    """

    def __init__(
        self,
        cache_path: Path | str = "cache/llm_responses.jsonl",
        api_key: str | None = None,
        api_keys: list[str] | None = None,
    ):
        from groq import Groq

        if api_keys is None:
            if api_key is not None:
                api_keys = [api_key]
            else:
                api_keys = _load_api_keys()

        if not api_keys:
            raise RuntimeError(
                "No GROQ_API_KEY found. Set GROQ_API_KEY (single) or "
                "GROQ_API_KEY_1 .. GROQ_API_KEY_6 (rotation) in .env. "
                "Get a free key at https://console.groq.com/keys"
            )

        self.api_keys = api_keys
        self._clients = [Groq(api_key=k) for k in api_keys]
        self._current_idx = 0
        self._exhausted: set[int] = set()
        self.cache = JsonlCache(cache_path)
        self._last_call_time = 0.0

        if len(api_keys) > 1:
            print(
                f"  GroqClient loaded with {len(api_keys)} keys; "
                f"auto-rotates on tokens-per-day exhaustion."
            )

    # ---- key management ----------------------------------------------------

    @property
    def _client(self):
        return self._clients[self._current_idx]

    def _rotate_key(self) -> bool:
        """Mark current key as exhausted, switch to next available.

        Returns True if rotation succeeded, False if all keys are exhausted.
        """
        self._exhausted.add(self._current_idx)
        for i in range(len(self.api_keys)):
            if i not in self._exhausted:
                self._current_idx = i
                print(
                    f"  → key #{i + 1} of {len(self.api_keys)} "
                    f"(previous hit tokens-per-day; "
                    f"{len(self._exhausted)} key(s) exhausted)"
                )
                self._last_call_time = 0.0  # let next call go through immediately
                return True
        return False

    # ---- rate limiting -----------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        self._last_call_time = time.monotonic()

    # ---- API call with rotation + retry -----------------------------------

    def _call_api(
        self,
        prompt: str,
        model: str,
        seed: int,
        temperature: float,
        max_tokens: int = 256,
        system: str | None = None,
    ) -> LLMResponse:
        """Single API call with key rotation on TPD errors and exponential
        backoff on other transient errors."""
        max_attempts = len(self.api_keys) + 5
        last_error: Exception | None = None

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_attempts):
            try:
                self._wait_for_rate_limit()
                start = time.monotonic()
                completion = self._client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                latency_ms = int((time.monotonic() - start) * 1000)
                return LLMResponse(
                    text=completion.choices[0].message.content or "",
                    model=model,
                    input_tokens=(
                        completion.usage.prompt_tokens if completion.usage else -1
                    ),
                    output_tokens=(
                        completion.usage.completion_tokens if completion.usage else -1
                    ),
                    latency_ms=latency_ms,
                    cached=False,
                )
            except Exception as e:  # noqa: BLE001
                last_error = e
                if _is_tpd_error(e):
                    if not self._rotate_key():
                        raise RuntimeError(
                            f"All {len(self.api_keys)} Groq keys exhausted on "
                            f"tokens-per-day. Wait for daily reset (typically "
                            f"24h from first use), or add more keys to .env."
                        ) from e
                    # Immediate retry on the next key, no backoff.
                    continue
                # Other transient error: backoff and retry on the same key.
                wait = min(2 ** attempt, 60)
                time.sleep(wait)

        # Exhausted attempts.
        if last_error is not None:
            raise last_error
        raise RuntimeError("API call failed after retries (no underlying error)")

    # ---- public: cached completion ----------------------------------------

    def cached_complete(
        self,
        prompt: str,
        seed: int,
        model: str = MODEL_FAST,
        temperature: float = 0.7,
        max_tokens: int = 256,
        system: str | None = None,
    ) -> LLMResponse:
        """Run a completion, hitting the JSONL cache if seen before.

        The cache key is `sha256(system||prompt, model, seed, temperature)`. It
        does NOT include which API key produced the response — the same model
        on different free-tier accounts yields the same outputs for a given
        seed.
        """
        full_prompt = f"<system>{system}</system>\n{prompt}" if system else prompt
        key = CacheKey(prompt=full_prompt, model=model, seed=seed, temperature=temperature)

        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(
                text=hit.response_text,
                model=hit.model,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                latency_ms=hit.latency_ms,
                cached=True,
            )

        resp = self._call_api(
            prompt=prompt,
            model=model,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        self.cache.put(
            key,
            CacheEntry(
                key_hash=key.to_hash(),
                prompt=full_prompt,
                model=model,
                seed=seed,
                temperature=temperature,
                response_text=resp.text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_ms=resp.latency_ms,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )
        return resp


def smoke_test() -> None:
    """Tiny end-to-end test: prints the number of keys loaded and runs one call.

    Run via:  python -m src.llm.client
    """
    client = GroqClient()
    print(f"  keys loaded: {len(client.api_keys)}")
    resp = client.cached_complete(
        prompt="In one sentence: what is a recommendation system?",
        seed=42,
        temperature=0.0,
        max_tokens=80,
    )
    print(f"  model:    {resp.model}")
    print(f"  cached:   {resp.cached}")
    print(f"  latency:  {resp.latency_ms} ms")
    print(f"  tokens:   {resp.input_tokens} in / {resp.output_tokens} out")
    print(f"  response: {resp.text}")


if __name__ == "__main__":
    smoke_test()
