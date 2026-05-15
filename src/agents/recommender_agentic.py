"""TANDEM agentic recommender (plan -> score -> reflect -> re-rank).

The default recommender in ``src.agents.recommender`` is intentionally
deterministic (argsort by predicted rating, tie-break by review length).
That deterministic design is load-bearing for the H7 architectural ablation:
any LLM call in the recommender would confound the cultural-effect
attribution.

The competition brief, however, asks for "agentic workflows that reason
before recommending." This module is the additional mode that answers that
ask without disturbing the H7 baseline. It composes three reasoning steps:

  1. PLAN     — one LLM call that produces a short list of user priorities
                conditioned on the persona and a brief view of the candidate
                pool. This is the "reason-before" step.
  2. SCORE    — for each candidate, the existing simulator agent emits a
                predicted review + rating. Cached on
                (persona_id, item_id, condition, architecture).
  3. REFLECT  — one LLM call that takes the persona's priorities together
                with the top-N scored candidates and returns a re-ranked
                shortlist with one-sentence justifications. This is the
                self-correction / reflection step.

Total cost per request with C candidates: C simulator calls + 2 reasoning
calls (plan, reflect). All calls hit the same cache as the rest of the
system so repeated requests cost zero tokens.

This module never touches the H7 ablation results; the original
``src.agents.recommender`` remains the canonical Task-B implementation.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.agents.simulator import predict as simulator_predict
from src.llm.cache import deterministic_seed
from src.llm.client import MODEL_FAST, GroqClient

_PLAN_PROMPT = """You are advising a recommendation system on what matters most for the user described below. Output the user's top three practical decision priorities for products like the candidates listed.

User profile:
- Name: {name}
- Ethnic context: {ethnic_hint}
- Religious orientation: {religious_hint}
- Preference summary: {preference_summary}
- Recent purchases (most recent first):
{history}

Candidate pool (first ten titles, illustrative):
{candidate_titles}

Output exactly three priorities, one per line, each prefixed with "- ".
Each priority must be a concrete decision criterion (e.g., "halal ingredients", "matches dark complexion", "Nigerian-founded brand") rather than a generic platitude. No preamble, no commentary."""

_REFLECT_PROMPT = """You are a recommendation reviewer. You will receive a user's prioritised decision criteria and a tentative top-{n} ranking produced by a separate scoring agent. Re-order the top-{n} so the order better reflects the user's priorities, and give a one-sentence reason for each.

User priorities:
{priorities}

Tentative top-{n} (rating in brackets):
{candidates}

Output a single JSON array of length {n}, in your preferred order, with this exact schema:
[{{"item_id": "...", "rank": 1, "reason": "<one sentence>"}}, ...]

Output JSON only, no Markdown fences."""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  - (no prior history)"
    lines = []
    for h in history[-5:]:
        rating = h.get("rating", "?")
        summary = _truncate((h.get("summary") or h.get("review_text", "")), 80)
        lines.append(f"  - rated {rating}/5: {summary}")
    return "\n".join(lines)


def _format_candidates_for_plan(items: list[dict]) -> str:
    return "\n".join(f"  - {_truncate(it.get('title', ''), 80)}" for it in items[:10])


def _parse_priorities(text: str) -> list[str]:
    """Extract bullet lines from the planner's output."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•\d\.\)]+\s*", "", line).strip()
        if line:
            out.append(line)
        if len(out) == 3:
            break
    return out or ["(planner produced no parsable priorities)"]


def _format_reflect_candidates(scored: list[dict]) -> str:
    lines = []
    for i, r in enumerate(scored, start=1):
        title = _truncate(r.get("item_title", r.get("item_id", "")), 60)
        rating = r.get("predicted_rating", 0.0)
        review = _truncate(r.get("predicted_review", ""), 120)
        lines.append(f"{i}. [{rating:.1f}] item_id={r['item_id']}  title={title}\n     reason: {review}")
    return "\n".join(lines)


def _parse_reflect_json(text: str, fallback_order: list[str]) -> list[dict]:
    """Robust JSON extraction from the reflector's output. Falls back to the
    pre-reflection order if parsing fails (so the endpoint never errors out
    on a malformed JSON response).
    """
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if not m:
        return [{"item_id": iid, "rank": i + 1, "reason": "(reflector output unparsable; kept score-order)"}
                for i, iid in enumerate(fallback_order)]
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [{"item_id": iid, "rank": i + 1, "reason": "(reflector JSON malformed; kept score-order)"}
                for i, iid in enumerate(fallback_order)]
    seen: set[str] = set()
    cleaned: list[dict] = []
    for i, row in enumerate(parsed):
        iid = str(row.get("item_id", "")).strip()
        if not iid or iid in seen or iid not in fallback_order:
            continue
        seen.add(iid)
        cleaned.append({
            "item_id": iid,
            "rank": i + 1,
            "reason": str(row.get("reason", "")).strip()[:240],
        })
    for iid in fallback_order:
        if iid not in seen:
            cleaned.append({"item_id": iid, "rank": len(cleaned) + 1, "reason": "(reflector omitted; appended in score-order)"})
    return cleaned


def plan(client: GroqClient, persona: dict, items: list[dict]) -> list[str]:
    """Reasoning step 1: identify the user's top-3 decision priorities."""
    prompt = _PLAN_PROMPT.format(
        name=persona.get("naija_name") or persona.get("default_name", "User"),
        ethnic_hint=persona.get("ethnic_hint", "unspecified"),
        religious_hint=persona.get("religious_hint", "unspecified"),
        preference_summary=persona.get("preference_summary", "(none)"),
        history=_format_history(persona.get("history_window", [])),
        candidate_titles=_format_candidates_for_plan(items),
    )
    seed = deterministic_seed(persona.get("persona_id", "p_request"), "plan-v1")
    response = client.cached_complete(
        prompt=prompt, seed=seed, model=MODEL_FAST,
        temperature=0.2, max_tokens=160,
    )
    return _parse_priorities(response.text)


def reflect(client: GroqClient, persona: dict, priorities: list[str], scored_top: list[dict]) -> list[dict]:
    """Reasoning step 3: re-rank the top-N with explicit justifications."""
    n = len(scored_top)
    prompt = _REFLECT_PROMPT.format(
        n=n,
        priorities="\n".join(f"  - {p}" for p in priorities),
        candidates=_format_reflect_candidates(scored_top),
    )
    seed = deterministic_seed(
        persona.get("persona_id", "p_request"),
        "reflect-v1",
        "|".join(r["item_id"] for r in scored_top),
    )
    response = client.cached_complete(
        prompt=prompt, seed=seed, model=MODEL_FAST,
        temperature=0.1, max_tokens=480,
    )
    fallback = [r["item_id"] for r in scored_top]
    return _parse_reflect_json(response.text, fallback)


def recommend_agentic(
    client: GroqClient,
    persona: dict,
    candidates: list[dict],
    top_k: int = 5,
    reflect_window: int = 5,
    condition: str = "cultural-on",
) -> dict[str, Any]:
    """End-to-end agentic recommendation. Returns the full reasoning trace
    so the response is auditable (the brief asks for "reason before
    recommending" --- this is what reasoning looks like exposed).
    """
    if not candidates:
        raise ValueError("candidates is empty")

    priorities = plan(client, persona, candidates)

    scored: list[dict] = []
    cached_hits = 0
    for item in candidates:
        rec = simulator_predict(
            client=client,
            persona=persona,
            item=item,
            condition=condition,
            architecture="decomposed",
        )
        if rec.get("cached"):
            cached_hits += 1
        scored.append({
            "item_id":          item["item_id"],
            "item_title":       item.get("title", ""),
            "predicted_rating": rec["predicted_rating"],
            "predicted_review": rec["predicted_review"],
            "cached":           rec.get("cached", False),
        })

    scored.sort(key=lambda r: (-r["predicted_rating"], -len(r.get("predicted_review", ""))))

    pre_reflection = scored[: max(reflect_window, top_k)]
    reflected = reflect(client, persona, priorities, pre_reflection)

    by_id = {r["item_id"]: r for r in scored}
    ranked: list[dict] = []
    for row in reflected[:top_k]:
        s = by_id.get(row["item_id"])
        if s is None:
            continue
        ranked.append({
            "item_id":          row["item_id"],
            "rank":             row["rank"],
            "reason":           row["reason"],
            "predicted_rating": s["predicted_rating"],
            "predicted_review": s["predicted_review"],
            "title":            s["item_title"],
        })

    return {
        "priorities": priorities,
        "ranked":     ranked,
        "trace": {
            "n_candidates":            len(candidates),
            "reflect_window":          len(pre_reflection),
            "score_calls_cached":      cached_hits,
            "score_calls_total":       len(candidates),
            "reasoning_calls":         2,
            "score_order_top_k":       [r["item_id"] for r in pre_reflection[:top_k]],
            "agentic_order_top_k":     [r["item_id"] for r in ranked],
        },
    }
