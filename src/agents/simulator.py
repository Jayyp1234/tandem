"""TANDEM simulator agent — decomposed and monolithic variants.

Both variants call the same Groq LLaMA-3.1-8B endpoint; the difference is the
final instruction line of the system prompt:

  - decomposed: simulator's task is purely persona-conditioned review/rating prediction.
  - monolithic: simulator's task additionally embeds the ranking-recommendation
                framing inside the same prompt — testing whether decomposition
                isolates the simulator from ranking-context confounds (H7).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from src.llm.cache import deterministic_seed
from src.llm.client import MODEL_FAST, GroqClient, LLMResponse
from src.overlay import Condition, apply_overlay

Architecture = Literal["decomposed", "monolithic"]

_DECOMPOSED_INSTRUCTION = (
    "Your sole task is to predict — as faithfully as possible — how this user "
    "would rate and review this candidate item. Be honest about negatives where "
    "the user's history suggests they would dislike it. Do not adjust the rating "
    "for any other purpose."
)
_MONOLITHIC_INSTRUCTION = (
    "Your task is to predict how this user would rate and review the candidate "
    "item, with the understanding that the predicted rating will be used to RANK "
    "this candidate against other items for recommendation. Higher predicted "
    "ratings will surface this candidate higher in the user's feed."
)


def _build_system_prompt(persona: dict, condition: Condition,
                         architecture: Architecture, item: dict) -> str:
    base = apply_overlay(persona, condition, item)
    tail = _DECOMPOSED_INSTRUCTION if architecture == "decomposed" else _MONOLITHIC_INSTRUCTION
    return base + "\n\n" + tail


def predict(
    client: GroqClient,
    persona: dict,
    item: dict,
    condition: Condition,
    architecture: Architecture,
    model: str = MODEL_FAST,
) -> dict:
    """One simulator prediction. Cached deterministically per
    (persona_id, item_id, condition, architecture).
    """
    seed = deterministic_seed(
        persona["persona_id"], item["item_id"], condition, architecture
    )
    system_prompt = _build_system_prompt(persona, condition, architecture, item)

    response = client.cached_complete(
        prompt="Output the JSON object now.",
        seed=seed,
        model=model,
        temperature=0.7,
        max_tokens=240,
        system=system_prompt,
    )
    return _to_record(response, persona, item, condition, architecture)


def _to_record(response: LLMResponse, persona: dict, item: dict,
               condition: str, architecture: str) -> dict:
    rating, review = _parse_json_response(response.text)
    return {
        "persona_id": persona["persona_id"],
        "item_id": item["item_id"],
        "condition": condition,
        "architecture": architecture,
        "predicted_rating": rating,
        "predicted_review": review,
        "model": response.model,
        "cached": response.cached,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
    }


def _parse_json_response(text: str) -> tuple[float, str]:
    """Parse rating + review from the LLM output, robust to formatting noise."""
    # Locate first JSON object (greedy enough to grab nested-quote reviews)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            rating_raw = obj.get("rating", 3)
            rating = float(rating_raw)
            rating = max(1.0, min(5.0, rating))
            review = str(obj.get("review", "")).strip()
            return rating, review
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: look for a "rating: X" pattern; default to 3.0 + raw text.
    m = re.search(r'rating["\s:]+([1-5])', text, re.IGNORECASE)
    rating = float(m.group(1)) if m else 3.0
    return rating, text.strip()[:400]


def run_cell(
    client: GroqClient,
    personas: list[dict],
    items_meta: dict[str, dict],
    condition: Condition,
    architecture: Architecture,
    output_path: Path,
) -> None:
    """Run one (condition × architecture) cell over each persona's candidate set.

    Per persona, we predict on `candidate_item_ids` (target + 99 negatives) —
    the standard SASRec-style protocol.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = sum(len(p["candidate_item_ids"]) for p in personas)
    pbar = tqdm(total=n_total, desc=f"{architecture}/{condition}")

    with output_path.open("w", encoding="utf-8") as f:
        for persona in personas:
            for iid in persona["candidate_item_ids"]:
                item = items_meta.get(iid)
                if item is None:
                    pbar.update(1)
                    continue
                rec = predict(client, persona, item, condition, architecture)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.update(1)
    pbar.close()
