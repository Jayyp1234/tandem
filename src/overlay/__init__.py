"""apply_overlay() — the toggle mechanism per overlay_spec.md §5.

Three conditions:
  - overlay-off    : default (Western-leaning English) prompt
  - noise-on       : length-matched random Naija-vocab tokens (no semantic structure)
  - cultural-on    : structured three-layer Nigerian overlay

Toggling `condition` is the ONLY difference between control and treatment;
all other inputs (history, item, model, seed, temperature) are held constant.
"""
from __future__ import annotations

import hashlib
import random
from typing import Literal

from src.overlay.grounding import format_grounding
from src.overlay.linguistic import sample_markers_balanced, sample_random_markers
from src.overlay.preferences import format_preferences_for_prompt

Condition = Literal["overlay-off", "noise-on", "cultural-on"]


def _stable_seed(persona_id: str) -> int:
    """Deterministic seed for the overlay's RNG, stable across Python processes.

    We use SHA-256 instead of Python's built-in hash() because string hashing is
    randomized per process by default (PYTHONHASHSEED), which would change the
    marker sample between runs and break cache-key reproducibility.
    """
    digest = hashlib.sha256(persona_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


_DEFAULT_PROMPT = """You are an Amazon Beauty product reviewer. Predict how the user below would rate and review the candidate product.

User profile
- Name: {name}
- Recent purchase history (oldest first):
{history}
- Implied preferences from history: {preferences_summary}

Candidate item
- Title: {item_title}
- Brand: {item_brand}
- Description: {item_description}

Output a single JSON object with exactly these two keys:
  "rating": integer 1-5
  "review": string of 60-100 words written as this user would write it
Output the JSON only — no commentary, no markdown.
"""


_OVERLAY_PROMPT = """You are an Amazon Beauty product reviewer giving this Nigerian customer's perspective. Predict how this user would rate and review the candidate product, in their voice.

Voice and language: write in Nigerian English with natural Pidgin code-switching where contextually appropriate. Use authentic markers sparingly — not in every sentence. Examples to draw from: {markers}.

Cultural context to weight when reasoning about the product:
{preferences}

User profile
- Name: {name} ({grounding})
- Recent purchase history (oldest first):
{history}
- Implied preferences from history: {preferences_summary}

Candidate item
- Title: {item_title}
- Brand: {item_brand}
- Description: {item_description}

Output a single JSON object with exactly these two keys:
  "rating": integer 1-5
  "review": string of 60-100 words in Nigerian-English / Pidgin blend, as this user would write it
Output the JSON only — no commentary, no markdown.
"""


def apply_overlay(persona: dict, condition: Condition, item: dict) -> str:
    """Return the simulator's system prompt for the given condition."""
    rng = random.Random(_stable_seed(persona["persona_id"]))
    history_str = _format_history(persona["history_window"])
    item_title = item.get("title", "")
    item_brand = item.get("brand", "")
    item_desc = (item.get("description", "") or "")[:400]

    if condition == "overlay-off":
        return _DEFAULT_PROMPT.format(
            name=persona["default_name"],
            history=history_str,
            preferences_summary=persona.get("preference_summary", ""),
            item_title=item_title,
            item_brand=item_brand,
            item_description=item_desc,
        )

    if condition == "noise-on":
        markers = sample_random_markers(rng, n=10)
        return _OVERLAY_PROMPT.format(
            markers=", ".join(markers),
            preferences="(no specific cultural preferences for this product category)",
            name=persona["default_name"],
            grounding="generic",
            history=history_str,
            preferences_summary=persona.get("preference_summary", ""),
            item_title=item_title,
            item_brand=item_brand,
            item_description=item_desc,
        )

    # cultural-on
    markers = sample_markers_balanced(rng, n=10)
    pref_str = format_preferences_for_prompt(
        item.get("category", ""),
        persona.get("ethnic_hint"),
        persona.get("religious_hint"),
    )
    return _OVERLAY_PROMPT.format(
        markers=", ".join(markers),
        preferences=pref_str,
        name=persona["naija_name"],
        grounding=format_grounding(persona),
        history=history_str,
        preferences_summary=persona.get("preference_summary", ""),
        item_title=item_title,
        item_brand=item_brand,
        item_description=item_desc,
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (no prior history available)"
    lines = []
    for h in history:
        rating = h.get("rating", "?")
        text = (h.get("summary", "") or h.get("review_text", ""))[:80]
        lines.append(f"  - rated {rating}/5: {text}")
    return "\n".join(lines)
