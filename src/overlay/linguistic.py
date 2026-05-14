"""Naija / Nigerian-Pidgin linguistic markers (per phase3_cultural_resources.md §2)."""
from __future__ import annotations

import random

# (token, category, register)
NAIJA_MARKERS: list[tuple[str, str, str]] = [
    # Greetings
    ("how far", "greeting", "casual"),
    ("wetin dey happen", "greeting", "casual"),
    ("how body", "greeting", "casual"),
    ("good good", "greeting", "casual"),
    # Intensifiers
    ("no be small thing", "intensifier", "either"),
    ("die", "intensifier", "casual"),
    ("well well", "intensifier", "either"),
    ("sotay", "intensifier", "casual"),
    ("too much", "intensifier", "casual"),
    # Hedges
    ("I think say", "hedge", "either"),
    ("small small", "hedge", "either"),
    # Exclamations
    ("ah ah", "exclamation", "casual"),
    ("chai", "exclamation", "casual"),
    ("na wa", "exclamation", "casual"),
    ("kai", "exclamation", "casual"),
    # Discourse fillers / particles
    ("abeg", "filler", "casual"),
    ("jare", "filler", "casual"),
    ("o", "particle", "casual"),
    ("now now", "filler", "casual"),
    ("sef", "particle", "casual"),
    # Connectives
    ("make I", "connective", "either"),
    ("e be like say", "connective", "casual"),
    # Quality / value descriptors
    ("correct", "quality", "casual"),
    ("solid", "quality", "either"),
    ("real", "quality", "casual"),
    ("original", "quality", "either"),
    ("sharp", "quality", "casual"),
    # Common Pidgin verbs / aspect markers
    ("dey", "auxiliary", "casual"),
    ("don", "aspect", "casual"),
    ("go", "modal", "either"),
    # Common nouns
    ("wahala", "noun", "casual"),
    ("oga", "noun", "casual"),
    ("madam", "noun", "either"),
    # Address forms
    ("aunty", "address", "casual"),
    ("uncle", "address", "casual"),
    ("bros", "address", "casual"),
    ("baba", "address", "casual"),
    # Reactions
    ("e shock me", "reaction", "casual"),
    ("nothing do am", "reaction", "casual"),
    ("e too much", "reaction", "casual"),
    ("na real deal", "reaction", "casual"),
]

ALL_TOKENS = [t for t, _, _ in NAIJA_MARKERS]


def sample_markers_balanced(rng: random.Random, n: int = 10) -> list[str]:
    """Sample n markers, balanced across categories. Used for cultural-on prompt."""
    by_cat: dict[str, list[str]] = {}
    for token, cat, _ in NAIJA_MARKERS:
        by_cat.setdefault(cat, []).append(token)
    cats = list(by_cat.keys())
    rng.shuffle(cats)
    selected: list[str] = []
    while len(selected) < n and any(by_cat.values()):
        for cat in cats:
            pool = by_cat.get(cat, [])
            if not pool or len(selected) >= n:
                continue
            selected.append(pool.pop(rng.randrange(len(pool))))
    return selected[:n]


def sample_random_markers(rng: random.Random, n: int = 10) -> list[str]:
    """Length-matched random sample for the noise-on condition."""
    return rng.sample(ALL_TOKENS, k=min(n, len(ALL_TOKENS)))


def naija_token_count(text: str) -> int:
    """Count Naija marker occurrences in `text` (case-insensitive substring match)."""
    text_lower = text.lower()
    return sum(text_lower.count(t) for t in ALL_TOKENS)
