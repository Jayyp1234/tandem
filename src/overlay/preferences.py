"""Cultural preference axes for Nigerian beauty consumers (overlay_spec.md §3)."""
from __future__ import annotations

# Each axis: trigger_categories (which item categories activate it), tokens
# (the cultural-vocabulary anchors), guidance (a one-line note for the LLM).
PREFERENCE_AXES: dict[str, dict] = {
    "skin_tone": {
        "trigger_categories": ["foundation", "concealer", "bb cream", "skin", "powder"],
        "tokens": ["dudu (dark)", "funfun (light)", "yellow undertone",
                   "deep complexion", "ebony", "honey", "caramel"],
        "guidance": "Pay close attention to undertone and the broad melanin-rich complexion range.",
    },
    "hair_type": {
        "trigger_categories": ["hair", "wig", "weave", "shampoo", "conditioner", "edge"],
        "tokens": ["4C", "kinky", "natural hair", "edges", "shrinkage",
                   "weave", "braids", "locs", "relaxed"],
        "guidance": "Many users have textured / 4C hair; humidity resistance and edge control matter.",
    },
    "religious": {
        "trigger_categories": ["all"],
        "tokens": ["halal", "alcohol-free", "kosher", "natural"],
        "guidance": "Some users prefer alcohol-free or halal-compliant cosmetics for religious reasons.",
    },
    "ingredients": {
        "trigger_categories": ["skin", "soap", "butter", "oil", "lotion", "cream", "moisturizer"],
        "tokens": ["shea butter", "ori", "black soap", "dudu-osun",
                   "palm kernel oil", "neem", "coconut oil"],
        "guidance": "Culturally-affirmed ingredients (shea butter / ori, black soap / dudu-osun) are highly valued.",
    },
    "climate": {
        "trigger_categories": ["foundation", "fragrance", "sunscreen", "lotion", "spray"],
        "tokens": ["humidity-resistant", "sweat-proof", "harmattan",
                   "heat-stable", "long-wear"],
        "guidance": "Nigeria is hot and humid most of the year, with a dusty harmattan season; products must hold up.",
    },
    "brand_familiarity": {
        "trigger_categories": ["all"],
        "tokens": ["Zaron", "House of Tara", "Iman", "Sleek", "Black Up",
                   "Mented", "Tropical Naturals", "Dudu-Osun"],
        "guidance": "Pan-African and Nigerian-founded brands are often preferred for melanin-rich-skin compatibility.",
    },
}

# Tokens used by H4 (cultural-topic coverage rate) — `phase3_findings.md` §F1.
CULTURAL_TOPIC_TOKENS: list[str] = sorted({
    tok.lower()
    for axis in PREFERENCE_AXES.values()
    for tok in axis["tokens"]
})


def axes_for_category(item_category: str) -> list[dict]:
    """Return the preference axes relevant to the given product category."""
    cat_lower = (item_category or "").lower()
    relevant: list[dict] = []
    for name, axis in PREFERENCE_AXES.items():
        triggers = axis["trigger_categories"]
        if "all" in triggers or any(t in cat_lower for t in triggers):
            relevant.append({"name": name, **axis})
    return relevant


def format_preferences_for_prompt(
    item_category: str,
    ethnic_hint: str | None = None,
    religious_hint: str | None = None,
) -> str:
    """One-block summary of culturally-relevant preferences for the system prompt."""
    axes = axes_for_category(item_category)
    if not axes:
        return "(no specific cultural preferences for this product category)"

    lines: list[str] = []
    for axis in axes:
        # Suppress halal markers for explicit Christian users (cleaner conditioning)
        if axis["name"] == "religious" and religious_hint == "Christian":
            continue
        lines.append(f"- {axis['name'].replace('_', ' ')}: {axis['guidance']}")
        lines.append(f"  vocabulary: {', '.join(axis['tokens'][:5])}")
    return "\n".join(lines)


def cultural_topic_coverage(text: str) -> int:
    """Count occurrences of cultural-topic vocabulary in `text` (used by H4)."""
    text_lower = text.lower()
    return sum(1 for tok in CULTURAL_TOPIC_TOKENS if tok in text_lower)
