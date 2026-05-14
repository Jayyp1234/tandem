"""Persona grounding helpers (cultural_resources.md §4 + §5)."""
from __future__ import annotations

HONORIFICS_NAIJA = ["Aunty", "Uncle", "Sista", "Bros", "Ma", "Sir", "Oga"]
HONORIFICS_FORMAL = ["Mr", "Mrs", "Dr"]

CULTURAL_REFERENCES = [
    "owambe (party / event)",
    "asoebi (matching outfits at events)",
    "Sallah",
    "Christmas",
    "harmattan (dry windy season)",
    "Detty December",
    "wedding",
    "naming ceremony",
]


def format_grounding(persona: dict) -> str:
    """Short grounding string for the system prompt: 'Yoruba Christian from Nigeria'."""
    parts = []
    if persona.get("ethnic_hint"):
        parts.append(persona["ethnic_hint"])
    if persona.get("religious_hint"):
        parts.append(persona["religious_hint"])
    parts.append("from Nigeria")
    return " ".join(parts)
