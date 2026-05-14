"""Construct 20 personas from held-out Amazon Beauty users.

Per the SASRec / BERT4Rec / P5 evaluation protocol, each persona has:
  - persona_id, user_id (de-identified)
  - history_window: the last 10 items of their history EXCLUDING the target
  - target_item_id: the held-out next interaction (the positive in NDCG@10)
  - candidate_item_ids: target + 99 random negatives (the standard 1+99 protocol)
  - default_name (English) and naija_name (Nigerian) — toggled by overlay condition
  - ethnic_hint, religious_hint — for the cultural-on overlay grounding

Run via:  make personas
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path("data/beauty_5core")
OUTPUT = Path("data/personas_20.jsonl")

SEED = 42
N_PERSONAS = 20
HISTORY_WINDOW = 10
N_NEGATIVES = 99
MIN_HISTORY_LENGTH = HISTORY_WINDOW + 1   # need at least this many items (window + target)

# --- naming catalogues (overlay_spec.md §4 + cultural_resources.md §4) -------
WESTERN_FIRST_NAMES = [
    "Sarah", "Emily", "Jessica", "Michael", "David", "Jennifer", "Lisa",
    "Amanda", "Nicole", "Rachel", "Andrew", "James", "Kelly", "Rebecca",
    "Stephanie", "Jonathan", "Ashley", "Brandon", "Megan", "Tyler",
]
WESTERN_SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller",
    "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
]

NAIJA_NAMES_BY_GROUP: dict[str, list[tuple[str, str]]] = {
    "Yoruba": [
        ("Adunni", "Adebayo"), ("Olamide", "Adeleke"), ("Tunde", "Ogundimu"),
        ("Folake", "Ojo"),     ("Damilola", "Adeyemi"), ("Bukola", "Adeniran"),
    ],
    "Igbo": [
        ("Adaeze", "Okonkwo"), ("Chinedu", "Eze"),    ("Ngozi", "Nwosu"),
        ("Obinna", "Anyanwu"), ("Chiamaka", "Iwobi"), ("Emeka", "Nnaji"),
    ],
    "Hausa-Fulani": [
        ("Aminu", "Bello"),    ("Hauwa", "Sani"),     ("Ibrahim", "Musa"),
        ("Fatima", "Yusuf"),   ("Aisha", "Garba"),    ("Sadiq", "Lawal"),
    ],
    "Other": [
        ("Eseoghene", "Okoro"), ("Tamunoemi", "Wokoma"),
    ],
}
DEMOGRAPHIC_WEIGHTS = {"Yoruba": 0.21, "Hausa-Fulani": 0.29, "Igbo": 0.18, "Other": 0.32}
RELIGIOUS_BY_GROUP = {
    "Yoruba":       {"Christian": 0.55, "Muslim": 0.40, "Traditional": 0.05},
    "Igbo":         {"Christian": 0.95, "Muslim": 0.04, "Traditional": 0.01},
    "Hausa-Fulani": {"Muslim": 0.95,    "Christian": 0.05},
    "Other":        {"Christian": 0.50, "Muslim": 0.40, "Traditional": 0.10},
}


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(w for _, w in items)
    r = rng.random() * total
    cum = 0.0
    for name, w in items:
        cum += w
        if r <= cum:
            return name
    return items[-1][0]


def _summarize_preferences(history: list[dict], item_meta: dict[str, dict]) -> str:
    titles = [item_meta.get(h["item_id"], {}).get("title", "")[:60] for h in history[-5:]]
    return " | ".join(t for t in titles if t)


def _load_jsonl(path: Path) -> list[dict]:
    """JSONL reader that's robust to Unicode line separators inside strings.

    File iteration splits on \\n only, unlike str.splitlines() which also splits
    on \\u2028, \\u2029, \\x85, etc.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def construct_personas() -> None:
    rng = random.Random(SEED)
    users = _load_jsonl(DATA_DIR / "users.jsonl")
    items_meta = {item["item_id"]: item for item in _load_jsonl(DATA_DIR / "items.jsonl")}
    all_item_ids = list(items_meta.keys())

    # Filter to users with enough history (window + target)
    eligible = [u for u in users if len(u["history"]) >= MIN_HISTORY_LENGTH]
    rng.shuffle(eligible)
    selected = eligible[:N_PERSONAS]
    print(f"  selected {len(selected)} personas from {len(eligible)} eligible users")

    western_first_pool = list(WESTERN_FIRST_NAMES)
    rng.shuffle(western_first_pool)
    naija_pool = [(g, f, l) for g, names in NAIJA_NAMES_BY_GROUP.items() for f, l in names]
    rng.shuffle(naija_pool)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for i, user in enumerate(selected):
            history = user["history"]
            target = history[-1]
            history_window = history[-MIN_HISTORY_LENGTH:-1]
            interacted_ids = {h["item_id"] for h in history}

            # Sample 99 deterministic negatives from items the user has not interacted with.
            persona_rng = random.Random(SEED + i)
            negative_pool = [iid for iid in all_item_ids if iid not in interacted_ids]
            negatives = persona_rng.sample(negative_pool, k=min(N_NEGATIVES, len(negative_pool)))
            candidate_ids = [target["item_id"], *negatives]
            persona_rng.shuffle(candidate_ids)  # don't leak target position

            # Names
            western_first = western_first_pool[i % len(western_first_pool)]
            western_last = WESTERN_SURNAMES[i % len(WESTERN_SURNAMES)]
            ethnic_hint = _weighted_choice(rng, DEMOGRAPHIC_WEIGHTS)
            naija_choice = next(
                ((g, fn, ln) for g, fn, ln in naija_pool if g == ethnic_hint),
                naija_pool[i % len(naija_pool)],
            )
            religious_hint = _weighted_choice(rng, RELIGIOUS_BY_GROUP[ethnic_hint])

            persona = {
                "persona_id": f"p{i:02d}",
                "user_id": user["user_id"],
                "history_window": history_window,
                "target_item_id": target["item_id"],
                "candidate_item_ids": candidate_ids,
                "preference_summary": _summarize_preferences(history_window, items_meta),
                "default_name": f"{western_first} {western_last}",
                "naija_name": f"{naija_choice[1]} {naija_choice[2]}",
                "ethnic_hint": ethnic_hint,
                "religious_hint": religious_hint,
            }
            f.write(json.dumps(persona, ensure_ascii=False) + "\n")

    print(f"  wrote {N_PERSONAS} personas to {OUTPUT} (each with target + {N_NEGATIVES} negatives)")


if __name__ == "__main__":
    construct_personas()
