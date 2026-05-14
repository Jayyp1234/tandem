"""Extract 10 paired qualitative examples for the paper's appendix.

For each selected (persona, item), we pair the simulator's overlay-off review
(cell A) with the cultural-on review (cell C). The selection is stratified
random: at most 2 examples per persona, balanced across ethnic-group hints.
Outputs both a JSON for downstream tooling and a Markdown preview for reading.

Run via:  python -m src.eval.qualitative_examples
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/beauty_5core")
RESULTS_DIR = Path("results")
PERSONAS_PATH = Path("data/personas_20.jsonl")
CELL_A_SIM = RESULTS_DIR / "cell_A_simulator.jsonl"
CELL_C_SIM = RESULTS_DIR / "cell_C_simulator.jsonl"
OUTPUT_JSON = RESULTS_DIR / "qualitative_examples.json"
OUTPUT_MD = RESULTS_DIR / "qualitative_examples.md"
N_EXAMPLES = 10
MAX_PER_PERSONA = 2
SEED = 42


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rng = random.Random(SEED)

    personas = {p["persona_id"]: p for p in _load_jsonl(PERSONAS_PATH)}
    items_meta = {
        it["item_id"]: it
        for it in _load_jsonl(DATA_DIR / "items.jsonl")
    }
    cell_a = {(r["persona_id"], r["item_id"]): r for r in _load_jsonl(CELL_A_SIM)}
    cell_c = {(r["persona_id"], r["item_id"]): r for r in _load_jsonl(CELL_C_SIM)}

    # Find all pairs where both conditions have a prediction
    candidates = [k for k in cell_a if k in cell_c]
    rng.shuffle(candidates)

    # Stratified pick: max 2 per persona, balance across ethnic groups
    by_persona: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_group_count: dict[str, int] = defaultdict(int)
    selected: list[tuple[str, str]] = []
    for pid, iid in candidates:
        if len(selected) >= N_EXAMPLES:
            break
        if len(by_persona[pid]) >= MAX_PER_PERSONA:
            continue
        # Light ethnic balance: cap each group at ceil(N/3) per non-Other
        group = personas[pid].get("ethnic_hint", "Other")
        if group != "Other" and by_group_count[group] >= 4:
            continue
        by_persona[pid].append((pid, iid))
        by_group_count[group] += 1
        selected.append((pid, iid))

    # Build the example records
    examples: list[dict] = []
    for pid, iid in selected:
        persona = personas[pid]
        item = items_meta.get(iid, {"item_id": iid, "title": "(metadata missing)"})
        a = cell_a[(pid, iid)]
        c = cell_c[(pid, iid)]
        examples.append({
            "persona_id":      pid,
            "ethnic_hint":     persona.get("ethnic_hint", "?"),
            "religious_hint":  persona.get("religious_hint", "?"),
            "naija_name":      persona.get("naija_name", "?"),
            "item_id":         iid,
            "item_title":      item.get("title", "")[:120],
            "item_brand":      item.get("brand", ""),
            "overlay_off":     {
                "predicted_rating": a.get("predicted_rating"),
                "predicted_review": a.get("predicted_review", ""),
            },
            "cultural_on":     {
                "predicted_rating": c.get("predicted_rating"),
                "predicted_review": c.get("predicted_review", ""),
            },
        })

    OUTPUT_JSON.write_text(json.dumps({
        "selection_methodology": (
            "Stratified random sample (seed=42): all (persona, item) pairs "
            "for which both cell-A and cell-C predictions exist; at most "
            f"{MAX_PER_PERSONA} per persona; light ethnic-group balance."
        ),
        "n_examples": len(examples),
        "examples": examples,
    }, indent=2))

    # Markdown preview for human reading
    lines = [
        "# Qualitative Examples — overlay-off vs cultural-on",
        "",
        "*Selection methodology: stratified random (seed=42), all (persona, item) "
        f"pairs with both predictions available, max {MAX_PER_PERSONA}/persona.*",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"## {i}. {ex['naija_name']} ({ex['ethnic_hint']}, {ex['religious_hint']})")
        lines.append(f"**Item:** {ex['item_title']}  ·  brand: {ex['item_brand'] or '—'}")
        lines.append("")
        lines.append(f"**Overlay-off (rating {ex['overlay_off']['predicted_rating']}):**")
        lines.append(f"> {ex['overlay_off']['predicted_review']}")
        lines.append("")
        lines.append(f"**Cultural-on (rating {ex['cultural_on']['predicted_rating']}):**")
        lines.append(f"> {ex['cultural_on']['predicted_review']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines))

    print(f"  selected {len(examples)} examples")
    print(f"  JSON:     {OUTPUT_JSON}")
    print(f"  Markdown: {OUTPUT_MD}")
    print(f"\n  open {OUTPUT_MD} to read them side by side.")


if __name__ == "__main__":
    main()
