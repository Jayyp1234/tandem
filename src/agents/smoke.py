"""Tiny end-to-end smoke test of the TANDEM simulator.

Calls the simulator on 1 persona × 3 items × cultural-on × decomposed = 3 Groq
requests (negligible quota cost). Verifies the pipeline is wired correctly
before committing to the full ~28 000-call experiment matrix.

Run via:  make smoke   OR   python -m src.agents.smoke
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.simulator import predict
from src.llm.client import GroqClient


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    personas = _load_jsonl(Path("data/personas_20.jsonl"))
    items_list = _load_jsonl(Path("data/beauty_5core/items.jsonl"))
    items_meta = {it["item_id"]: it for it in items_list}

    persona = personas[0]
    item_ids = persona["candidate_item_ids"][:3]
    items = [items_meta[i] for i in item_ids if i in items_meta]

    print(f"\nPersona: {persona['persona_id']}  ·  Naija name: {persona['naija_name']}")
    print(f"  ethnic_hint: {persona['ethnic_hint']}, religious: {persona['religious_hint']}")
    print(f"  history (last 3): {[h['item_id'] for h in persona['history_window'][-3:]]}")
    print(f"  testing on {len(items)} items, cultural-on × decomposed\n")

    client = GroqClient(cache_path="cache/llm_responses.jsonl")
    for i, item in enumerate(items, 1):
        rec = predict(
            client=client,
            persona=persona,
            item=item,
            condition="cultural-on",
            architecture="decomposed",
        )
        print(f"--- Item {i}: {item.get('title', '(no title)')[:70]}")
        print(f"    rating:  {rec['predicted_rating']}")
        print(f"    cached:  {rec['cached']}   latency: {rec['latency_ms']} ms")
        print(f"    review:  {rec['predicted_review'][:280]}")
        print()

    print("Smoke OK. Reviews should sound like Nigerian English / Pidgin; "
          "ratings should be 1-5 integers.")
    print("If both look right, you're cleared to run `make experiments`.")


if __name__ == "__main__":
    main()
