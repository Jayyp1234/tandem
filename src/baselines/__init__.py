"""Baselines: P5-zero-shot and Chat-Rec.

Both call the same Groq LLaMA-3.1-8B endpoint as TANDEM, on the same persona
candidate sets, for matched-protocol comparison. EXP3RT is NOT included as a
baseline (per phase3_findings.md §F1: requires fine-tuning we do not reproduce);
it is cited in related work only.

Run via:  python -m src.baselines
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from src.llm.cache import deterministic_seed
from src.llm.client import MODEL_FAST, GroqClient

Baseline = Literal["p5_zero", "chat_rec"]

_P5_ZERO_PROMPT = """You are a recommendation model in a beauty-products store. Given a user's recent purchase history, predict an integer rating (1-5) the user would give the candidate item.

User history (oldest first):
{history}

Candidate item: {item_title}

Reply with a single integer between 1 and 5. No commentary.
"""

_CHAT_REC_PROMPT = """You are an interactive recommendation assistant. Below is a user's beauty-product purchase history. Predict the user's rating (1-5) for the candidate item, then provide a one-sentence explanation.

User history:
{history}

Candidate item: {item_title} (brand: {item_brand})

Reply in the format:
RATING: <integer 1-5>
EXPLANATION: <one sentence>
"""


def _format_history(history: list[dict]) -> str:
    return "\n".join(
        f"  - rated {h.get('rating', '?')}/5: {(h.get('summary') or h.get('review_text', ''))[:60]}"
        for h in history
    )


def _parse_rating(text: str) -> float:
    m = re.search(r"\b([1-5])\b", text)
    return float(m.group(1)) if m else 3.0


def predict_baseline(client: GroqClient, persona: dict, item: dict, baseline: Baseline) -> dict:
    history_str = _format_history(persona["history_window"])
    if baseline == "p5_zero":
        prompt = _P5_ZERO_PROMPT.format(
            history=history_str, item_title=item.get("title", ""),
        )
    else:
        prompt = _CHAT_REC_PROMPT.format(
            history=history_str,
            item_title=item.get("title", ""),
            item_brand=item.get("brand", ""),
        )
    seed = deterministic_seed(persona["persona_id"], item["item_id"], baseline)
    response = client.cached_complete(
        prompt=prompt, seed=seed, model=MODEL_FAST,
        temperature=0.0, max_tokens=120,
    )
    return {
        "persona_id": persona["persona_id"],
        "item_id":    item["item_id"],
        "baseline":   baseline,
        "predicted_rating": _parse_rating(response.text),
        "raw_text":   response.text[:300],
        "model":      response.model,
        "cached":     response.cached,
    }


def run_baseline(
    client: GroqClient, personas: list[dict], items_meta: dict[str, dict],
    baseline: Baseline, output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = sum(len(p["candidate_item_ids"]) for p in personas)
    pbar = tqdm(total=n_total, desc=baseline)
    with output_path.open("w", encoding="utf-8") as f:
        for persona in personas:
            for iid in persona["candidate_item_ids"]:
                item = items_meta.get(iid)
                if item is None:
                    pbar.update(1)
                    continue
                rec = predict_baseline(client, persona, item, baseline)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.update(1)
    pbar.close()


def _rank_baseline_outputs(
    sim_path: Path, personas: list[dict], rank_path: Path, baseline_name: str,
) -> None:
    """Rank per-persona by predicted_rating, score against held-out target."""
    from collections import defaultdict
    from src.agents.recommender import hit_at_k, mrr, ndcg_at_k

    persona_targets = {p["persona_id"]: p["target_item_id"] for p in personas}
    by_persona: dict[str, list[dict]] = defaultdict(list)
    with open(sim_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_persona[rec["persona_id"]].append(rec)

    rank_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rank_path, "w", encoding="utf-8") as f:
        for pid, recs in by_persona.items():
            target = persona_targets.get(pid)
            if target is None:
                continue
            ranked = sorted(recs, key=lambda r: -r["predicted_rating"])
            ranked_ids = [r["item_id"] for r in ranked]
            f.write(json.dumps({
                "persona_id": pid,
                "baseline":   baseline_name,
                "target_item_id": target,
                "ndcg_10":  ndcg_at_k(ranked_ids, target, k=10),
                "hit_10":   hit_at_k(ranked_ids, target, k=10),
                "hit_5":    hit_at_k(ranked_ids, target, k=5),
                "mrr":      mrr(ranked_ids, target),
                "top_k_items":   ranked_ids[:10],
                "top_k_ratings": [r["predicted_rating"] for r in ranked[:10]],
            }, ensure_ascii=True) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    personas = _load_jsonl(Path("data/personas_20.jsonl"))
    items_meta = {it["item_id"]: it for it in _load_jsonl(Path("data/beauty_5core/items.jsonl"))}
    client = GroqClient(cache_path="cache/llm_responses.jsonl")
    for baseline in ("p5_zero", "chat_rec"):
        sim_out = Path(f"results/baseline_{baseline}.jsonl")
        rank_out = Path(f"results/baseline_{baseline}_ranking.jsonl")
        print(f"\n=== Baseline: {baseline} ===")
        run_baseline(client, personas, items_meta, baseline, sim_out)
        _rank_baseline_outputs(sim_out, personas, rank_out, baseline)
        print(f"  ranked outputs -> {rank_out}")


if __name__ == "__main__":
    main()
