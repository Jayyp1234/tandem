"""TANDEM recommender agent.

Per experiment_matrix.md §3: aggregates the simulator's per-(persona, item)
predicted ratings into a top-k ranking for each persona. Argsort by predicted
rating, ties broken by predicted-review length as a weak tie-breaker.

Computes the standard sequential-rec metrics: NDCG@10, Hit@10, MRR, against
each persona's held-out target_item_id (the SASRec / BERT4Rec / P5 protocol).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rank_and_score(
    simulator_output_path: Path,
    personas: list[dict],
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Read simulator output, rank candidates per (persona, condition, architecture),
    score against each persona's target. Write JSONL with NDCG@k, Hit@k, MRR.
    """
    persona_targets = {p["persona_id"]: p["target_item_id"] for p in personas}

    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for line in simulator_output_path.read_text().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["persona_id"], rec["condition"], rec["architecture"])
        by_key[key].append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for (pid, cond, arch), recs in by_key.items():
            target = persona_targets.get(pid)
            if target is None:
                continue
            # Sort by predicted_rating descending (tie-break: review length)
            ranked = sorted(
                recs,
                key=lambda r: (-r["predicted_rating"], -len(r.get("predicted_review", ""))),
            )
            ranked_ids = [r["item_id"] for r in ranked]
            row = {
                "persona_id": pid,
                "condition": cond,
                "architecture": arch,
                "target_item_id": target,
                "ndcg_10":  ndcg_at_k(ranked_ids, target, k=top_k),
                "hit_10":   hit_at_k(ranked_ids, target, k=top_k),
                "hit_5":    hit_at_k(ranked_ids, target, k=5),
                "mrr":      mrr(ranked_ids, target),
                "top_k_items":   ranked_ids[:top_k],
                "top_k_ratings": [r["predicted_rating"] for r in ranked[:top_k]],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ndcg_at_k(predicted: list[str], target: str, k: int = 10) -> float:
    """Binary-relevance NDCG@k for a single positive."""
    top_k = predicted[:k]
    if target not in top_k:
        return 0.0
    return float(1.0 / np.log2(top_k.index(target) + 2))


def hit_at_k(predicted: list[str], target: str, k: int = 10) -> int:
    return int(target in predicted[:k])


def mrr(predicted: list[str], target: str) -> float:
    if target not in predicted:
        return 0.0
    return float(1.0 / (predicted.index(target) + 1))
