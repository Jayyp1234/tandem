"""Cold-start baselines (P5-zero, Chat-Rec, Popularity) for the comparative
cold-start narrative.

Mirrors src/eval/cold_start.py but runs the two LLM baselines (and a non-LLM
popularity baseline) on the *same* truncated-history personas, so we can
answer the question the cold_start.py result alone cannot:

  In the cold-start regime, does TANDEM beat the LLM baselines, or do all
  three LLM-based approaches benefit equivalently from short prompts?

Outputs:
  results/cold_start_baseline_p5_zero.jsonl
  results/cold_start_baseline_p5_zero_ranking.jsonl
  results/cold_start_baseline_chat_rec.jsonl
  results/cold_start_baseline_chat_rec_ranking.jsonl
  results/cold_start_baseline_popularity_ranking.jsonl
  results/cold_start_baselines_summary.json

Compute: ~4,000 LLM calls (20 personas x 100 candidates x 2 LLM baselines).
At Groq free-tier rates with 15-key rotation, ~1-2 h wall-clock.

Run via:  python -m src.eval.cold_start_baselines
         OR  python -m src.eval.cold_start_baselines --n-history 1 --n-personas 20
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.agents.recommender import hit_at_k, mrr, ndcg_at_k
from src.baselines import run_baseline, _rank_baseline_outputs
from src.llm.client import GroqClient


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _agg(records: list[dict], metric: str) -> float:
    return float(np.mean([r[metric] for r in records])) if records else 0.0


def _ci(records: list[dict], metric: str, rng_seed: int = 42, n_boot: int = 500) -> tuple[float, float]:
    """Cluster-bootstrap CI on persona; clusters = personas."""
    if not records:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    by_persona: dict[str, list[float]] = {}
    for r in records:
        by_persona.setdefault(r["persona_id"], []).append(r[metric])
    persona_means = {p: float(np.mean(v)) for p, v in by_persona.items()}
    personas = list(persona_means.keys())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(personas, size=len(personas), replace=True)
        boot[i] = np.mean([persona_means[p] for p in sampled])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def compute_popularity_ranking(
    cold_personas: list[dict], users_jsonl: Path, output_path: Path,
) -> None:
    """Non-LLM reference baseline: rank candidates by global training-set frequency.

    No model calls. Just counts how often each item appears in users.jsonl's
    history field and sorts each persona's candidate list by that count
    (descending). Ties broken by item_id lexicographic order for determinism.
    """
    item_counts: Counter[str] = Counter()
    with open(users_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u = json.loads(line)
            for h in u.get("history", []):
                iid = h.get("item_id") if isinstance(h, dict) else h
                if iid:
                    item_counts[iid] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for persona in cold_personas:
            target = persona["target_item_id"]
            cands = list(persona["candidate_item_ids"])
            ranked_ids = sorted(cands, key=lambda iid: (-item_counts.get(iid, 0), iid))
            f.write(json.dumps({
                "persona_id": persona["persona_id"],
                "baseline":   "popularity",
                "target_item_id": target,
                "ndcg_10":  ndcg_at_k(ranked_ids, target, k=10),
                "hit_10":   hit_at_k(ranked_ids, target, k=10),
                "hit_5":    hit_at_k(ranked_ids, target, k=5),
                "mrr":      mrr(ranked_ids, target),
                "top_k_items":   ranked_ids[:10],
                "top_k_counts":  [int(item_counts.get(iid, 0)) for iid in ranked_ids[:10]],
            }, ensure_ascii=True) + "\n")


def _format_row(name: str, recs: list[dict]) -> str:
    if not recs:
        return f"    {name:<32} (no data)"
    n_lo, n_hi = _ci(recs, "ndcg_10")
    h10_lo, h10_hi = _ci(recs, "hit_10")
    h5_lo, h5_hi = _ci(recs, "hit_5")
    m_lo, m_hi = _ci(recs, "mrr")
    return (
        f"    {name:<32} "
        f"{_agg(recs, 'ndcg_10'):.3f} [{n_lo:.3f},{n_hi:.3f}]  "
        f"{_agg(recs, 'hit_10'):.3f} [{h10_lo:.3f},{h10_hi:.3f}]  "
        f"{_agg(recs, 'hit_5'):.3f} [{h5_lo:.3f},{h5_hi:.3f}]  "
        f"{_agg(recs, 'mrr'):.3f} [{m_lo:.3f},{m_hi:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-history", type=int, default=1)
    parser.add_argument("--n-personas", type=int, default=20)
    args = parser.parse_args()

    base = _load_jsonl(Path("data/personas_20.jsonl"))[: args.n_personas]

    # Build cold-start personas (mirror cold_start.py).
    cold_personas = []
    for p in base:
        cold = copy.deepcopy(p)
        cold["persona_id"] = f"{p['persona_id']}-cold{args.n_history}"
        cold["history_window"] = p["history_window"][-args.n_history:]
        cold_personas.append(cold)

    items_meta = {
        it["item_id"]: it
        for it in _load_jsonl(Path("data/beauty_5core/items.jsonl"))
    }

    client = GroqClient(cache_path="cache/llm_responses.jsonl")

    # --- LLM baselines (P5-zero, Chat-Rec) on cold-start personas ---
    for baseline in ("p5_zero", "chat_rec"):
        sim_out = Path(f"results/cold_start_baseline_{baseline}.jsonl")
        rank_out = Path(f"results/cold_start_baseline_{baseline}_ranking.jsonl")
        print(f"\n=== Cold-start baseline: {baseline} "
              f"(n_history={args.n_history}, {len(cold_personas)} personas) ===")
        run_baseline(client, cold_personas, items_meta, baseline, sim_out)
        _rank_baseline_outputs(sim_out, cold_personas, rank_out, baseline)
        print(f"  ranked outputs -> {rank_out}")

    # --- Non-LLM popularity baseline ---
    pop_out = Path("results/cold_start_baseline_popularity_ranking.jsonl")
    print(f"\n=== Cold-start baseline: popularity (training-set frequency) ===")
    compute_popularity_ranking(
        cold_personas, Path("data/beauty_5core/users.jsonl"), pop_out,
    )
    print(f"  ranked outputs -> {pop_out}")

    # --- Comparison table ---
    tandem_cold = _load_jsonl(Path("results/cold_start_ranking.jsonl"))
    tandem_full = _load_jsonl(Path("results/cell_C_ranking.jsonl"))
    p5_full     = _load_jsonl(Path("results/baseline_p5_zero_ranking.jsonl"))
    chatrec_full = _load_jsonl(Path("results/baseline_chat_rec_ranking.jsonl"))
    p5_cold = _load_jsonl(Path("results/cold_start_baseline_p5_zero_ranking.jsonl"))
    chatrec_cold = _load_jsonl(Path("results/cold_start_baseline_chat_rec_ranking.jsonl"))
    pop_cold = _load_jsonl(pop_out)

    print("\n  Cold-start vs full-history comparison "
          "(decomposed cultural-on for TANDEM; same LLM, same protocol):")
    print(f"    {'method':<32} {'NDCG@10':>20}  {'Hit@10':>20}  {'Hit@5':>20}  {'MRR':>20}")
    print("    " + "-" * 110)
    print(_format_row("TANDEM full-history (10 items)", tandem_full))
    print(_format_row("TANDEM cold-start  (1 item)",  tandem_cold))
    print("    " + " " * 110)
    print(_format_row("P5-zero full-history",         p5_full))
    print(_format_row("P5-zero cold-start",           p5_cold))
    print("    " + " " * 110)
    print(_format_row("Chat-Rec full-history",        chatrec_full))
    print(_format_row("Chat-Rec cold-start",          chatrec_cold))
    print("    " + " " * 110)
    print(_format_row("Popularity cold-start",        pop_cold))

    # --- Persist summary ---
    def _summary(recs: list[dict]) -> dict:
        if not recs:
            return {"n": 0}
        n_lo, n_hi = _ci(recs, "ndcg_10")
        h10_lo, h10_hi = _ci(recs, "hit_10")
        h5_lo, h5_hi = _ci(recs, "hit_5")
        return {
            "n":       len(recs),
            "ndcg_10": _agg(recs, "ndcg_10"),
            "ndcg_10_ci95": [n_lo, n_hi],
            "hit_10":  _agg(recs, "hit_10"),
            "hit_10_ci95":  [h10_lo, h10_hi],
            "hit_5":   _agg(recs, "hit_5"),
            "hit_5_ci95":   [h5_lo, h5_hi],
            "mrr":     _agg(recs, "mrr"),
        }

    summary = {
        "n_history":  args.n_history,
        "n_personas": len(cold_personas),
        "tandem_full":   _summary(tandem_full),
        "tandem_cold":   _summary(tandem_cold),
        "p5_zero_full":  _summary(p5_full),
        "p5_zero_cold":  _summary(p5_cold),
        "chat_rec_full": _summary(chatrec_full),
        "chat_rec_cold": _summary(chatrec_cold),
        "popularity_cold": _summary(pop_cold),
    }
    Path("results/cold_start_baselines_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n  summary -> results/cold_start_baselines_summary.json")


if __name__ == "__main__":
    main()
