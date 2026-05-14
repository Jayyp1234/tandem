"""Cold-start subset experiment for Task B's 25-point rubric line.

Re-runs TANDEM (decomposed, cultural-on) on each persona's candidate set, but
truncates the persona's history to a single most-recent interaction instead of
the usual 10-item window. The simulator therefore has almost no signal about
the user --- the LLM has to do zero-shot reasoning over (name, ethnic context,
overlay, one prior purchase). The pitch: TANDEM degrades gracefully on
cold-start because the LLM reasons over the persona text, not collaborative
filtering matrices.

Outputs: results/cold_start_simulator.jsonl and results/cold_start_ranking.jsonl
Also prints a side-by-side comparison with full-history cell C.

Run via:  python -m src.eval.cold_start
         OR  python -m src.eval.cold_start --n-history 2 --n-personas 10
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.agents import recommender, simulator
from src.llm.client import GroqClient


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _agg(records: list[dict], metric: str) -> float:
    return float(np.mean([r[metric] for r in records])) if records else 0.0


def _ci(records: list[dict], metric: str, rng_seed: int = 42, n_boot: int = 500) -> tuple[float, float]:
    """Cluster-bootstrap CI on persona; here clusters = personas."""
    if not records:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    by_persona: dict[str, list[float]] = {}
    for r in records:
        by_persona.setdefault(r["persona_id"], []).append(r[metric])
    persona_means = {p: np.mean(v) for p, v in by_persona.items()}
    personas = list(persona_means.keys())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(personas, size=len(personas), replace=True)
        boot[i] = np.mean([persona_means[p] for p in sampled])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-history", type=int, default=1,
        help="History items to retain per persona for the cold-start condition (default 1).",
    )
    parser.add_argument(
        "--n-personas", type=int, default=20,
        help="Number of personas to evaluate (default 20).",
    )
    args = parser.parse_args()

    base = _load_jsonl(Path("data/personas_20.jsonl"))[: args.n_personas]

    # Build cold-start personas by truncating each persona's history window.
    # We append a suffix to persona_id so the simulator's deterministic seed
    # diverges from the full-history version, producing fresh cache entries.
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
    sim_out = Path("results/cold_start_simulator.jsonl")
    rank_out = Path("results/cold_start_ranking.jsonl")

    print(
        f"\n=== Cold-Start: decomposed × cultural-on, "
        f"n_history={args.n_history}, {len(cold_personas)} personas ==="
    )
    simulator.run_cell(
        client=client,
        personas=cold_personas,
        items_meta=items_meta,
        condition="cultural-on",
        architecture="decomposed",
        output_path=sim_out,
    )
    recommender.rank_and_score(
        simulator_output_path=sim_out,
        personas=cold_personas,
        output_path=rank_out,
    )
    print(f"\n  cold-start done — wrote {sim_out} and {rank_out}")

    # Compare to full-history cell C
    cell_c = _load_jsonl(Path("results/cell_C_ranking.jsonl"))
    cold = _load_jsonl(rank_out)

    print("\n  Cold-start vs full-history (decomposed × cultural-on):")
    print(f"    {'condition':<24} {'NDCG@10':>14}  {'Hit@10':>14}  {'Hit@5':>14}  {'MRR':>14}")
    for name, recs in (("full history (10 items)", cell_c),
                       (f"cold-start ({args.n_history} item{'s' if args.n_history != 1 else ''})", cold)):
        if not recs:
            print(f"    {name:<24} (no data)")
            continue
        n_lo_n, n_hi_n = _ci(recs, "ndcg_10")
        h_lo, h_hi = _ci(recs, "hit_10")
        h5_lo, h5_hi = _ci(recs, "hit_5")
        m_lo, m_hi = _ci(recs, "mrr")
        print(
            f"    {name:<24} "
            f"{_agg(recs, 'ndcg_10'):.3f} [{n_lo_n:.3f},{n_hi_n:.3f}]  "
            f"{_agg(recs, 'hit_10'):.3f} [{h_lo:.3f},{h_hi:.3f}]  "
            f"{_agg(recs, 'hit_5'):.3f} [{h5_lo:.3f},{h5_hi:.3f}]  "
            f"{_agg(recs, 'mrr'):.3f} [{m_lo:.3f},{m_hi:.3f}]"
        )

    summary = {
        "n_history":      args.n_history,
        "n_personas":     len(cold_personas),
        "full_history": {
            "ndcg_10": _agg(cell_c, "ndcg_10"),
            "hit_10":  _agg(cell_c, "hit_10"),
            "hit_5":   _agg(cell_c, "hit_5"),
            "mrr":     _agg(cell_c, "mrr"),
        },
        "cold_start": {
            "ndcg_10": _agg(cold, "ndcg_10"),
            "hit_10":  _agg(cold, "hit_10"),
            "hit_5":   _agg(cold, "hit_5"),
            "mrr":     _agg(cold, "mrr"),
        },
    }
    Path("results/cold_start_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n  summary -> results/cold_start_summary.json")


if __name__ == "__main__":
    main()
