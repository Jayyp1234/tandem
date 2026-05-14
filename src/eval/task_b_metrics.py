"""Task B metrics: aggregate NDCG@10, Hit@10, Hit@5, MRR per condition × architecture.

Reads `results/cell_*_ranking.jsonl` (produced by run_all_cells.py) and the two
baseline ranking files, and reports the mean and cluster-bootstrap CI (on
persona) for each top-k metric. Hits the Task B rubric's "Ranking Quality
(NDCG@10 / Hit Rate)" line.

Run via:  python -m src.eval.task_b_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
OUTPUT = RESULTS_DIR / "task_b_metrics.json"
RNG_SEED = 42
N_BOOT = 1000


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray,
    n_boot: int = N_BOOT, ci: float = 0.95, rng_seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(rng_seed)
    unique = np.unique(clusters)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idxs = np.concatenate([np.where(clusters == c)[0] for c in sampled])
        boot[i] = values[idxs].mean()
    a = (1 - ci) / 2
    return float(values.mean()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def _summarise(name: str, recs: list[dict]) -> dict:
    if not recs:
        return {"name": name, "n": 0, "note": "no rankings"}

    personas = np.array([r["persona_id"] for r in recs])

    out: dict = {"name": name, "n": len(recs)}
    for metric in ("ndcg_10", "hit_10", "hit_5", "mrr"):
        if metric not in recs[0]:
            continue
        vals = np.array([r[metric] for r in recs], dtype=float)
        mean, lo, hi = _cluster_bootstrap_ci(vals, personas)
        out[metric] = {"mean": mean, "ci": [lo, hi]}
    return out


def main() -> None:
    cells = ["A", "B", "C", "D", "E"]
    summary: dict[str, dict] = {}

    for cell in cells:
        path = RESULTS_DIR / f"cell_{cell}_ranking.jsonl"
        summary[f"cell_{cell}"] = _summarise(f"cell_{cell}", _load_jsonl(path))

    for baseline in ("p5_zero", "chat_rec"):
        path = RESULTS_DIR / f"baseline_{baseline}_ranking.jsonl"
        summary[f"baseline_{baseline}"] = _summarise(f"baseline_{baseline}", _load_jsonl(path))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print("\n  Task B ranking metrics (mean [95% cluster-bootstrap CI on persona]):")
    header = f"    {'system':<22} {'n':>3}  {'NDCG@10':>20}  {'Hit@10':>20}  {'Hit@5':>20}  {'MRR':>20}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    def _fmt(d: dict, key: str) -> str:
        if key not in d:
            return f"{'—':>20}"
        m = d[key]["mean"]
        lo, hi = d[key]["ci"]
        return f"{m:.4f} [{lo:.3f}, {hi:.3f}]"

    for k, v in summary.items():
        if v.get("n", 0) == 0:
            print(f"    {k:<22} {v['n']:>3}  (no data)")
            continue
        print(
            f"    {k:<22} {v['n']:>3}  "
            f"{_fmt(v, 'ndcg_10'):>20}  "
            f"{_fmt(v, 'hit_10'):>20}  "
            f"{_fmt(v, 'hit_5'):>20}  "
            f"{_fmt(v, 'mrr'):>20}"
        )
    print(f"\n  written to {OUTPUT}")


if __name__ == "__main__":
    main()
