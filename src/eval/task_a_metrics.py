"""Task A metrics: rating RMSE/MAE + review-text quality on held-out targets.

For each persona, compare the simulator's prediction for their TARGET item
(the user's chronologically last, held-out interaction) against the user's
real review text and rating. These hit the Task A rubric's "Review Text
Quality (ROUGE / BERTScore)" and "Rating Accuracy (RMSE)" lines directly.

We use:
  - RMSE / MAE on rating
  - ROUGE-1 and ROUGE-L F1 (manual LCS implementation, no extra deps)
  - Sentence-transformer cosine similarity as a BERTScore-style semantic match

Run via:  python -m src.eval.task_a_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/beauty_5core")
RESULTS_DIR = Path("results")
PERSONAS_PATH = Path("data/personas_20.jsonl")
CELL_C_SIM = RESULTS_DIR / "cell_C_simulator.jsonl"      # cultural-on, decomposed
CELL_A_SIM = RESULTS_DIR / "cell_A_simulator.jsonl"      # overlay-off, decomposed
OUTPUT = RESULTS_DIR / "task_a_metrics.json"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rouge_1(reference: str, candidate: str) -> float:
    """Unigram F1 (set-based)."""
    ref = set(reference.lower().split())
    cand = set(candidate.lower().split())
    if not ref or not cand:
        return 0.0
    overlap = len(ref & cand)
    if overlap == 0:
        return 0.0
    p = overlap / len(cand)
    r = overlap / len(ref)
    return 2 * p * r / (p + r)


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1: F1 of longest common subsequence over tokens."""
    ref = reference.lower().split()
    cand = candidate.lower().split()
    if not ref or not cand:
        return 0.0
    m, n = len(ref), len(cand)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref[i] == cand[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / len(cand)
    r = lcs / len(ref)
    return 2 * p * r / (p + r)


def _aggregate(records: list[dict], *, condition: str) -> dict:
    """Compute mean RMSE / MAE / ROUGE / similarity / BERTScore for one condition."""
    if not records:
        return {"condition": condition, "n": 0, "note": "no matching predictions"}

    errors = np.array([r["rating_error"] for r in records])
    rouge_1s = np.array([r["rouge_1"] for r in records])
    rouge_ls = np.array([r["rouge_l"] for r in records])
    sims = np.array([r["semantic_similarity"] for r in records])

    out = {
        "condition":               condition,
        "n":                       len(records),
        "rating_RMSE":             float(np.sqrt(np.mean(errors ** 2))),
        "rating_MAE":              float(np.mean(np.abs(errors))),
        "rouge_1_mean":            float(np.mean(rouge_1s)),
        "rouge_1_std":             float(np.std(rouge_1s)),
        "rouge_l_mean":            float(np.mean(rouge_ls)),
        "rouge_l_std":             float(np.std(rouge_ls)),
        "semantic_similarity_mean": float(np.mean(sims)),
        "semantic_similarity_std":  float(np.std(sims)),
    }
    if records and "bertscore_f1" in records[0]:
        f1s = np.array([r["bertscore_f1"] for r in records])
        out["bertscore_f1_mean"] = float(np.mean(f1s))
        out["bertscore_f1_std"]  = float(np.std(f1s))
    return out


def _compute_bertscore(reference_texts: list[str], candidate_texts: list[str]) -> list[float] | None:
    """Compute BERTScore F1 for paired reference/candidate texts.

    Returns None gracefully if `bert-score` is not installed. The model is
    bert-base-uncased (~440 MB on first download) for size vs accuracy balance.
    """
    try:
        from bert_score import score  # type: ignore[import-not-found]
    except ImportError:
        print("  bert-score not installed (pip install bert-score); skipping BERTScore.")
        return None

    print("  computing BERTScore-F1 (bert-base-uncased) on", len(reference_texts), "pairs...")
    _, _, f1 = score(
        candidate_texts, reference_texts,
        lang="en", model_type="bert-base-uncased", verbose=False,
    )
    return [float(x) for x in f1.tolist()]


def main() -> None:
    print("  loading data...")
    users = {u["user_id"]: u for u in _load_jsonl(DATA_DIR / "users.jsonl")}
    personas = _load_jsonl(PERSONAS_PATH)

    # Index simulator outputs by (persona_id, item_id) for both conditions
    indices: dict[str, dict[tuple[str, str], dict]] = {}
    for name, path in (("overlay-off", CELL_A_SIM), ("cultural-on", CELL_C_SIM)):
        if not path.exists():
            print(f"  WARN: {path} not found — skipping {name} condition.")
            continue
        recs = _load_jsonl(path)
        indices[name] = {(r["persona_id"], r["item_id"]): r for r in recs}

    if not indices:
        print("  no simulator outputs found. Run `make experiments` first.")
        return

    print("  loading sentence-transformers model (first call downloads ~90MB)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    per_condition_records: dict[str, list[dict]] = {k: [] for k in indices}

    for persona in personas:
        user = users.get(persona["user_id"])
        if not user:
            continue
        target_id = persona["target_item_id"]
        target = next(
            (h for h in user["history"] if h["item_id"] == target_id),
            None,
        )
        if not target:
            continue

        actual_rating = float(target.get("rating", 0.0))
        actual_review = (
            target.get("review_text", "")
            or target.get("summary", "")
            or ""
        ).strip()
        if not actual_review:
            continue

        for cond, idx in indices.items():
            pred = idx.get((persona["persona_id"], target_id))
            if not pred:
                continue
            predicted_rating = float(pred["predicted_rating"])
            predicted_review = (pred.get("predicted_review") or "").strip()

            emb = model.encode([actual_review, predicted_review], convert_to_numpy=True)
            sim = float(
                np.dot(emb[0], emb[1])
                / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
            )

            per_condition_records[cond].append({
                "persona_id":     persona["persona_id"],
                "target_item_id": target_id,
                "actual_rating":  actual_rating,
                "predicted_rating": predicted_rating,
                "rating_error":   predicted_rating - actual_rating,
                "actual_review_tokens":    len(actual_review.split()),
                "predicted_review_tokens": len(predicted_review.split()),
                "rouge_1":             rouge_1(actual_review, predicted_review),
                "rouge_l":             rouge_l(actual_review, predicted_review),
                "semantic_similarity": sim,
            })

    # Compute BERTScore-F1 if the optional dep is installed.
    for cond, recs in per_condition_records.items():
        if not recs:
            continue
        refs = []
        cands = []
        for r in recs:
            user = users.get(next(p for p in personas if p["persona_id"] == r["persona_id"])["user_id"])
            target = next(h for h in user["history"] if h["item_id"] == r["target_item_id"])
            refs.append((target.get("review_text") or target.get("summary") or "").strip())
            cands.append(r.get("predicted_review_text", "") or "")
        # We didn't store predicted_review_text earlier; reconstruct from cache instead
        # using the same `pred` records we just iterated. Update record loop below.
        # Simpler: re-fetch predicted texts from the matching cell C / cell A index
        del refs, cands

    # Re-do BERTScore using the original simulator outputs we already loaded.
    for cond, recs in per_condition_records.items():
        if not recs:
            continue
        idx = indices.get(cond, {})
        refs: list[str] = []
        cands: list[str] = []
        order: list[int] = []
        for i, r in enumerate(recs):
            pred = idx.get((r["persona_id"], r["target_item_id"]))
            if not pred:
                continue
            user = users.get(next(p for p in personas if p["persona_id"] == r["persona_id"])["user_id"])
            target = next(h for h in user["history"] if h["item_id"] == r["target_item_id"])
            refs.append((target.get("review_text") or target.get("summary") or "").strip())
            cands.append((pred.get("predicted_review") or "").strip())
            order.append(i)

        f1s = _compute_bertscore(refs, cands) if refs else None
        if f1s is not None:
            for i, f1 in zip(order, f1s, strict=True):
                recs[i]["bertscore_f1"] = f1

    summary = {
        cond: _aggregate(recs, condition=cond)
        for cond, recs in per_condition_records.items()
    }
    summary["per_record"] = per_condition_records

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print("\n  Task A metrics on held-out target items:")
    print(f"    {'condition':<14} {'n':>3}  {'RMSE':>6}  {'MAE':>6}  {'R-1':>6}  {'R-L':>6}  {'sim':>6}  {'BS-F1':>6}")
    for cond, agg in summary.items():
        if cond == "per_record":
            continue
        if agg.get("n", 0) == 0:
            print(f"    {cond:<14} {agg['n']:>3}  (no data)")
            continue
        bs = agg.get("bertscore_f1_mean")
        bs_str = f"{bs:>6.3f}" if bs is not None else "  ----"
        print(
            f"    {cond:<14} {agg['n']:>3}  "
            f"{agg['rating_RMSE']:>6.3f}  "
            f"{agg['rating_MAE']:>6.3f}  "
            f"{agg['rouge_1_mean']:>6.3f}  "
            f"{agg['rouge_l_mean']:>6.3f}  "
            f"{agg['semantic_similarity_mean']:>6.3f}  "
            f"{bs_str}"
        )
    print(f"\n  written to {OUTPUT}")


if __name__ == "__main__":
    main()
