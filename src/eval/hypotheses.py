"""Hypothesis tests H1–H7 per protocol_spec.md §4.

Each test takes the simulator-output JSONL files for the relevant cells, computes
an effect size with 95% (cluster-)bootstrap CI on persona, and returns a verdict.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from src.eval.classifier import naija_scores
from src.overlay.linguistic import naija_token_count
from src.overlay.preferences import cultural_topic_coverage

RNG_SEED = 42
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cell(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def _by_pair(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Index by (persona_id, item_id) for paired comparisons."""
    return {(r["persona_id"], r["item_id"]): r for r in records}


def _cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray,
    n_boot: int = N_BOOT, ci: float = 0.95, rng_seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """95% cluster-bootstrap CI on `values` clustered by `clusters` (e.g. persona)."""
    rng = np.random.default_rng(rng_seed)
    unique = np.unique(clusters)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idxs = np.concatenate([np.where(clusters == c)[0] for c in sampled])
        boot[i] = values[idxs].mean()
    a = (1 - ci) / 2
    return float(values.mean()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank on paired (a - b). Returns (statistic, p)."""
    diff = a - b
    diff = diff[diff != 0]  # zsigned-rank ignores zeros
    if len(diff) == 0:
        return 0.0, 1.0
    res = stats.wilcoxon(diff, alternative="greater")
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# H1 — floor: Naija n-gram density
# ---------------------------------------------------------------------------

def test_h1(cell_a: list[dict], cell_b: list[dict], cell_c: list[dict]) -> dict:
    pair_a, pair_b, pair_c = _by_pair(cell_a), _by_pair(cell_b), _by_pair(cell_c)
    keys = sorted(pair_a.keys() & pair_b.keys() & pair_c.keys())

    def density(rec: dict) -> float:
        text = rec.get("predicted_review", "") or ""
        n_tokens = max(1, len(text.split()))
        return naija_token_count(text) * 100.0 / n_tokens

    d_off  = np.array([density(pair_a[k]) for k in keys])
    d_noise = np.array([density(pair_b[k]) for k in keys])
    d_cult = np.array([density(pair_c[k]) for k in keys])
    clusters = np.array([k[0] for k in keys])

    diff_off = d_cult - d_off
    diff_noise = d_cult - d_noise
    mean_off, lo_off, hi_off = _cluster_bootstrap_ci(diff_off, clusters)
    _, p_off = _wilcoxon_paired(d_cult, d_off)
    _, p_noise = _wilcoxon_paired(d_cult, d_noise)

    threshold = 0.30
    passed = (mean_off >= threshold) and (lo_off > 0) and (p_noise < 0.05)
    return {
        "hypothesis": "H1",
        "class": "floor",
        "effect_size_vs_off": mean_off, "ci_off": [lo_off, hi_off],
        "p_vs_off": p_off, "p_vs_noise": p_noise,
        "threshold": threshold, "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# H2 — substantive: rating-distribution shift on skin-care
# ---------------------------------------------------------------------------

def test_h2(cell_a: list[dict], cell_b: list[dict], cell_c: list[dict],
            items_meta: dict[str, dict], category: str = "skin") -> dict:
    pair_a, pair_b, pair_c = _by_pair(cell_a), _by_pair(cell_b), _by_pair(cell_c)
    keys = [
        k for k in (pair_a.keys() & pair_b.keys() & pair_c.keys())
        if category.lower() in (items_meta.get(k[1], {}).get("category", "")).lower()
    ]
    if not keys:
        return {"hypothesis": "H2", "passed": False, "note": f"no items match category={category}"}
    r_off = np.array([pair_a[k]["predicted_rating"] for k in keys])
    r_noise = np.array([pair_b[k]["predicted_rating"] for k in keys])
    r_cult = np.array([pair_c[k]["predicted_rating"] for k in keys])
    _, p_off = _wilcoxon_paired(r_cult, r_off)
    _, p_noise = _wilcoxon_paired(r_cult, r_noise)
    passed = (p_off < 0.05) and (p_noise < 0.05)
    return {
        "hypothesis": "H2", "class": "substantive", "category": category,
        "n_items_in_category": len(keys),
        "p_vs_off": p_off, "p_vs_noise": p_noise, "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# H3 — substantive: regression sentiment ~ overlay × ingredient + length
# ---------------------------------------------------------------------------

def test_h3(cells: dict[str, list[dict]]) -> dict:
    """Light implementation: use a sentiment proxy (positive-word fraction)
    instead of pulling a transformer at hackathon scope. Test the interaction
    coefficient via OLS."""
    import statsmodels.formula.api as smf
    import pandas as pd

    pos_words = {"good", "love", "great", "best", "amazing", "perfect", "smooth", "clean"}
    cultural_ingredients = {"shea", "black soap", "dudu-osun", "palm kernel", "ori", "neem"}

    rows = []
    for cond, recs in cells.items():
        for r in recs:
            txt = (r.get("predicted_review", "") or "").lower()
            tokens = txt.split()
            if not tokens:
                continue
            pos_frac = sum(1 for t in tokens if t.strip(".,!?") in pos_words) / len(tokens)
            ingr = int(any(ing in txt for ing in cultural_ingredients))
            rows.append({
                "sentiment": pos_frac, "overlay_on": int(cond == "cultural-on"),
                "ingredient": ingr, "length": len(tokens), "persona_id": r["persona_id"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"hypothesis": "H3", "passed": False, "note": "no data"}

    model = smf.ols(
        "sentiment ~ overlay_on * ingredient + length", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["persona_id"]})
    interaction_term = "overlay_on:ingredient"
    beta = float(model.params.get(interaction_term, 0.0))
    pval = float(model.pvalues.get(interaction_term, 1.0))
    return {
        "hypothesis": "H3", "class": "substantive",
        "interaction_beta": beta, "interaction_p": pval,
        "passed": bool(beta > 0 and pval < 0.05),
    }


# ---------------------------------------------------------------------------
# H4 — floor: cultural-topic coverage rate
# ---------------------------------------------------------------------------

def test_h4(cell_b: list[dict], cell_c: list[dict]) -> dict:
    pair_b, pair_c = _by_pair(cell_b), _by_pair(cell_c)
    keys = sorted(pair_b.keys() & pair_c.keys())

    def covers(rec: dict) -> int:
        return int(cultural_topic_coverage(rec.get("predicted_review", "")) >= 1)

    cov_noise = np.array([covers(pair_b[k]) for k in keys])
    cov_cult = np.array([covers(pair_c[k]) for k in keys])
    clusters = np.array([k[0] for k in keys])

    rng = np.random.default_rng(RNG_SEED)
    boot_noise = np.empty(N_BOOT); boot_cult = np.empty(N_BOOT)
    unique = np.unique(clusters)
    for i in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idxs = np.concatenate([np.where(clusters == c)[0] for c in sampled])
        boot_noise[i] = cov_noise[idxs].mean()
        boot_cult[i]  = cov_cult[idxs].mean()
    ci_noise = (np.quantile(boot_noise, 0.025), np.quantile(boot_noise, 0.975))
    ci_cult  = (np.quantile(boot_cult,  0.025), np.quantile(boot_cult,  0.975))
    passed = ci_cult[0] > ci_noise[1]  # non-overlapping, cultural higher
    return {
        "hypothesis": "H4", "class": "floor",
        "cov_noise_mean": float(cov_noise.mean()), "ci_noise": [float(ci_noise[0]), float(ci_noise[1])],
        "cov_cult_mean":  float(cov_cult.mean()),  "ci_cult":  [float(ci_cult[0]),  float(ci_cult[1])],
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# H5 — substantive: persona consistency (within > between) on cultural-on
# ---------------------------------------------------------------------------

def test_h5(cell_c: list[dict]) -> dict:
    """Within-persona TF-IDF style similarity vs between-persona baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    by_persona: dict[str, list[str]] = defaultdict(list)
    for r in cell_c:
        by_persona[r["persona_id"]].append(r.get("predicted_review", "") or "")

    personas = sorted(by_persona.keys())
    if len(personas) < 2:
        return {"hypothesis": "H5", "passed": False, "note": "need >=2 personas"}

    all_docs = [d for p in personas for d in by_persona[p]]
    persona_of_doc = [p for p in personas for _ in by_persona[p]]
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=10000)
    X = vec.fit_transform(all_docs)
    sims = cosine_similarity(X)

    within, between = [], []
    n = len(all_docs)
    for i in range(n):
        for j in range(i + 1, n):
            (within if persona_of_doc[i] == persona_of_doc[j] else between).append(sims[i, j])
    if not within or not between:
        return {"hypothesis": "H5", "passed": False, "note": "insufficient pairs"}

    res = stats.mannwhitneyu(within, between, alternative="greater")
    return {
        "hypothesis": "H5", "class": "substantive",
        "within_mean": float(np.mean(within)), "between_mean": float(np.mean(between)),
        "u_stat": float(res.statistic), "p_value": float(res.pvalue),
        "passed": bool(res.pvalue < 0.05 and np.mean(within) > np.mean(between)),
    }


# ---------------------------------------------------------------------------
# H6 — substantive: Naija classifier authenticity score
# ---------------------------------------------------------------------------

def test_h6(cell_b: list[dict], cell_c: list[dict]) -> dict:
    pair_b, pair_c = _by_pair(cell_b), _by_pair(cell_c)
    keys = sorted(pair_b.keys() & pair_c.keys())
    texts_noise = [pair_b[k].get("predicted_review", "") or "" for k in keys]
    texts_cult  = [pair_c[k].get("predicted_review", "") or "" for k in keys]
    s_noise = naija_scores(texts_noise)
    s_cult  = naija_scores(texts_cult)
    _, p = _wilcoxon_paired(s_cult, s_noise)
    clusters = np.array([k[0] for k in keys])
    diff = s_cult - s_noise
    mean, lo, hi = _cluster_bootstrap_ci(diff, clusters)
    return {
        "hypothesis": "H6", "class": "substantive",
        "delta_mean": float(mean), "ci": [lo, hi], "p_vs_noise": float(p),
        "passed": bool(p < 0.05 and lo > 0),
    }


# ---------------------------------------------------------------------------
# H7 — C1 falsifier: decomposed > monolithic on H6 effect size
# ---------------------------------------------------------------------------

def test_h7(
    cell_b_decomposed: list[dict], cell_c_decomposed: list[dict],
    cell_d_monolithic: list[dict], cell_e_monolithic: list[dict],
) -> dict:
    """Compare H6 effect (cultural - noise) under decomposed vs monolithic.

    Persona-level effect sizes; paired-bootstrap on personas.
    """
    def per_persona_effect(noise: list[dict], cult: list[dict]) -> dict[str, float]:
        pn, pc = _by_pair(noise), _by_pair(cult)
        keys = sorted(pn.keys() & pc.keys())
        if not keys:
            return {}
        texts_n = [pn[k].get("predicted_review", "") or "" for k in keys]
        texts_c = [pc[k].get("predicted_review", "") or "" for k in keys]
        sn = naija_scores(texts_n); sc = naija_scores(texts_c)
        per_p: dict[str, list[float]] = defaultdict(list)
        for k, dv in zip(keys, sc - sn, strict=True):
            per_p[k[0]].append(float(dv))
        return {p: float(np.mean(v)) for p, v in per_p.items()}

    eff_dec = per_persona_effect(cell_b_decomposed, cell_c_decomposed)
    eff_mon = per_persona_effect(cell_d_monolithic, cell_e_monolithic)
    common = sorted(eff_dec.keys() & eff_mon.keys())
    if not common:
        return {"hypothesis": "H7", "passed": False, "note": "no shared personas"}
    a = np.array([eff_dec[p] for p in common])
    b = np.array([eff_mon[p] for p in common])
    diff = a - b

    rng = np.random.default_rng(RNG_SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[i] = diff[idx].mean()
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    return {
        "hypothesis": "H7", "class": "C1-falsifier",
        "decomposed_effect_mean": float(a.mean()),
        "monolithic_effect_mean": float(b.mean()),
        "delta_mean": float(diff.mean()), "ci": [lo, hi],
        "passed": bool(lo > 0),  # decomposed > monolithic strictly
        "note": "if passed=False, the architectural claim is empirically falsified — report honestly",
    }
