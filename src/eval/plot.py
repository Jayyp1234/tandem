"""Generate paper figures from results/.

Per research/figure_specs.md, produces:
  - figures/fig3_main_results.pdf       (top-k metrics: TANDEM vs baselines)
  - figures/fig4_architectural_ablation.pdf  (decomposed vs monolithic — H7 visualization)
  - figures/fig5_cultural_validity.pdf  (H1, H5, H6, H7 panel)
  - figures/table1_hypothesis_summary.tex  (LaTeX table for the paper)

Defensive: missing result files → empty/skipped panels with a warning, not a crash.

Run via:  make figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# plotstyle.py lives in figures/ at the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures"))
from plotstyle import OKABE, apply_style  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("figures")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARN: {path} not found — skipping")
        return []
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def _avg(rankings: list[dict], metric: str) -> float:
    if not rankings:
        return 0.0
    return float(np.mean([r.get(metric, 0.0) for r in rankings]))


# ---------------------------------------------------------------------------
# Figure 3 — Main results
# ---------------------------------------------------------------------------

def fig3_main_results() -> None:
    """TANDEM (Cell C) vs baselines on NDCG@10 / Hit@10 / MRR."""
    apply_style("wide")

    cell_c = _load_jsonl(RESULTS / "cell_C_ranking.jsonl")
    # baselines need a separate ranker pass; we approximate with Cell A for now
    # if baseline_*_ranking.jsonl files exist they take precedence
    p5_rank = _load_jsonl(RESULTS / "baseline_p5_zero_ranking.jsonl")
    cr_rank = _load_jsonl(RESULTS / "baseline_chat_rec_ranking.jsonl")

    methods = []
    # Published numbers — see literature_evidence.md / phase3 lit check
    methods.append(("SASRec\n(Kang & McAuley '18)", 0.3219, 0.4854, np.nan))
    methods.append(("BERT4Rec\n(replicability '22)", 0.156, 0.40, np.nan))
    if p5_rank:
        methods.append(("P5-zero", _avg(p5_rank, "ndcg_10"),
                        _avg(p5_rank, "hit_10"), _avg(p5_rank, "mrr")))
    if cr_rank:
        methods.append(("Chat-Rec", _avg(cr_rank, "ndcg_10"),
                        _avg(cr_rank, "hit_10"), _avg(cr_rank, "mrr")))
    if cell_c:
        methods.append(("TANDEM", _avg(cell_c, "ndcg_10"),
                        _avg(cell_c, "hit_10"), _avg(cell_c, "mrr")))

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    metric_names = ["NDCG@10", "Hit@10", "MRR"]
    for ax_idx, (ax, mname) in enumerate(zip(axes, metric_names)):
        names = [m[0] for m in methods]
        vals = [m[ax_idx + 1] for m in methods]
        # Highlight TANDEM in vermillion; baselines in sky
        colors = [
            OKABE["vermillion"] if "TANDEM" in n else OKABE["sky"]
            for n in names
        ]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(mname)
        ax.set_ylim(0, max((v for v in vals if not np.isnan(v)), default=1) * 1.2)

    out = FIGURES / "fig3_main_results.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Architectural ablation (H7 visualization)
# ---------------------------------------------------------------------------

def fig4_ablation() -> None:
    apply_style("paper")

    cell_a = _load_jsonl(RESULTS / "cell_A_ranking.jsonl")
    cell_b = _load_jsonl(RESULTS / "cell_B_ranking.jsonl")
    cell_c = _load_jsonl(RESULTS / "cell_C_ranking.jsonl")
    cell_e = _load_jsonl(RESULTS / "cell_E_ranking.jsonl")
    hyp = _load_hypothesis_results()

    rows = [
        ("overlay-off (decomposed)",     _avg(cell_a, "ndcg_10")),
        ("noise-on (decomposed)",        _avg(cell_b, "ndcg_10")),
        ("cultural-on (decomposed)",     _avg(cell_c, "ndcg_10")),
        ("cultural-on (monolithic)",     _avg(cell_e, "ndcg_10")),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: NDCG@10 across rows
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [OKABE["sky"], OKABE["yellow"], OKABE["vermillion"], OKABE["orange"]]
    ax1.barh(range(len(rows)), vals, color=colors)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("NDCG@10")
    ax1.invert_yaxis()
    ax1.set_title("Top-k quality")

    # Right: H7 — H6 effect size by architecture
    h7 = hyp.get("H7", {})
    if h7 and "decomposed_effect_mean" in h7:
        means = [h7["decomposed_effect_mean"], h7["monolithic_effect_mean"]]
        ax2.bar(["Decomposed", "Monolithic"], means,
                color=[OKABE["blue"], OKABE["orange"]])
        ax2.set_ylabel("H6 effect size (Naija classifier Δ)")
        verdict = "PASS" if h7.get("passed") else "FAIL"
        ax2.set_title(f"H7 falsifier: {verdict}", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "H7 results not available\n(run experiments first)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=8)
        ax2.axis("off")

    out = FIGURES / "fig4_architectural_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Cultural-validity panel (H1, H5, H6, H7)
# ---------------------------------------------------------------------------

def fig5_cultural_validity() -> None:
    apply_style("paper")
    hyp = _load_hypothesis_results()
    if not hyp:
        print("  no hypothesis results — skipping fig5")
        return

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))

    # Panel (a) — H1: Naija density delta
    h1 = hyp.get("H1", {})
    ax = axes[0, 0]
    if "effect_size_vs_off" in h1:
        es = h1["effect_size_vs_off"]
        lo, hi = h1.get("ci_off", [0, 0])
        ax.bar(["cultural − overlay-off"], [es],
               yerr=[[es - lo], [hi - es]], color=OKABE["vermillion"], capsize=4)
        ax.axhline(0.30, color="gray", linestyle="--", linewidth=0.5,
                   label="threshold (0.30)")
        ax.set_title(f"H1 (floor) — Naija density: {'PASS' if h1.get('passed') else 'FAIL'}", fontsize=8)
        ax.set_ylabel("tokens / 100 tokens")
        ax.legend(fontsize=6)
    else:
        ax.set_title("H1 — pending")

    # Panel (b) — H6: classifier score delta
    h6 = hyp.get("H6", {})
    ax = axes[0, 1]
    if "delta_mean" in h6:
        d = h6["delta_mean"]
        lo, hi = h6.get("ci", [0, 0])
        ax.bar(["cultural − noise"], [d],
               yerr=[[d - lo], [hi - d]], color=OKABE["green"], capsize=4)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_title(f"H6 (substantive) — Naija classifier: {'PASS' if h6.get('passed') else 'FAIL'}", fontsize=8)
        ax.set_ylabel("classifier P(naija) Δ")

    # Panel (c) — H5: within vs between persona similarity
    h5 = hyp.get("H5", {})
    ax = axes[1, 0]
    if "within_mean" in h5:
        ax.bar(["within-persona", "between-persona"],
               [h5["within_mean"], h5["between_mean"]],
               color=[OKABE["vermillion"], OKABE["sky"]])
        ax.set_title(f"H5 (substantive) — persona consistency: {'PASS' if h5.get('passed') else 'FAIL'}", fontsize=8)
        ax.set_ylabel("TF-IDF cosine similarity")

    # Panel (d) — H7: architectural falsifier
    h7 = hyp.get("H7", {})
    ax = axes[1, 1]
    if "decomposed_effect_mean" in h7:
        ax.bar(["Decomposed", "Monolithic"],
               [h7["decomposed_effect_mean"], h7["monolithic_effect_mean"]],
               color=[OKABE["blue"], OKABE["orange"]])
        ax.set_title(f"H7 (C1 falsifier): {'PASS' if h7.get('passed') else 'FAIL'}", fontsize=8)
        ax.set_ylabel("H6 effect size by arch")

    out = FIGURES / "fig5_cultural_validity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Table 1 — Hypothesis summary (LaTeX booktabs)
# ---------------------------------------------------------------------------

def table1_hypothesis_summary() -> None:
    hyp = _load_hypothesis_results()
    if not hyp:
        print("  no hypothesis results — skipping table1")
        return

    rows: list[str] = []
    for hid in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        h = hyp.get(hid, {})
        cls = h.get("class", "—")
        passed = "\\checkmark" if h.get("passed") else "\\textbf{fail}"
        # effect-size column: try common keys
        es_keys = ["effect_size_vs_off", "delta_mean", "interaction_beta",
                   "within_mean", "cov_cult_mean"]
        es = next((h[k] for k in es_keys if k in h), None)
        es_str = f"{es:.3f}" if isinstance(es, (int, float)) else "—"
        rows.append(f"{hid} & {cls} & {es_str} & {passed} \\\\")

    out = FIGURES / "table1_hypothesis_summary.tex"
    out.write_text(
        "\\begin{tabular}{llrl}\n"
        "\\toprule\n"
        "Hypothesis & Class & Effect size & Outcome \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    print(f"  saved {out}")


# ---------------------------------------------------------------------------

def _load_hypothesis_results() -> dict:
    p = RESULTS / "hypothesis_results.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig3_main_results()
    fig4_ablation()
    fig5_cultural_validity()
    table1_hypothesis_summary()


if __name__ == "__main__":
    main()
