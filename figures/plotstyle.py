"""TANDEM paper — shared matplotlib style.

Every figure in the paper imports `apply_style()` from this module before
plotting. This guarantees consistent fonts, colors, and sizing across all
figures in the paper, which is what separates "looks like a hackathon paper"
from "looks like a top-tier venue submission."

Usage:
    from plotstyle import apply_style, OKABE
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(x, y, color=OKABE['blue'])
    fig.savefig('fig3_main_results.pdf', bbox_inches='tight')
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette: Okabe-Ito (colorblind-safe, used by NeurIPS / Nature etc.)
# ---------------------------------------------------------------------------

OKABE = {
    "black":     "#000000",
    "orange":    "#E69F00",  # overlay toggle
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",  # simulator agent
    "vermillion":"#D55E00",  # recommender agent / TANDEM bars
    "purple":    "#CC79A7",
}

CONDITION_COLORS = {
    "overlay-off":   OKABE["sky"],
    "noise-on":      OKABE["yellow"],
    "cultural-on":   OKABE["vermillion"],
}

ARCHITECTURE_COLORS = {
    "decomposed":  OKABE["blue"],
    "monolithic":  OKABE["orange"],
}


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style(context: str = "paper") -> None:
    """Apply TANDEM paper style to matplotlib's rcParams.

    Args:
        context: "paper" (default, single-column figure size) or
                 "wide" (column-spanning figure size).
    """
    sizes = {
        "paper": (3.5, 2.4),  # single matplotlib column ~ 3.5" wide
        "wide":  (7.0, 4.0),  # column-spanning
    }
    fig_w, fig_h = sizes.get(context, sizes["paper"])

    mpl.rcParams.update({
        # Typography
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":       9,
        "axes.titlesize":  10,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # Figure size & layout
        "figure.figsize":   (fig_w, fig_h),
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,
        "figure.constrained_layout.use": True,

        # Axes & ticks (clean, minimal)
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3,
        "ytick.major.size":  3,

        # Lines & bars
        "lines.linewidth":   1.5,
        "lines.markersize":  4,
        "patch.edgecolor":   "white",
        "patch.linewidth":   0.5,

        # Color cycle (first six Okabe-Ito colors)
        "axes.prop_cycle":   plt.cycler(
            color=[OKABE["blue"], OKABE["vermillion"], OKABE["green"],
                   OKABE["orange"], OKABE["sky"], OKABE["purple"]],
        ),

        # Output formats
        "pdf.fonttype":      42,  # TrueType (embeddable, vector)
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
    })


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def add_significance_bracket(ax, x1: float, x2: float, y: float,
                              text: str = "*", linewidth: float = 0.7) -> None:
    """Draw a significance-bracket between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y * 1.02, y * 1.02, y],
            color="black", linewidth=linewidth)
    ax.text((x1 + x2) / 2, y * 1.04, text,
            ha="center", va="bottom", fontsize=8)


def cluster_bootstrap_ci(values, clusters, n_boot: int = 1000,
                         ci: float = 0.95, rng=None):
    """95% cluster-bootstrap CI on `values` clustered by `clusters`.

    Used for headline numbers to mitigate within-persona observation
    correlation when reporting confidence intervals.
    """
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(0)
    values = np.asarray(values)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters),
                              replace=True)
        # Build resampled value vector from sampled clusters
        idxs = np.concatenate(
            [np.where(clusters == c)[0] for c in sampled]
        )
        boot_means[i] = values[idxs].mean()
    alpha = (1 - ci) / 2
    return (
        float(values.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1 - alpha)),
    )


if __name__ == "__main__":
    # Quick smoke test: render the palette as bars to confirm the style applies.
    apply_style("paper")
    import numpy as np
    fig, ax = plt.subplots()
    names = list(OKABE.keys())
    vals = np.arange(len(names))
    ax.bar(names, vals, color=[OKABE[n] for n in names])
    ax.set_title("Okabe-Ito palette smoke test")
    ax.set_ylabel("index")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig("plotstyle_smoke.pdf")
    print("Saved plotstyle_smoke.pdf")
