# Paper — TANDEM

LaTeX source for the BCT LLM Agent Challenge submission.

## Layout

- `main.tex` — full paper source
- `references.bib` — BibTeX entries; every citation manually verified against its arXiv / ACL Anthology page in May 2026
- `Makefile` — build automation
- `PREVIEW.md` — Markdown mirror of the prose for quick reading without a LaTeX install
- `../figures/` — figure PDFs and `plotstyle.py` (matplotlib shared style)

## Build

```bash
make pdf       # build main.pdf via latexmk (or use tectonic main.tex)
make clean     # remove LaTeX intermediates
```

Requires either `tectonic` (recommended; auto-downloads packages) or a full TeX Live install with `latexmk`, `pdflatex`, `bibtex`, and the `tcolorbox` / `tikz` / `booktabs` / `natbib` packages.

## Style commitments

- 9pt body, two-column layout, A4
- `booktabs` tables only (top/mid/bot rules; no vertical lines)
- Figures via vector PDF (matplotlib output; see `../figures/plotstyle.py`)
- Okabe-Ito colorblind-safe palette
- `natbib` numeric citations, sorted and compressed
- No emoji, no marketing language, no hype words
