# TANDEM paper — build automation.
# Usage:
#   make pdf      build main.pdf via latexmk
#   make figures  regenerate all figures from data + figures/plotstyle.py
#   make clean    remove LaTeX intermediates
#   make veryclean also remove main.pdf

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

PAPER := main
FIGDIR := ../figures

.PHONY: pdf figures clean veryclean check

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib $(wildcard $(FIGDIR)/*.pdf)
	$(LATEXMK) $(LATEXMK_FLAGS) $(PAPER).tex

figures:
	cd $(FIGDIR) && python plotstyle.py
	@echo "  (paper figures are generated from experiment outputs via the top-level Makefile)"

check:
	@echo "Language check:"
	@! grep -iE 'groundbreaking|revolutionary|cutting.edge' $(PAPER).tex \
	    || (echo "BANNED WORD FOUND in $(PAPER).tex" && false)
	@echo "  ok"

clean:
	$(LATEXMK) -c
	rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.fdb_latexmk *.fls

veryclean: clean
	rm -f $(PAPER).pdf
