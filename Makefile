# TANDEM — top-level build automation.
# Every target either populates a paper artefact or supports reproducibility.

PYTHON := python
UV := uv

.PHONY: help install data personas experiments eval eval-from-cache figures pdf clean check-key smoke

help:
	@echo "TANDEM — make targets:"
	@echo "  install           install dependencies (uv sync)"
	@echo "  check-key         verify GROQ_API_KEY is set"
	@echo "  data              download + preprocess Amazon Beauty 5-core"
	@echo "  personas          construct 20 personas from held-out users"
	@echo "  experiments       run full 5-cell matrix + baselines (~3 days clock, \$$0)"
	@echo "  eval              run all H1-H7 hypothesis tests"
	@echo "  eval-from-cache   reproduce paper numbers from shipped cache (no API calls)"
	@echo "  figures           generate all paper figures + tables"
	@echo "  pdf               build paper/main.pdf"
	@echo "  smoke             tiny end-to-end smoke test (1 persona x 1 item)"
	@echo "  clean             remove caches and generated artefacts"

install:
	$(UV) sync

check-key:
	@if [ -z "$$GROQ_API_KEY" ] && [ -z "$$GROQ_API_KEY_1" ] && [ ! -f .env ]; then \
		echo "ERROR: no Groq key set. Copy .env.example to .env and fill in either:"; \
		echo "  GROQ_API_KEY=...               (single key), or"; \
		echo "  GROQ_API_KEY_1=... .. GROQ_API_KEY_6=...  (rotation mode)"; \
		echo "Get a free key at https://console.groq.com/keys"; \
		exit 1; \
	fi
	@echo "  Groq key(s) configured."

data: data/beauty_5core/.done

data/beauty_5core/.done:
	$(PYTHON) -m src.data.beauty
	@touch data/beauty_5core/.done

personas: data/personas_20.jsonl

data/personas_20.jsonl: data/beauty_5core/.done
	$(PYTHON) -m src.data.personas

experiments: check-key data personas
	@echo "Running full experiment matrix at Groq free-tier limits (~30 RPM => ~8-10 hours wall-clock)."
	@echo "Cached results preserved across interrupts; safe to Ctrl-C and resume."
	$(PYTHON) -m src.eval.classifier   # train Naija classifier (idempotent; ~30s)
	$(PYTHON) -m src.agents.run_all_cells
	$(PYTHON) -m src.baselines

eval: experiments
	$(PYTHON) -m src.eval.run_all

eval-from-cache:
	$(PYTHON) -m src.eval.run_all --from-cache

# --- Task A / Task B / qualitative extras ---------------------------------
task-a:
	$(PYTHON) -m src.eval.task_a_metrics

task-b:
	$(PYTHON) -m src.eval.task_b_metrics

qual:
	$(PYTHON) -m src.eval.qualitative_examples

eval-extras: task-a task-b qual

# --- Cold-start experiment (Task B 25-pt rubric line) ---------------------
cold-start: check-key
	$(PYTHON) -m src.eval.cold_start

figures: eval
	$(PYTHON) -m src.eval.plot

pdf:
	$(MAKE) -C paper pdf

smoke: check-key
	$(PYTHON) -m src.agents.smoke

clean:
	rm -rf cache/* results/* figures/*.pdf data/beauty_5core/.done

# --- guardrails ---------------------------------------------------------

.PHONY: forbidden-language

forbidden-language:
	@! grep -riE 'groundbreaking|revolutionary|cutting.edge|state.of.the.art' paper/ \
	    || (echo "Hype word detected in paper/" && false)
	@echo "  language check passed."
