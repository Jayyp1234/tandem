"""Eval orchestrator — runs H1–H7 against the experiment cache.

Run via:
  make eval               # runs experiments first, then evals
  make eval-from-cache    # skip experiments; eval the shipped cache only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval import hypotheses

RESULTS = Path("results")
DATA = Path("data/beauty_5core")


def _load_cell(name: str) -> list[dict]:
    p = RESULTS / f"cell_{name}_simulator.jsonl"
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()] if p.exists() else []


def _load_items_meta() -> dict[str, dict]:
    p = DATA / "items.jsonl"
    return {
        rec["item_id"]: rec
        for rec in (
            json.loads(l) for l in p.read_text().split("\n") if l.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="Eval shipped cache only; do not run experiments.")
    parser.add_argument("--output", default=str(RESULTS / "hypothesis_results.json"))
    args = parser.parse_args()

    cell_a = _load_cell("A")
    cell_b = _load_cell("B")
    cell_c = _load_cell("C")
    cell_d = _load_cell("D")
    cell_e = _load_cell("E")
    items_meta = _load_items_meta()

    results = {}
    print("\n=== Running H1 (floor — Naija density) ===")
    results["H1"] = hypotheses.test_h1(cell_a, cell_b, cell_c)
    print(json.dumps(results["H1"], indent=2))

    print("\n=== Running H2 (substantive — rating shift on skin-care) ===")
    results["H2"] = hypotheses.test_h2(cell_a, cell_b, cell_c, items_meta)
    print(json.dumps(results["H2"], indent=2))

    print("\n=== Running H3 (substantive — sentiment x ingredient interaction) ===")
    results["H3"] = hypotheses.test_h3({"overlay-off": cell_a, "noise-on": cell_b, "cultural-on": cell_c})
    print(json.dumps(results["H3"], indent=2))

    print("\n=== Running H4 (floor — cultural-topic coverage) ===")
    results["H4"] = hypotheses.test_h4(cell_b, cell_c)
    print(json.dumps(results["H4"], indent=2))

    print("\n=== Running H5 (substantive — persona consistency) ===")
    results["H5"] = hypotheses.test_h5(cell_c)
    print(json.dumps(results["H5"], indent=2))

    print("\n=== Running H6 (substantive — Naija classifier authenticity) ===")
    results["H6"] = hypotheses.test_h6(cell_b, cell_c)
    print(json.dumps(results["H6"], indent=2))

    print("\n=== Running H7 (C1 falsifier — decomposed vs monolithic) ===")
    results["H7"] = hypotheses.test_h7(cell_b, cell_c, cell_d, cell_e)
    print(json.dumps(results["H7"], indent=2))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  results written to {args.output}")
    print("  any test with passed=False is a negative result; report honestly.")


if __name__ == "__main__":
    main()
