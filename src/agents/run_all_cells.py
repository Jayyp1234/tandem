"""Orchestrator: run all 5 experiment cells per experiment_matrix.md §1.

Cells (architecture × condition):
  - Cell A: decomposed × overlay-off
  - Cell B: decomposed × noise-on
  - Cell C: decomposed × cultural-on   (headline)
  - Cell D: monolithic × noise-on
  - Cell E: monolithic × cultural-on   (H7 falsifier)

Cell (monolithic × overlay-off) is intentionally cut as redundant.

Run via:  make experiments
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents import recommender, simulator
from src.llm.client import GroqClient

CELLS: list[tuple[str, str, str]] = [
    ("A", "decomposed", "overlay-off"),
    ("B", "decomposed", "noise-on"),
    ("C", "decomposed", "cultural-on"),
    ("D", "monolithic", "noise-on"),
    ("E", "monolithic", "cultural-on"),
]


def main() -> None:
    personas = [
        json.loads(l)
        for l in Path("data/personas_20.jsonl").read_text().split("\n")
        if l.strip()
    ]
    items_meta = {
        item["item_id"]: item
        for item in (
            json.loads(l)
            for l in Path("data/beauty_5core/items.jsonl").read_text().split("\n")
            if l.strip()
        )
    }

    Path("results").mkdir(exist_ok=True)
    client = GroqClient(cache_path="cache/llm_responses.jsonl")

    for cell, arch, cond in CELLS:
        sim_out = Path(f"results/cell_{cell}_simulator.jsonl")
        rec_out = Path(f"results/cell_{cell}_ranking.jsonl")
        print(f"\n=== Cell {cell}: {arch} × {cond} ===")
        simulator.run_cell(
            client=client,
            personas=personas,
            items_meta=items_meta,
            condition=cond,
            architecture=arch,
            output_path=sim_out,
        )
        recommender.rank_and_score(
            simulator_output_path=sim_out,
            personas=personas,
            output_path=rec_out,
        )
        print(f"  cell {cell} done — wrote {sim_out} and {rec_out}")


if __name__ == "__main__":
    main()
