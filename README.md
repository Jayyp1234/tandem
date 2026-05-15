---
title: TANDEM — Two-Agent LLM Recommender
emoji: 💄
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Two-agent LLM recommender with a Nigerian persona overlay
---

# TANDEM

A two-agent LLM recommender: a **user simulator** predicts per-candidate reviews and ratings, a **recommender** ranks by those predictions. A Nigerian persona overlay attaches to the simulator alone, enabling counterfactual cultural-validity evaluation.

Submission for the **DSN × Bluechip Tech LLM Agent Challenge** (2026).

**Authors.** Johnpaul Okeke Ebube, Eruja Micheal, Emmanuel Solomon — Petroleum and Gas Engineering, University of Lagos.

---

## 🚀 Live API (hosted on HuggingFace Spaces)

| | |
|---|---|
| **Interactive Swagger UI** | **https://heisienberg-tandem.hf.space/docs** |
| **Space page** | https://huggingface.co/spaces/heisienberg/tandem |
| **Health check** | https://heisienberg-tandem.hf.space/health |
| **Task A (simulator)** | `POST https://heisienberg-tandem.hf.space/simulator/predict` |
| **Task B (recommender, deterministic)** | `POST https://heisienberg-tandem.hf.space/recommender/recommend` |
| **Task B (agentic — plan → score → reflect)** | `POST https://heisienberg-tandem.hf.space/recommender/recommend-agentic` |
| **Task B (multi-turn / conversational)** | `POST https://heisienberg-tandem.hf.space/recommender/converse` |

Try it from your browser at **`/docs`** — fill the JSON request body, click *Execute*, see the live response.

> Free-tier Spaces sleep after extended inactivity. **The first request after a sleep takes ~30 seconds to warm up**; subsequent requests respond in 1–3 s. The container holds a Groq free-tier `GROQ_API_KEY` as a HF Space secret, so live calls go to Llama-3.1-8B-Instant.

### One-liner sanity check

```bash
curl https://heisienberg-tandem.hf.space/health
# {"status":"ok","service":"TANDEM","version":"0.1.0"}
```

### Real Task A response (cultural-on overlay, Igbo Christian persona)

```bash
curl -X POST https://heisienberg-tandem.hf.space/simulator/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "persona": {
      "persona_id": "demo_adaeze",
      "history_window": [{"item_id":"X","rating":5.0,"summary":"shea butter dey work well well for my skin"}],
      "naija_name": "Adaeze Okonkwo",
      "ethnic_hint": "Igbo",
      "religious_hint": "Christian"
    },
    "item": {
      "item_id": "Y",
      "title": "Clinique iD Custom Blend Hydrator",
      "brand": "Clinique",
      "category": "skincare"
    },
    "condition": "cultural-on"
  }'
```

Response (verbatim from the live Space):

```json
{
  "rating": 3.0,
  "review": "E be like say Clinique iD Custom Blend Hydrator no bad, but e no dey meet my expectations. E no contain dudu-osun or ori, so e no go work well for my deep complexion. Shea butter dey inside, but e no enough to make much difference. I dey prefer natural products like shea butter dey work well well for my skin. Clinique good good, but e no real deal for me.",
  "model": "llama-3.1-8b-instant",
  "cached": false
}
```

Notice the Naija-Pidgin code-switching (*"E be like say"*, *"no dey"*, *"dey work well well"*), mention of **dudu-osun** (traditional Yoruba black soap), and **ori** (Yoruba for shea butter) — the three layers of the cultural overlay (linguistic, preference, grounding) firing together on a single call.

### Real agentic Task B response (plan → score → reflect)

```bash
curl -X POST https://heisienberg-tandem.hf.space/recommender/recommend-agentic \
  -H 'Content-Type: application/json' \
  -d '{
    "persona": {
      "persona_id": "demo_agentic_adaeze",
      "history_window": [{"item_id":"H1","rating":5.0,"summary":"shea butter from Zaron, real natural feel"}],
      "naija_name": "Adaeze Okonkwo", "ethnic_hint": "Igbo", "religious_hint": "Christian"
    },
    "candidates": [
      {"item_id":"C1","title":"Cetaphil Gentle Skin Cleanser","brand":"Cetaphil","category":"skincare"},
      {"item_id":"C2","title":"Dudu-Osun African Black Soap","brand":"Tropical Naturals","category":"skincare"},
      {"item_id":"C3","title":"Tarte Maneater Eyeliner","brand":"Tarte","category":"makeup"},
      {"item_id":"C4","title":"Zaron Cosmetics Foundation","brand":"Zaron","category":"makeup"}
    ],
    "top_k": 3, "reflect_window": 4, "condition": "cultural-on"
  }'
```

Response (verbatim from the live Space):

```json
{
  "priorities": [
    "Natural ingredients",
    "Cruelty-free and vegan-friendly",
    "Products endorsed by or made by Igbo or Nigerian brands"
  ],
  "ranked": [
    {"item_id":"C2","rank":1,"title":"Dudu-Osun African Black Soap",
     "predicted_rating":4.0,"reason":"Dudu-Osun is made with natural ingredients like ori and palm kernel oil."},
    {"item_id":"C4","rank":2,"title":"Zaron Cosmetics Foundation",
     "predicted_rating":4.0,"reason":"Zaron is endorsed by a Nigerian brand and fits the user's dark skin perfectly."},
    {"item_id":"C1","rank":3,"title":"Cetaphil Gentle Skin Cleanser",
     "predicted_rating":3.0,"reason":"Cetaphil has natural-leaning ingredients matching the user's priority."}
  ],
  "trace": {
    "score_order_top_k":   ["C2","C4","C3"],
    "agentic_order_top_k": ["C2","C4","C1"],
    "reasoning_calls": 2, "score_calls_total": 4
  }
}
```

The reflector overrode the simulator's higher Tarte rating in service of the planned priority *"products by Igbo or Nigerian brands"* — Cetaphil replaced Tarte in the agentic top-3.

### Real multi-turn / conversational response

```bash
# Turn 1 — open the session
curl -X POST https://heisienberg-tandem.hf.space/recommender/converse \
  -H 'Content-Type: application/json' \
  -d '{
    "persona": {
      "persona_id":"demo_converse_damilola",
      "history_window":[{"item_id":"H1","rating":5.0,"summary":"loved my Zaron foundation"}],
      "naija_name":"Damilola Adeyemi","ethnic_hint":"Yoruba","religious_hint":"Muslim"
    },
    "candidates":[
      {"item_id":"C1","title":"Tarte Maneater Eyeliner","brand":"Tarte","category":"makeup"},
      {"item_id":"C2","title":"Dudu-Osun African Black Soap","brand":"Tropical Naturals","category":"skincare"},
      {"item_id":"C3","title":"Synthetic Halo-Wire Hair Extension","brand":"Generic","category":"hair"},
      {"item_id":"C4","title":"Zaron Cosmetics Foundation","brand":"Zaron","category":"makeup"}
    ],
    "top_k":3,"condition":"cultural-on"
  }'

# returns session_id, top-3 ranking, assistant_reply

# Turn 2 — refine via natural-language feedback
curl -X POST https://heisienberg-tandem.hf.space/recommender/converse \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<from turn 1>","message":"me sef no fit use synthetic, na human hair I dey like"}'
```

The simulator in turn 2 incorporated the feedback verbatim (its regenerated review of the synthetic-hair candidate quoted *"me sef no fit use synthetic, na human hair I dey like"* back at the user) and dropped that candidate to a 2.0 rating.

---

## What's in this submission

| Deliverable | Where |
|---|---|
| **Paper PDF (8 pages)** | `paper/main.pdf` |
| **Live API on HuggingFace Spaces** | https://heisienberg-tandem.hf.space/docs |
| **Containerized API (same image, run locally)** | `Dockerfile`, `docker-compose.yml`, `src/api/main.py` |
| **Code repository** | `src/`, `pyproject.toml`, `Makefile` |
| **LLM response cache** (~22k entries; reproduces every paper number) | `make fetch-artifacts` mirrors from [huggingface.co/datasets/heisienberg/tandem-artifacts](https://huggingface.co/datasets/heisienberg/tandem-artifacts) |
| **Live demo log** (curl output against both endpoints) | `results/api_demo.txt` |
| **Pre-registered protocol + supporting docs** (hypotheses, overlay spec, lit notes) | `research/protocol_spec.md`, `research/overlay_spec.md`, `research/literature_evidence.md` |

---

## Headline findings (Section 5 of the paper)

- **Cultural conditioning works.** Three pre-registered substantive tests pass: cultural-topic coverage 22%→57% (H4), within-persona TF-IDF consistency (p≈0, H5), Naija classifier authenticity (Δ=0.025, p<10⁻²⁵, H6).
- **Architectural falsifier (H7) is null and honestly reported.** Decomposed and monolithic produce equivalent cultural-classifier effects.
- **Cold-start is TANDEM's strongest regime.** With 1 prior interaction (vs 10), NDCG@10 doubles (0.080→0.166), Hit@5 goes 5× with non-overlapping 95% CIs ([.000, .150] vs [.100, .450]). The zero-shot persona-text reasoning generalizes; long-history versions over-fit to past-purchase vocabulary.
- **Task A submission row** (overlay-off): RMSE 1.32, ROUGE-L 0.122, BERTScore-F1 0.513, sentence-cosine 0.572.

---

## Reproduce the paper numbers in one command (no API calls)

```bash
git clone https://github.com/Jayyp1234/tandem.git && cd tandem
make install fetch-artifacts eval-from-cache
```

That's it. `fetch-artifacts` pulls the two large files (~80 MB total) from a public [HuggingFace Datasets mirror](https://huggingface.co/datasets/heisienberg/tandem-artifacts):

| File | Size | Source |
|---|---|---|
| `cache/llm_responses.jsonl` | 47 MB | ~22k cached LLM generations (5-cell matrix + 2 baselines + cold-start head-to-head) |
| `data/beauty_5core/items.jsonl` | 33 MB | Amazon Beauty 5-core item metadata |

`eval-from-cache` runs H1–H7 + Task A + Task B metric aggregations from the local cache. No API access, no Groq key. Under 30 minutes on a laptop.

For figures, tables, and PDF rebuild:

```bash
make figures && make pdf       # requires Tectonic for the LaTeX compile
```

## Run the live API locally (same Docker image as the HF Space)

If you'd rather run the container on your own machine:

```bash
echo 'GROQ_API_KEY=gsk_<your_free_key>' > .env
docker compose up              # builds and starts the FastAPI service on :8000

# Health check
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"TANDEM","version":"0.1.0"}

# Interactive API docs (Swagger UI)
open http://127.0.0.1:8000/docs

# Task A endpoint sketch
curl -X POST http://127.0.0.1:8000/simulator/predict \
     -H 'Content-Type: application/json' \
     -d '{"persona": {...}, "item": {...}, "condition": "cultural-on"}'

# Task B endpoint sketch
curl -X POST http://127.0.0.1:8000/recommender/recommend \
     -H 'Content-Type: application/json' \
     -d '{"persona": {...}, "candidates": [...], "top_k": 10}'
```

A working example of both calls and the exact responses received in our test run is in `results/api_demo.txt`. **The hosted version on HuggingFace Spaces** (URL at the top of this README) is the same Docker image — judges can hit it from a browser at `/docs` without any local setup.

## Run the full experiments from scratch (~3 days at Groq free tier)

```bash
make data && make personas
make experiments               # ~14,000 LLM calls; auto-rotates across .env keys
make eval-from-cache && make figures && make pdf
```

`GROQ_API_KEY` (single) or `GROQ_API_KEY_1`...`_16` (rotation across multiple free-tier accounts) supported. See `.env.example`.

---

## Repository layout

```
.
├── paper/
│   ├── main.pdf            # the submission paper (compiled)
│   ├── main.tex            # LaTeX source
│   ├── references.bib      # 20+ verified citations
│   ├── PREVIEW.md          # Markdown mirror of the paper for quick reading
│   └── Makefile            # paper-only build
├── cache/
│   └── llm_responses.jsonl # response cache, ~22k entries, ~47 MB (fetch via `make fetch-artifacts`)
├── results/
│   ├── hypothesis_results.json     # H1–H7 outcomes
│   ├── task_a_metrics.json         # RMSE/MAE/ROUGE/BERTScore-F1/sim
│   ├── task_b_metrics.json         # NDCG@10/Hit@10/MRR per cell
│   ├── cold_start_summary.json     # cold-start NDCG@10 doubling vs full history
│   ├── qualitative_examples.md     # 10 paired (overlay-off vs cultural-on) reviews
│   └── api_demo.txt                # live container demo output
├── figures/
│   ├── plotstyle.py        # Okabe-Ito palette, shared matplotlib style
│   └── *.pdf               # generated figures (regenerable via `make figures`)
├── research/
│   ├── protocol_spec.md              # pre-registered H1–H7 protocol
│   ├── overlay_spec.md               # 3-layer Nigerian overlay specification
│   ├── cultural_resources.md         # Naija linguistic + cosmetics resources with licenses
│   ├── literature_evidence.md        # annotated references to prior work
│   └── lit_review.md                 # related-work search notes
├── src/
│   ├── llm/                # Groq client (multi-key rotation) + JSONL cache
│   ├── data/               # Amazon Beauty preprocessing + persona construction
│   ├── overlay/            # 3-layer cultural overlay (linguistic / preference / grounding)
│   ├── agents/             # simulator + recommender (decomposed and monolithic)
│   ├── baselines/          # P5-zero, Chat-Rec on the same model and protocol
│   ├── eval/               # H1–H7 + Task A + Task B + cold-start + figures
│   └── api/main.py         # FastAPI: /simulator/predict + /recommender/recommend
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md               # this file
```

---

## Reproducibility commitments

- **Cache mirrored.** ~22,000 LLM generations cached to JSONL and served from a public HuggingFace Datasets mirror; `make fetch-artifacts && make eval-from-cache` reproduces every paper number with no API calls.
- **Open-weight model.** Primary system runs on Llama-3.1-8B-Instruct via Groq's free tier — no closed-model dependency.
- **Pinned versions.** Model ID, temperature, seeds, and tokenizer all pinned in `src/llm/client.py`.
- **One-command reproduce.** `make eval-from-cache && make figures && make pdf` rebuilds the paper end-to-end in under 30 minutes from a clean clone.
- **Containerized.** `docker compose up` brings up both Task A and Task B endpoints with a single command.

---

## License

- Code: MIT.
- Cultural-overlay specifications and Naija linguistic resources: see `research/cultural_resources.md` for source-by-source license notes. Shipped artifacts are CC-BY-4.0 or MIT. NaijaSenti-Twitter is referenced but not redistributed.

---

## What we did *not* do, and why

- **Cross-domain experiment.** We use Amazon Beauty only. A cross-domain demonstration would require either re-running the full matrix on a second dataset or showing zero-shot transfer; both are larger commitments than the 14-day window allowed. The paper flags Yelp's Naija-cuisine and African-restaurant subsets as the natural next step.
- **Native-Naija lexical resource.** The overlay produces a Naija/WAPE blend, not pure Naija. The BBC-Pidgin and UD_Naija-NSC corpora that anchor our linguistic layer are documented (Adelani et al., 2024) as leaning toward West African Pidgin English; native Lagos speakers will recognize the words but may find some register slightly off. A pure-Naija extraction is flagged as the most consequential follow-up.
- **EXP3RT direct reproduction.** EXP3RT (Kim et al., 2025) is cited in related work but not invoked as a baseline; reproducing its fine-tuning pipeline was out of scope. We treat its monolithic structure as the contrast condition for our H7 falsifier.

These limitations are stated up front in `paper/main.pdf` §7 (Discussion and Limitations).
