# Counterfactual Cultural-Validity Evaluation Protocol

*Pre-registered protocol for evaluating Contribution C2 (Nigerian persona overlay). Every hypothesis below has one falsifying metric. The git commit history of this file establishes pre-registration timestamps relative to the experiment runs.*

---

## 1. Why this protocol exists

Existing LLM user-simulator evaluation (RecAgent, Agent4Rec) measures **faithfulness to real-user interaction logs** — alignment win-rate, discrimination accuracy, human believability. These are good metrics for "does the agent behave like real users in our training data."

This protocol measures something distinct: **cultural-conditioning consistency** — when the cultural overlay is toggled on the *same* persona, on the *same* candidate items, with all other variables held constant, do the simulator's outputs shift in measurable, culturally-appropriate, pre-registered ways?

Counterfactual framing (overlay-on vs overlay-off, ceteris paribus) controls for the candidate pool, the persona, and the simulator parameters. The only variable is the overlay. This isolates the cultural effect from confounds.

The hackathon rubric explicitly weights "Behavioural Fidelity (human eval)" on Task A and "Contextual Relevance (human eval)" on Task B. The simulator literature offers no protocol for the cultural variant of these axes. We propose one.

---

## 2. Setup

**Items.** N = 200 items from the held-out Amazon Beauty 5-core test partition, balanced across four product categories: skin care (50), hair care (50), makeup (50), fragrance (50). Items selected by stratified random sampling within category.

**Personas.** 20 synthetic personas, constructed from 20 real held-out users' interaction histories (de-identified). Each persona is run through TANDEM's user simulator under three conditions, all other inputs identical.

**Three conditions × two architectures (required).** Each (persona × item) pair is generated under three overlay conditions:
1. **overlay-off** (control / no Nigerian-context content)
2. **noise-overlay-on** (length-matched random Nigerian-vocabulary tokens injected into the system prompt — same token count as the cultural overlay, no semantic structure)
3. **cultural-overlay-on** (the structured overlay specified in `overlay_spec.md`)

…run on **two architectures**:
- **TANDEM-decomposed**: simulator and recommender are two distinct LLM agents; overlay applied to simulator system prompt only
- **TANDEM-monolithic**: one combined LLM with a stage-prompt module that activates the overlay during the simulation step (predicts review + rating, then re-ranks within the same model context)

This 3 × 2 = 6 conditions per (persona, item) is the headline experiment. The architectural claim (C1) is falsified if the cultural-overlay-on minus overlay-off delta is statistically equivalent across both architectures. C1 is empirically validated if the decomposed architecture produces cleaner / larger deltas (effect size, CI, statistical significance).

The cultural overlay's effect is the difference between (3) and (2), not just (3) and (1). This isolates cultural-content effects from prompt-length / verbosity confounds — without it, a reviewer can collapse our findings into "longer prompts produce different outputs."

**Generations.** For each (persona × item × condition × architecture) TANDEM's simulator (or monolithic stage-prompt) emits:
- Predicted star rating ∈ {1, 2, 3, 4, 5}
- Predicted review text, capped at 100 tokens

Total generations: 200 items × 20 personas × 3 conditions × 2 architectures = 24 000 (rating, review) pairs (4 000 per (condition × architecture) cell).

**Determinism.** Fixed seed per (persona, item, condition) tuple so the experiment is replayable from the response cache. Temperature pinned at 0.7 (commonly used for review-style generation).

---

## 3. Measurements (the four deltas)

### 3.1 Lexical delta — does the language actually change?

- **Naija n-gram density**: count of NaijaLex tokens (unigrams + bigrams) per 100 generated tokens. Lexicon source: §1 of `cultural_resources.md`.
- **Code-switch frequency**: number of English ↔ Pidgin transitions per review. Transition detected by a token-level classifier trained on the BBC-Naija corpus split (or, if classifier unavailable, by a rule-based detector seeded from the marker list in `cultural_resources.md` §2).

Both metrics computed per review; aggregated by mean ± 95% bootstrap CI per condition. Paired comparison via Wilcoxon signed-rank test.

### 3.2 Rating-distribution delta — does sentiment intensity move?

- **Wilcoxon signed-rank test** on per-(persona, item) rating differences (cultural-overlay-on minus overlay-off; and separately cultural-on minus noise-on). Paired structure exploits within-pair correlation; the appropriate test for matched samples.
- **Mean rating shift**, paired by (persona, item), with 95% bootstrap CI.
- **Per-category Wilcoxon** for the four product categories. We expect skin-care (high cultural salience) to show a larger shift than fragrance (low cultural salience).
- **Marginal Kolmogorov–Smirnov** on the aggregate rating distributions reported as a secondary, complementary check (not the primary inference).

### 3.3 Sentiment delta — direction of shift

- **Sentiment score** per review from `cardiffnlp/twitter-roberta-base-sentiment-latest` (or equivalent open-weight classifier; chosen for robustness to social-media-style English, which is closer to Nigerian register than Wikipedia-trained classifiers).
- **Per-item sentiment shift** between overlay conditions, paired.
- **Direction-conditional analysis**: sentiment is expected to shift *more positive* when the overlay-on review mentions culturally-affirmed ingredients (shea butter, black soap, palm kernel oil), and *more negative* (or neutral) when no such mention occurs.

### 3.4 Topic-coverage delta — are the right things being said?

- **Cultural-topic vocabulary** defined in `cultural_resources.md` §3: shea-butter / black-soap / palm-kernel-oil / climate / religious-dietary / brand-familiarity tokens.
- **Coverage rate**: fraction of reviews in each condition mentioning ≥ 1 cultural-topic token.
- **Coverage delta**: relative increase in cultural-topic mention rate from overlay-off to overlay-on.

---

## 4. Pre-registered hypotheses

Committed *before* running the experiment. Reported with effect size, 95% CI, and pass/fail. Negative results are reported honestly alongside positive ones; the paper's Results section reflects whatever the data showed.

**H1 and H4 are floor checks** (sanity tests that the overlay produces *any* effect). **H2, H3, H5, H6 are the substantive cultural-validity tests.** **H7 is the C1 falsifier.**

| ID  | Class | Hypothesis | Falsified if |
|----|-----|-----------|-----------|
| H1 | Floor | Cultural-overlay-on Naija n-gram density exceeds overlay-off by ≥ 0.30 per 100 tokens, AND exceeds noise-overlay-on at α = 0.05 (Wilcoxon paired) | Either comparison fails |
| H2 | Substantive | Skin-care rating distributions: cultural-on vs overlay-off differ at α = 0.05 (Wilcoxon paired, cluster-bootstrap on persona); cultural-on vs noise-on also differ at α = 0.05 | Either comparison fails |
| H3 | Substantive | In a regression `sentiment ~ overlay_condition × ingredient_mentioned + length + persona_RE`, the (cultural-on × ingredient-mentioned) interaction term β > 0 at α = 0.05 | Interaction β ≤ 0 or non-significant |
| H4 | Floor | Cultural-topic coverage rate (cultural-on) significantly exceeds coverage rate (noise-on) with non-overlapping 95% bootstrap CIs | CIs overlap |
| H5 | **Substantive (cultural validity, not in prompt)** | Within-persona style consistency (cosine similarity of TF-IDF vectors of reviews from same persona on different items in cultural-on condition) significantly exceeds between-persona consistency at α = 0.05. Tests whether the persona is internalized vs. template-substituted. | No significant within-vs-between gap |
| H6 | **Substantive (cultural validity, classifier-based)** | A small Naija-vs-English classifier (logistic regression on TF-IDF, trained on MasakhaNEWS-pcm + English news subset) scores cultural-overlay-on outputs as significantly more Naija-like than noise-overlay-on outputs (paired Wilcoxon on per-pair classifier scores at α = 0.05). | No significant classifier-score gap |
| H7 | **C1 falsifier (architectural)** | The cultural-overlay-on minus overlay-off effect size on H6 (Naija-classifier score) is significantly larger under TANDEM-decomposed than under TANDEM-monolithic, at α = 0.05 (paired bootstrap on persona-level effect sizes) | Effect sizes equivalent across architectures, OR monolithic is larger — either case empirically falsifies the "compositional control" claim |

H1 is the floor: if H1 fails, the overlay is a sticker, not an overlay, and we report that honestly.

---

## 5. Human evaluation — small-N, scope-honest

- Sample 50 (persona, item) pairs from the 4 000 — stratified across product category and overlay condition.
- Each pair shown to N = 5 Nigerian raters recruited via the WhatsApp community channel for the hackathon (consent recorded; no PII collected on raters).
- Five-point Likert on three axes:
  - **Cultural appropriateness** ("Does this review sound like something a Nigerian user might actually write?")
  - **Linguistic naturalness** ("Is the language natural Nigerian English / Pidgin?")
  - **Plausibility as a real review** ("Does the rating-text combination read as authentic?")
- Inter-rater agreement: Krippendorff's α (target ≥ 0.6, reported regardless).
- Comparison: overlay-on vs overlay-off Likert means per axis, paired Wilcoxon.

Limitations explicitly disclosed: N = 5, sample = 50 (per condition), single recruitment channel (community-sampled, not population-representative), no cross-validation with other Naija-speaking populations.

---

## 6. Reporting

For each of H1–H4: effect size, 95% CI, p-value where applicable, pass / fail.
For human eval: paired Likert means with CIs, Krippendorff's α with target.
Negative results reported as such, with one-paragraph analysis of why each failure occurred.
All generated reviews and all rater responses shipped in the repository under an open license, redacted of any PII.

A single ablation table in the paper presents the four deltas across overlay conditions, per category, with human-eval scores in a sidebar.

---

## 7. Threats to validity

| Threat | Mitigation |
|---|---|
| Sentiment classifier is English-trained; Pidgin sentiment may be misclassified | Spot-check 30 reviews manually; report agreement with classifier |
| Synthetic personas may over-represent culturally-active users vs. typical held-out users | Sample personas to match observed user-frequency distribution in held-out set |
| Small human-rater sample (N = 5) limits effect-size precision | Report Krippendorff's α; do not generalize beyond the studied population |
| Confound: a more verbose overlay could inflate coverage without genuine cultural shift | Control for review length in topic-coverage analysis (regress out token count) |
| Selection bias from WhatsApp recruitment | Disclose channel; explicitly scope claims to "convenience sample of Nigerian raters" |
| Overlay leakage: model may infer Nigerian persona from history alone, even with overlay off | Run a no-history control on a 50-item subset; verify overlay-off baseline does not exhibit overlay-on lexical signature |

---

## 8. What this protocol is *not*

- Not a population-level claim about Nigerian preferences. We do not have the sample size for that.
- Not a clinical-trial-grade validation of the overlay. Hackathon scope.
- Not a substitute for native-speaker evaluation at scale. We frame this as a first protocol; future work scales the human-eval N.

---

## 9. Pre-registration

This protocol is committed in this file before any overlay-on experiments are run. The git commit hash establishes timestamp. Any deviation between this document and the eventually-reported results is itself reported.

---

## 10. Open questions for the analysis stage

- Which two of the four deltas anchor the main ablation figure in the paper, vs. which go in the appendix? Likely H1 (lexical) and H4 (topic coverage) for the main figure; H2 (rating distribution) and H3 (sentiment direction interaction) for an appendix or extended analysis.

*(The noise-overlay control is no longer deferred — it is now required and is part of §2's three-condition setup.)*
