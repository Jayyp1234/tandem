# TANDEM: A Two-Agent Decomposition for Compositional Control in LLM-Based Recommendation, with a Nigerian Persona Case Study

*Full draft preview. The official build is `paper/main.tex` → `paper/main.pdf`.*

---

## Abstract

We pre-registered seven hypotheses for **TANDEM**, a two-agent LLM recommendation system in which a simulator predicts each user's review and rating for each candidate item and a recommender ranks by those predictions. A Nigerian persona overlay attaches to the simulator alone, holding the recommender fixed. On 20 personas and 100 candidates per persona, the overlay produces measurable cultural conditioning: cultural-topic coverage rises from 22% to 57% (H4); the simulator internalises personas with within-persona TF-IDF similarity significantly exceeding between-persona (p ≈ 0, H5); a held-out Naija-vs-English classifier scores cultural-on outputs significantly higher than a length-matched random-token control (Δ = 0.025, 95% CI [0.016, 0.033], p < 10⁻²⁵, H6). The architectural falsifier fails: applying the same overlay to a single monolithic LLM produces statistically indistinguishable cultural effects (Δ = 0.0006, 95% CI [−0.006, 0.006], H7). We report this null result honestly per our pre-registration: the cultural effect is real and measurable, but the two-agent decomposition does no additional work over a single-LLM stage-prompt control on the classifier axis. A post-hoc observation suggests the decomposition does matter on a different axis — ranking quality — which we flag for replication. The full experiment runs on Llama-3.1-8B-Instruct via Groq's free tier; the shipped response cache and `make eval-from-cache` reproduce every paper number in under thirty minutes.

---

## 1. Introduction

Adaeze Okonkwo, an Igbo Christian shopping for skincare in our experiments, gives the Clinique iD moisturizer two stars out of five. *"Me no be fan of Clinique, bros. Product no dey solid for my skin. I prefer Zaron or House of Tara products, dem dey work well for me. Clinique moisturizer too thick, no absorb quickly like my Mama's bubbling soap."*

Adaeze does not exist. She is persona `p00` in our experiments — a synthetic user constructed from one held-out interaction history in the Amazon Beauty corpus, given a name, an ethnic and religious context, and a structured Nigerian-Pidgin overlay applied to her LLM simulator. Her review of a product she has never seen was generated in 708 milliseconds by Llama-3.1-8B-Instruct, accessed through Groq's free tier. Among forty Nigerian-founded labels available in the overlay's preference layer, the model picked Zaron and House of Tara. The cultural reference to "Mama's bubbling soap" was not in any prompt.

This paper is about what makes that selection possible, and whether it generalises into a recommender that speaks its users' language in two senses at once.

We introduce TANDEM, a two-agent decomposition of LLM-based recommendation (Figure 1). The simulator predicts the target user's review and 1–5 star rating for each candidate. The recommender ranks candidates by those predicted ratings. The two agents share an open-weight model and a response cache; they differ only in their system prompts. A cultural overlay is a third component that attaches to the simulator alone, leaving the recommender's prompt untouched.

The decomposition is the contribution, not the model. Existing LLM-recommendation systems fuse simulator and recommender into one optimisation, and the seam shows. EXP3RT (Kim et al. 2025), the closest predecessor, fine-tunes a single Llama-3 to perform profile construction, reasoning, and rating prediction in sequence; the rating predictor is itself the reranker. ToolRec (Zhao et al. 2024) wraps an LLM in a tool-use loop in which the same model speaks as the user and as the recommender. UserMirrorer (Wei et al. 2026), RecoWorld (Liu et al. 2025), and Wang et al. (2025) pair an LLM simulator with a recommender, but consume the simulator's outputs at training time. None of these compositions admits a clean ablation in which the simulator changes while the recommender holds still. None permits the simulator to be evaluated independently. None supports the experiment we want to run.

The experiment we want to run is this: *toggle a cultural overlay on the simulator alone, hold everything else fixed, and measure what changes downstream.* Adelani et al. (2024) recently showed that frontier LLMs systematically misrepresent Naija (Nigerian Pidgin), conflating it with West African Pidgin English at register level. Their work establishes that the gap is real. It does not establish whether the downstream recommender suffers, or whether targeted intervention helps. We address both, with explicit acknowledgement of where our overlay still falls short of pure Naija (§7).

The cultural overlay is a structured three-layer prompt module: forty Naija-Pidgin code-switch markers from BBC-Pidgin via MasakhaNEWS-pcm and UD_Naija-NSC; six cosmetics-relevant cultural axes (skin tone, hair type, halal compatibility, ingredient priorities like *ori* and *dudu-osun*, harmattan-stable formulations, brand familiarity); and persona grounding (name, Yoruba/Igbo/Hausa-Fulani context, religious orientation), drawn in rough proportion to Nigerian census demographics.

The design admits a falsifier for the architectural claim itself (Hypothesis H7). If the cultural-overlay effect is equivalent under decomposed and monolithic architectures, the decomposition is a clerical convenience and we report that. We pre-registered this test, four substantive cultural-validity hypotheses (H2, H3, H5, H6), and two floor-check sanity tests (H1, H4) in the repository's git history before any cultural-overlay experiment ran.

**Contributions.**

1. TANDEM, a two-agent decomposition for LLM recommendation supporting independent evaluation and compositional overlays.
2. A Nigerian persona overlay grounded in publicly-licensed Naija corpora and cosmetics-specific anchors, with the WAPE/Naija blending limitation disclosed up front.
3. A counterfactual cultural-validity protocol with three conditions, two architectures, four pre-registered substantive hypotheses, an architectural falsifier, and a length-matched noise control.

The simulator response cache (~14,000 entries), preprocessed data, persona definitions, and the H6 classifier weights are shipped with the submission. `make eval-from-cache && make figures && make pdf` reproduces every number and figure from a clean clone in under thirty minutes, with no API access required.

---

## 2. Related Work

*[Section 2 prose unchanged from previous draft — three threads: LLM user simulators (RecAgent, Agent4Rec, AgentCF), LLM recommenders (P5, TALLRec, Chat-Rec, RecMind, LLaRA, EXP3RT), simulators-feeding-non-LLM-downstreams (UserMirrorer, RecoWorld), and cultural/multilingual NLP (Adelani et al., AfriBERTa, AfroXLMR).]*

---

## 3. Method

*[Section 3 prose unchanged from previous draft — overview/notation, simulator S, recommender R, cultural overlay O (linguistic + preferential + grounding), noise control, two architectures decomposed vs monolithic.]*

---

## 4. Experiments

*[Section 4 prose unchanged from previous draft — dataset (Amazon-Reviews-2023 All_Beauty, 20 personas × 100 candidates), protocol (3 conditions × 2 architectures = 5 cells, ~14,000 calls), seven hypotheses H1–H7, baselines (P5-zero, Chat-Rec), implementation details.]*

---

## 5. Results

We report results in three blocks: rating and review-text quality (Task A), top-k recommendation (Task B), and the pre-registered architectural ablation H7.

### 5.1 Rating accuracy and review-text quality (Task A)

For each persona we have the user's chronologically last interaction — the target — together with their actual rating and review text. We compare the simulator's prediction for the target against ground truth on two conditions: *overlay-off* (default Western-leaning persona) and *cultural-on* (Nigerian overlay).

| Condition | RMSE | MAE | ROUGE-1 | ROUGE-L | Sem-sim |
|---|---|---|---|---|---|
| overlay-off | 1.32 | 0.85 | 0.201 | 0.122 | 0.572 |
| cultural-on | 1.43 | 1.25 | 0.139 | 0.079 | 0.436 |

The *overlay-off* row is our Task A submission. Rating RMSE of 1.32 on the 5-point scale meaningfully beats the random baseline (RMSE ≈ 2.0); ROUGE-L of 0.122 reflects partial-overlap recovery typical of zero-shot review generation from a 10-item history. We do not claim top-of-leaderboard rating prediction; we claim a defensible Task A baseline against which the cultural overlay can be measured.

The *cultural-on* row is worse on every metric. We do not interpret this as the overlay degrading the simulator — we interpret it as the overlay successfully changing its behaviour. The held-out ground truth is in standard English from a US-Amazon user; cultural-on outputs are in Naija/Pidgin blend. Text-match metrics necessarily drop, and the rating diverges more from the actual user's. This trade-off is discussed in §7.

### 5.2 Top-k recommendation (Task B)

We rank each persona's 100-candidate set (target + 99 negatives, the SASRec/P5 protocol) by predicted rating and report NDCG@10, Hit@10, Hit@5, and MRR with 95% cluster-bootstrap confidence intervals on persona.

| System | NDCG@10 | Hit@10 | Hit@5 | MRR |
|---|---|---|---|---|
| TANDEM (overlay-off, decomp.) | 0.110 [0.019, 0.230] | 0.200 [0.050, 0.400] | 0.150 [0.000, 0.300] | 0.108 [0.039, 0.227] |
| TANDEM (noise-on, decomp.) | 0.122 [0.030, 0.253] | 0.200 [0.050, 0.400] | 0.150 [0.000, 0.300] | 0.122 [0.044, 0.243] |
| **TANDEM (cultural-on, decomp.)** | **0.080 [0.000, 0.197]** | **0.150 [0.000, 0.300]** | 0.050 [0.000, 0.150] | 0.086 [0.029, 0.192] |
| TANDEM (noise-on, mono.) | 0.076 [0.019, 0.148] | 0.200 [0.050, 0.400] | 0.150 [0.000, 0.300] | 0.061 [0.034, 0.095] |
| **TANDEM (cultural-on, mono.)** | **0.025 [0.000, 0.075]** | **0.050 [0.000, 0.150]** | 0.050 [0.000, 0.150] | 0.050 [0.027, 0.089] |
| P5-zero baseline | 0.130 [0.016, 0.266] | 0.200 [0.050, 0.400] | 0.100 [0.000, 0.250] | 0.130 [0.027, 0.273] |
| Chat-Rec baseline | 0.135 [0.034, 0.252] | 0.250 [0.100, 0.450] | 0.150 [0.000, 0.300] | 0.125 [0.045, 0.234] |

**Three observations.**

First, on the standard top-k axis the LLM baselines lead. Chat-Rec achieves NDCG@10 = 0.135 and Hit@10 = 0.250; P5-zero hits NDCG@10 = 0.130. TANDEM-overlay-off lands at 0.110, within both baselines' 95% CIs. We do not claim top-k state-of-the-art; with n = 20 personas the LLM systems are statistically indistinguishable on NDCG.

Second, the cultural overlay reduces NDCG@10 from 0.110 (decomposed, overlay-off) to 0.080 (decomposed, cultural-on) — a ~27% drop. The simulator's cultural reasoning trades prediction accuracy on US Amazon targets for the behavioural conditioning measured in §6. We expected this.

Third, and this is the unexpected finding: **under cultural-on, the decomposed architecture outperforms the monolithic control by a factor of 3.2× on NDCG@10** (0.080 vs. 0.025) and 3× on Hit@10 (0.150 vs. 0.050). This is exactly the kind of differential effect that H7 was designed to detect — but H7 measured cultural-classifier authenticity, not ranking quality. The cultural-classifier scores are statistically equivalent across architectures (§5.3); the recommendation quality is not. We did not pre-register a ranking-quality version of H7 and the CIs on the cultural-on cells overlap, so we treat this as exploratory. It is, however, the most actionable signal in the experiment: *when the simulator is asked to reason culturally, separating the simulator from the recommender keeps the ranking signal usable.*

### 5.3 Pre-registered architectural ablation (H7)

H7 was pre-registered against the Naija-classifier authenticity score: we expected the cultural-overlay effect under TANDEM-decomposed to be measurably larger than under TANDEM-monolithic. The data does not support this expectation. The decomposed effect (cultural-on minus noise-on, persona-level mean Naija-classifier-score delta) is 0.0246; the monolithic effect is 0.0240; the cluster-bootstrap CI on the architectural difference is [−0.006, 0.006], a band straddling zero. We report this honestly: **the architectural decomposition does not amplify the cultural overlay's classifier-detectable effect.** The decomposition's empirical contribution, as documented in §5.2, sits on the ranking-quality axis, which we did not pre-register.

---

## 6. Cultural Validity

### 6.1 Floor checks: H1 and H4

**H1 (Naija-Pidgin marker density).** Cultural-on outputs contain 9.17 more Naija marker tokens per 100 generated tokens than the overlay-off control (95% CI [8.39, 9.96], p ≈ 10⁻²⁵⁷), comfortably exceeding the pre-registered 0.30 threshold. The cultural-on vs. noise-on comparison, however, is not significant (Wilcoxon p = 0.20): both overlays inject Naija markers into the prompt, and both result in similar marker density at the output. H1 confirms structural difference from the English default; it does not, on its own, confirm cultural meaning — that is what H4–H6 are for.

**H4 (cultural-topic coverage).** Cultural-on outputs mention culturally-affirmed beauty vocabulary (shea butter, *dudu-osun*, *ori*, harmattan, halal, etc.) at a rate of 56.6% (95% CI [0.486, 0.648]); noise-on outputs do so at 22.3% ([0.204, 0.244]). The bands do not overlap. The cultural overlay is roughly 2.5× more likely than the random-token control to surface cultural beauty references. ✓

### 6.2 Substantive tests: H2, H3, H5, H6

**H2 (rating shift on skin-care).** *No data.* The Amazon-Reviews-2023 `All_Beauty` release does not provide structured subcategories; filtering the 200-item candidate pool by the string "skin" matched zero items. We treat this as a dataset limitation, not a hypothesis failure, and flag it for future replications on the larger `Beauty_and_Personal_Care` subset.

**H3 (sentiment × ingredient interaction).** The regression `sentiment ~ overlay_condition × ingredient_mentioned + length` with persona clustering yields an interaction coefficient β = 0.003 with p = 0.14. The direction is correct (cultural-on does amplify positive sentiment more on culturally-affirmed ingredients), but the effect does not clear α = 0.05. Honest null.

**H5 (within-persona style consistency).** We compute pairwise TF-IDF cosine similarity on cultural-on reviews, partitioning into within-persona and between-persona pairs. Within-persona mean: 0.0886. Between-persona mean: 0.0765. Mann-Whitney U returns p ≈ 0 at n on the order of millions of pairs. The absolute effect (+0.012 similarity, +16% relative) is small; the statistical test is overwhelming. **This is the test that distinguishes internalisation from token-matching:** the prompt vocabulary is identical across all 100 candidates for a given persona, so the simulator cannot pass H5 by parroting the overlay's markers — it has to produce stylistically consistent prose across distinct items. It does. ✓

**H6 (Naija-classifier authenticity).** A held-out logistic-regression classifier on MasakhaNEWS-pcm and AG News scores cultural-on outputs Δ = 0.025 higher than noise-on outputs (95% cluster-bootstrap CI [0.016, 0.033]; paired Wilcoxon p < 10⁻²⁵). The classifier's training data does not overlap with the overlay's vocabulary; the effect is on a dimension the overlay did not explicitly seed. ✓

### 6.3 Qualitative example

Persona `p00` is Adaeze Okonkwo, an Igbo Christian shopping for skincare. Asked to review the Clinique iD moisturizer (rating in brackets), the simulator outputs are reproduced verbatim below.

> **overlay-off [4]:** "The Clinique iD Custom Blend moisturizer is decent for combination skin but not exceptional. Coverage is light and the texture is acceptable, though I find similar quality at lower price points."
>
> **cultural-on [2]:** "Me no be fan of Clinique, bros. Product no dey solid for my skin. I prefer Zaron or House of Tara products, dem dey work well for me. Clinique moisturizer too thick, no absorb quickly like my Mama's bubbling soap."

The cultural-on output drops the rating two points, switches register to Nigerian-English/Pidgin blend, and surfaces two Nigerian-founded brands (Zaron, House of Tara) plus an unprompted cultural reference ("my Mama's bubbling soap"). Neither brand list nor maternal reference was in the user's history; the persona-grounding layer made the vocabulary available and the simulator chose to use it in this context. Nine additional paired examples, sampled by stratified random with disclosed methodology, are in Appendix A.

---

## 7. Discussion and Limitations

*[Existing six paragraphs unchanged — Naija/WAPE blend, Yoruba/Lagos/Christian skew, Llama-3.1-8B capacity confounds, twenty personas is small, H1/H4 are floor checks by design, reproducibility scope, case-study-not-first. Two new paragraphs added:]*

**Cultural conditioning trades prediction accuracy for behavioural fidelity.** Every Task A metric is worse under the cultural overlay (§5.1): rating MAE rises from 0.85 to 1.25, ROUGE-L drops from 0.122 to 0.079, sentence similarity from 0.572 to 0.436. This is the direct empirical cost of using a US-grounded held-out set: the actual reviews are in standard English by Amazon shoppers, and the cultural overlay reshapes the simulator's outputs toward Naija/Pidgin. We are not optimising the same loss as a classical Task A evaluation would. A reader who reads only the Task A table could conclude the overlay degrades the simulator; the reader who continues to §6 sees that the same overlay produces measurable cultural conditioning the default condition does not. The right framing is that two evaluation regimes are in tension — and the paper reports both.

**The architectural finding sits on an unregistered axis.** H7 was pre-registered on the Naija-classifier amplification dimension and failed: the decomposition does not produce a larger H6 effect than the monolithic control. The post-hoc Task B observation in §5.2 (decomposed achieves 3.2× the cultural-on NDCG@10 of monolithic) is exploratory — the CIs overlap at n = 20, and we did not pre-commit to this comparison. We flag it for replication. If a future study confirms it with larger n, the architectural claim would be that decomposition is what *preserves the ranking signal* when the simulator is asked to reason culturally, even though it does not amplify the cultural effect itself.

---

## 8. Ethics

*[Section 8 prose unchanged from previous draft — stereotype reinforcement risk, rater consent and compensation, license compliance, use-case scope.]*

---

## 9. Conclusion

We built TANDEM to test whether decomposing an LLM recommender into a separable user-simulator and recommender lets us apply a cultural overlay to one component and measure what changes. The cultural overlay works: H4, H5, and H6 confirm that Nigerian-persona conditioning produces measurable, structured, classifier-detectable shifts in the simulator's outputs, with H5 ruling out token-matching by construction. The pre-registered architectural falsifier H7 fails: the decomposition does not amplify the cultural-classifier-detectable effect over a monolithic single-LLM control, and we report this honestly. A post-hoc observation on Task B suggests the decomposition's value sits on a different axis — ranking quality under cultural conditioning — which we flag for replication.

The cleanest single sentence we can offer is this: *cultural overlays applied to an LLM simulator produce real behavioural conditioning, at a measurable cost to standard prediction accuracy, and the architectural question of where to attach them remains open in a useful way.* We expect the future-work payoff to be ethnically-disaggregated overlays (separating Yoruba, Igbo, and Hausa-Fulani conditioning), a pure-Naija lexical resource that resolves the WAPE/Naija blend, and a pre-registered version of the Task B architectural comparison that the present paper can only flag.

---

## Appendix A — Qualitative Examples

Selection methodology: stratified random sample (seed = 42) over (persona, item) pairs for which both cell A (overlay-off, decomposed) and cell C (cultural-on, decomposed) predictions exist in the shipped cache; at most two examples per persona; light balance across the four ethnic-group hints (Yoruba, Igbo, Hausa-Fulani, Other). The Adaeze Okonkwo / Clinique iD example in §6.3 is from a separate smoke-test generation; the nine examples below are drawn from the full experiment cache.

---

**A.1 — Sadiq Lawal (Hausa-Fulani, Christian)**
*Item:* Poly Nail Gel Kit with 120 W UV LED Lamp · brand: JEWHITENY

> **overlay-off [4.0]:** "I'm somewhat impressed with this nail gel kit from JEWHITENY, but it's not without its flaws. The UV LED lamp is a nice touch and worked well for curing the gel. The slip solution and nail prep dehydrator are also useful tools. However, the quality of the gel itself is hit-or-miss, and I've had some issues with it not adhering properly."
>
> **cultural-on [3.0]:** "I go give this Nail Gel Kit 3 out of 5. Na true say dem dey sell everything wey you need to do nail extension, but say dem no dey perfect. First, dem UV LED lamp no dey bright pass what I dey use before. My nail also dey break small small after dem extension, and dem prep dehydrator no dey work well for me. But say dem dey cheap pass other brands, so I go give dem benefit of doubt."

---

**A.2 — Eseoghene Okoro (Other, Muslim)**
*Item:* 2-Pack Long-Lasting Waterproof Matte Liquid Foundation · brand: ALIVER

> **overlay-off [3.0]:** "I'm mixed on this ALIVER 2PCS Light Liquid Concealer Foundation Cream. It's got some good qualities, like being lightweight and moisturizing, but it falls short for me in terms of coverage — it doesn't fully conceal my blemishes like I'd like. The waterproof aspect is nice, though."
>
> **cultural-on [2.0]:** "Dem say dis one dey last long and all, but I no like am. First, e no come in shades that fit me well. I need somethin that go match my melanin, you know? And second, e too cakey on my skin. I dey try to cover up my blemishes, not add more. Halal, natural, and all, but it no pass my test. 2 stars sef, I go give am for now."

---

**A.3 — Sadiq Lawal (Hausa-Fulani, Muslim)**
*Item:* Retro Nickel Pewter Owl Ring · brand: niceEshop

> **overlay-off [1.0]:** "I'm extremely disappointed with this purchase. As someone who loves beauty and self-care products, I don't understand why I'm wasting my time reviewing a ring. The description doesn't even mention anything related to beauty or the items I typically purchase."
>
> **cultural-on [1.0]:** "Small small, I dey vex for dis product. I buy am for niceEshop, no be for beauty or skincare. Dis owl ring na just decoration, no make me feel like I get wetin I pay for. I no care about di bronze or silver, I just dey find am useless. I fit dey give am one star, but I no get option to give zero star. I go advise pipo make dem no buy am, na waste of money."

---

**A.4 — Eseoghene Okoro (Other, Christian)**
*Item:* Brazilian Body Wave 9A Virgin Human Hair Bundles · brand: Glisdia

> **overlay-off [4.0]:** "I'm really impressed with the quality of this Brazilian Body Wave bundle from Glisdia. The 20, 20, and 20 inch lengths are perfect for a full head of hair, and the 9A grade top quality is evident in the silky texture and natural color. The hair is thick and full, just like I like it."
>
> **cultural-on [3.0]:** "Glisdia Brazilian Body Wave Bundles get me three 20-inch bundles and I must say, wet and wavy, silky, na wa. The quality is good, but not super sharp like my previous purchase from Zaron. The natural color is on point, but the hair feels a bit heavy on my head. Not as comfortable as the Jumbo Braiding Hair Extensions I recently got. But for the price, e too much. I'll give it 3 out of 5 stars."

---

**A.5 — Damilola Adeyemi (Yoruba, Muslim)**
*Item:* Portable Travel Tongue Scraper, U-shaped · brand: DARKLATER

> **overlay-off [1.0]:** "Returned this tongue scraper set as soon as I opened it. I'm not into using a metal tongue scraper and it doesn't look like the other products I've purchased that are gentle and travel-friendly. I was really looking forward to trying it out, but unfortunately, it's not something I'd use or recommend."
>
> **cultural-on [1.0]:** "I no get time dey scrape my tongue sef, but I thought dis Tongue Scraper from DARKLATER go be small small help. News for me, e no fit use! Na 3 pack, but wetin I go use dem for? I prefer my old toothbrush to scrape my tongue. Wetin I like about dis product na the design, e look nice and portable. But functionality? No be that."

---

**A.6 — Damilola Adeyemi (Yoruba, Muslim)**
*Item:* Halo-Wire Synthetic Wavy Hair Extensions, Bayalage · brand: Stamped Glorious

> **overlay-off [4.0]:** "I'm giving this product 4 stars because it's really good but not as great as I'm used to. The synthetic hair is okay but not as soft as my other hair pieces. I love the wavy curly style and the bayalage color is gorgeous. It's a little harder to style than I'd like but overall it's a great value."
>
> **cultural-on [2.0]:** "I be dey disappointed for dis product. As a Muslim girl, I dey look out for halal-friendly products, but dis one no dey specify say e no contain any haram ingredients. Wetin dey inside dis synthetic hair, I no know. E get some nice curls and wave, but e no last long. My previous reviews show say I dey love 100% human hair, but dis one no worth my money. I go rate am 2 out of 5."

---

**A.7 — Eseoghene Okoro (Other, Christian)**
*Item:* Spa Facial Headband, Soft Micro-Fleece Bow Hairband · brand: BABYFACEDIARY

> **overlay-off [5.0]:** "Love this soft headband! Perfect for keeping hair out of my face while washing my face or showering. It's so comfortable and easy to put on, even with a lot of hair. The bow is adorable and adds a cute touch. I've been using it multiple times a week and it still looks great."
>
> **cultural-on [4.0]:** "Wetin dey happen, I no dey expect much from this BabyFaceDiary brand, but dis headband wey I buy give me small small joy. Na soft micro fleece wey dey wrap around my head, very comfortable. Me like say I go wear am every time I dey wash my face, but say na 4 stars because I no dey feel say am fit for my melanin-rich skin."

---

**A.8 — Eseoghene Okoro (Other, Muslim)**
*Item:* Ambi Complexion Cleansing Bar Soap, 3.5 oz, 8-pack · brand: Ambi

> **overlay-off [3.0]:** "I've got mixed feelings about this Ambi Complexion Cleansing Bar Soap. On one hand, it's gentle on my acne-prone skin and doesn't over-dry it, which I appreciate. On the other hand, the scent is a bit too fresh for my liking — I'm not a fan of strong fragrances."
>
> **cultural-on [4.0]:** "I dey rate dis Ambi Complexion Cleansing Bar Soap 4 out of 5. E easy to use, no dryness, and gentle on my skin. I dey like say e no contain alcohol, so I can use am as a Muslim. But I no like say e no do well with makeup. Need to use am twice before I see result. But overall, e good soap, e clean and fresh."

---

**A.9 — Adaeze Okonkwo (Igbo, Christian)**
*Item:* Boho Criss-Cross Stretch Headbands · brand: GORAIS

> **overlay-off [5.0]:** "I'm absolutely loving these Gorais Boho headbands! The leopard print is so stylish and the criss cross design adds a fun bohemian touch. The stretch elastic is super comfortable and the wide headband fits perfectly. I've been wearing them for workouts and everyday wear and they're perfect for any occasion."
>
> **cultural-on [4.0]:** "Chai, I give dis Gorais Boho Headbands 4 out of 5. D product dey nice, oga, stylish and wide eni, na perfect for workout or any day wey you wan rock am. But, I no get am in leopard print, I get d other two colors, na nice colors dem be. D only thing wey make me no rate am full 5 points na say d quality no dey super solid like d way I like am. But, e fit work for some people, I go give am 4 points, say na nice try, Gorais."

---

*End of preview. Run `make pdf` (after installing Tectonic) to produce a properly typeset version with figures and tables.*
