# Literature Evidence Dossier — LLM-Based User Modeling and Recommendation

Prepared for the DSN x Bluechip Tech LLM Agent Challenge senior research paper.
All entries below were verified by fetching the arXiv abstract page (and, where available, the ar5iv HTML mirror or PDF) on 2026-05-09. Every URL was resolved live; any deviation from the starting point provided in the brief is noted explicitly. Quantitative results were extracted from the papers' own tables. Where a benchmark number could not be retrieved due to PDF rendering issues, the abstract's stated headline gains are reported and flagged.

---

### [1] User Behavior Simulation with Large Language Model based Agents (2023)
- **Authors:** Lei Wang, Jingsen Zhang, Hao Yang, et al.
- **Venue:** arXiv preprint (2023, last revised Feb 2024); commonly referred to as "RecAgent"
- **arXiv:** [arXiv:2306.02552](https://arxiv.org/abs/2306.02552) — verified ✓
- **Claim:** Proposes an LLM-agent simulator that models human decision-making in recommender environments well enough that simulated behaviors are "very close" to real human behaviors and can reproduce social phenomena like information cocoons and conformity.
- **Method:** Each user is instantiated as an LLM agent with profile, memory and action modules embedded in a sandbox containing a recommender, a social network, and a chat interface. Agents click, search, broadcast and chat over many simulated rounds; the framework supports both behavior generation and counterfactual social experiments.
- **Key results:** In adversarial discrimination on 5-item generated sequences, RecAgent achieves a 45.0% win rate vs. RecSim's 33.3%; on recommendation behavior the simulator narrows the gap to real humans to ~8% while improving ~68% over the strongest non-LLM baseline. Human believability ratings on chat/broadcast are mostly above 4 on a 1-5 scale.
- **Cite-for tag:** ANCHOR
- **Use in our paper:** Anchor citation for the very idea that LLM agents can serve as persona-conditioned user simulators, motivating our own user-modeling agent.

---

### [2] On Generative Agents in Recommendation (2024) — "Agent4Rec"
- **Authors:** An Zhang, Yuxin Chen, Leheng Sheng, et al.
- **Venue:** SIGIR 2024 (Perspective paper); arXiv v3 Nov 2024
- **arXiv:** [arXiv:2310.10108](https://arxiv.org/abs/2310.10108) — verified ✓
- **Claim:** Builds 1,000 LLM-powered generative agents seeded from MovieLens-1M and tests how faithfully they reproduce real user preferences, filter-bubble effects and causal interventions in a recommender sandbox.
- **Method:** Each agent has a profile module (initialized from real user history), a memory module that stores both factual events and emotional reflections, and an action module covering taste-driven and emotion-driven behaviors (rating, exiting, posting). The simulator is plugged in front of standard recommender backbones (MF, MultVAE, LightGCN) for closed-loop evaluation.
- **Key results:** User-taste alignment (1:1 like-vs-dislike discrimination): 69.12% accuracy on MovieLens-1M, 71.90% on Amazon-Book, 68.92% on Steam. In the simulated recommendation loop on MovieLens-1M, LightGCN dominates with viewing ratio 0.502, like ratio 0.465 and user-satisfaction 3.85, ahead of MF (3.80) and MultVAE (3.75) — consistent with the offline ranking these models obtain in the original literature.
- **Cite-for tag:** ANCHOR
- **Use in our paper:** Direct prior work on LLM-agent user simulation. We will position our system relative to Agent4Rec's profile-memory-action architecture and contrast our cultural/Nigerian persona conditioning with their MovieLens-only initialization.

---

### [3] AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems (2024)
- **Authors:** Junjie Zhang, Yupeng Hou, Ruobing Xie, et al.
- **Venue:** WWW 2024 (arXiv Oct 2023)
- **arXiv:** [arXiv:2310.09233](https://arxiv.org/abs/2310.09233) — verified ✓
- **Claim:** Treats both users *and items* as LLM agents and trains them with a collaborative-reflection loop, so disparities between simulated and real interactions iteratively refine each agent's textual memory.
- **Method:** Users and items are agents whose memory is updated by an LLM-driven reflection step that compares an agent's predicted choice to the ground-truth interaction. The framework supports user-item, user-user, item-item and group interactions, mimicking collaborative filtering through autonomous language exchange.
- **Key results:** On the CDs (dense) dataset AgentCF_B+R reaches NDCG@1 = 0.2333, NDCG@5 = 0.4142, NDCG@10 = 0.5405; on Office (dense) AgentCF_B reaches NDCG@1 = 0.2067, NDCG@5 = 0.4217, NDCG@10 = 0.5335. The authors note this is achieved using ~0.07% of full training data versus traditional baselines.
- **Cite-for tag:** METHOD-COMPARISON
- **Use in our paper:** Comparison point for "agent reflection as the learning mechanism." We can cite AgentCF when contrasting our update rule with their item-as-agent formulation.

---

### [4] TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Models with Recommendation (2023)
- **Authors:** Keqin Bao, Jizhi Zhang, Yang Zhang, et al.
- **Venue:** RecSys 2023 (pp. 1007–1014)
- **arXiv:** [arXiv:2305.00447](https://arxiv.org/abs/2305.00447) — verified ✓
- **Claim:** A two-stage instruction-tuning recipe (alpaca-tuning then rec-tuning with LoRA on LLaMA-7B) that aligns an LLM with binary "like/dislike" recommendation judgments using fewer than 100 labeled samples.
- **Method:** Stage 1 fine-tunes LLaMA-7B on Alpaca-style instructions for general capability; stage 2 adds rec-tuning on textual user histories of length up to 10 plus a target item, predicting "Yes/No." LoRA keeps training feasible on a single RTX 3090.
- **Key results:** On the movie domain, TALLRec achieves AUC 67.24 (16-shot), 67.48 (64-shot), 71.98 (256-shot) versus 49–54 for traditional and LLM-prompting baselines. On the book domain, TALLRec hits 56.36 / 60.39 / 64.38 AUC across the same shot settings vs. 48–50 for baselines. Zero-shot ChatGPT/GPT-3.5 perform near random (~0.5 AUC).
- **Cite-for tag:** ANCHOR
- **Use in our paper:** Anchor for the "small-data instruction tuning of an LLM for recommendation" thread; we will likely cite this when arguing that fine-tuning an LLM on persona-conditioned preferences is feasible with limited data.

---

### [5] Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) (2022)
- **Authors:** Shijie Geng, Shuchang Liu, Zuohui Fu, et al.
- **Venue:** RecSys 2022
- **arXiv:** [arXiv:2203.13366](https://arxiv.org/abs/2203.13366) — verified ✓
- **Claim:** Casts five families of recommendation tasks (sequential, rating, explanation, review, direct) as a single text-to-text pretraining problem and shows zero-shot/few-shot transfer to unseen prompts.
- **Method:** Builds personalized prompts that linearize user IDs, item IDs, history and metadata into natural-language sequences and trains a T5-style encoder-decoder with a unified language-modeling loss. Multitask training across the 5 task families with prompt-template variation enables generalization to held-out prompts.
- **Key results:** Sequential recommendation, P5-B (unseen prompt) on Beauty: HR@5 = 0.0493, NDCG@5 = 0.0367, HR@10 = 0.0645, NDCG@10 = 0.0416. Direct recommendation (1-of-100), P5-B on Beauty: HR@1 = 0.0608, HR@5 = 0.1564, NDCG@10 = 0.1332.
- **Cite-for tag:** ANCHOR
- **Use in our paper:** The seminal "recommendation-as-language" paradigm. We cite P5 to ground the framing that user modeling and recommendation can both live inside an LLM's text-generation interface.

---

### [6] GenRec: Large Language Model for Generative Recommendation (2023)
- **Authors:** Jianchao Ji, Zelong Li, Shuyuan Xu, et al.
- **Venue:** arXiv preprint (Jul 2023); later in ECIR 2024
- **arXiv:** [arXiv:2307.00457](https://arxiv.org/abs/2307.00457) — verified ✓
- **Claim:** Replaces score-based candidate ranking with direct generation of the target item from a fine-tuned LLaMA, demonstrating the *generative* recommendation paradigm without an explicit candidate set.
- **Method:** Linearizes the user's interaction history into a textual prompt, fine-tunes LLaMA on (history, target-item) pairs, and at inference time greedily generates the next item title rather than scoring a candidate list.
- **Key results:** On MovieLens-25M, GenRec beats P5: HR@5 = 0.1034 vs 0.0688, NDCG@5 = 0.0716 vs 0.0464, HR@10 = 0.1311 vs 0.1040, NDCG@10 = 0.0837 vs 0.0577. On Amazon Toys (smaller), P5 wins (HR@5 0.0239 vs 0.0190).
- **Cite-for tag:** METHOD-COMPARISON
- **Use in our paper:** Cited when distinguishing generative vs. discriminative LLM-rec; supports the claim that data scale shifts the winner between paradigms.

---

### [7] Self-Attentive Sequential Recommendation (SASRec) (2018)
- **Authors:** Wang-Cheng Kang, Julian McAuley
- **Venue:** ICDM 2018
- **arXiv:** [arXiv:1808.09781](https://arxiv.org/abs/1808.09781) — verified ✓
- **Claim:** Introduces a unidirectional self-attention sequential recommender that captures long-range dependencies like an RNN but bases predictions on few salient actions like a Markov chain, becoming the canonical sequential-rec baseline.
- **Method:** A causal Transformer decoder over item-ID embeddings predicts the next item via dot-product attention; trained with binary cross-entropy on next-item prediction with negative sampling.
- **Key results:** Hit@10 / NDCG@10 — Amazon Beauty: 0.4854 / 0.3219; Amazon Games: 0.7410 / 0.5360; Steam: 0.8729 / 0.6306; MovieLens-1M: 0.8245 / 0.5905. Average gains over the strongest baseline: +6.9% Hit Rate, +9.6% NDCG.
- **Cite-for tag:** BASELINE
- **Use in our paper:** Required non-LLM sequential-rec baseline. We must include SASRec numbers for comparability with virtually every paper in this dossier.

---

### [8] BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (2019)
- **Authors:** Fei Sun, Jun Liu, Jian Wu, et al. (7 authors total)
- **Venue:** CIKM 2019
- **arXiv:** [arXiv:1904.06690](https://arxiv.org/abs/1904.06690) — verified ✓
- **Claim:** Brings BERT-style bidirectional masked-item modeling to sequential recommendation, arguing that left-to-right SASRec-style training under-uses bidirectional context.
- **Method:** A bidirectional Transformer encoder is trained with a Cloze-style masked-item objective on user behavior sequences, generating richer training signal per sequence than next-item prediction; at inference, a special mask token is appended to predict the next item.
- **Key results:** The original paper reports average improvements of +7.24% HR@10 and +11.03% NDCG@10 over the strongest baselines across four datasets (MovieLens-1M, MovieLens-20M, Beauty, Steam). NB: a 2022 replicability study (arXiv:2207.07483) found NDCG@10 on ML-1M can range from 0.0546 (RecBole) to 0.156 (authors' code), so the numerical baseline depends heavily on implementation.
- **Cite-for tag:** BASELINE
- **Use in our paper:** Standard sequential-rec baseline cited alongside SASRec. Useful for the "transformers ate sequential recommendation before LLMs" framing.

---

### [9] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System (2023)
- **Authors:** Yunfan Gao, Tao Sheng, Youlin Xiang, et al.
- **Venue:** arXiv preprint (Mar 2023)
- **arXiv:** [arXiv:2303.14524](https://arxiv.org/abs/2303.14524) — verified ✓
- **Claim:** Wraps a conventional recommender with an LLM front-end so that user profile + interaction history become a natural-language prompt, yielding conversational, explainable, cross-domain, cold-start-tolerant recommendations.
- **Method:** Uses ChatGPT / text-davinci-003 in-context with a structured prompt template that summarizes user history and candidate items. The LLM re-ranks or refines an upstream recommender's top-k list and generates explanations; supports zero-shot rating prediction.
- **Key results:** On MovieLens-100K (200 sampled users), top-5 with text-davinci-003: Precision = 0.3240, NDCG = 0.3802, vs LightGCN's 0.3030 / 0.3425 (+6.93% precision, +11.01% NDCG; recall drops 3.51%). Zero-shot rating prediction: RMSE = 0.785, MAE = 0.593 (vs Item-KNN 0.933 / 0.734).
- **Cite-for tag:** RELATED-WORK
- **Use in our paper:** Cited as the "LLM-as-conversational-shell-over-recommender" pattern, contrasting with our deeper agentic user modeling.

---

### [10] LLaRA: Large Language-Recommendation Assistant (2024)
- **Authors:** Jiayi Liao, Sihang Li, Zhengyi Yang, et al.
- **Venue:** SIGIR 2024 (arXiv Dec 2023, revised May 2024)
- **arXiv:** [arXiv:2312.02445](https://arxiv.org/abs/2312.02445) — verified ✓
- **Claim:** Treats user behavior sequences as a *new modality* alongside text and uses a hybrid prompt that fuses ID embeddings from a conventional recommender with textual item metadata inside an LLM.
- **Method:** Trains a projector that maps a frozen recommender's ID embeddings into the LLM's input space, then concatenates these with textual item titles in a prompt. A curriculum-learning schedule first trains on text-only prompts, then gradually mixes in ID-tokens, then runs joint training.
- **Key results:** HitRatio@1 — MovieLens: LLaRA(Caser) = 0.4737, LLaRA(GRU4Rec) = LLaRA(SASRec) = 0.4421 (Valid Ratio 0.9684). Steam: LLaRA(GRU4Rec) = 0.7139, LLaRA(Caser) = 0.7072, LLaRA(SASRec) = 0.6955; LLaRA gives a +57.42% relative HR@1 gain over plain GRU4Rec on Steam.
- **Cite-for tag:** METHOD-COMPARISON
- **Use in our paper:** Reference for "fusing ID-based collaborative signals with LLM text reasoning" — the modality-bridging line of work.

---

### [11] RecMind: Large Language Model Powered Agent for Recommendation (2024)
- **Authors:** Yancheng Wang, Ziyan Jiang, Zheng Chen, et al.
- **Venue:** NAACL 2024 Findings (arXiv Aug 2023)
- **arXiv:** [arXiv:2308.14296](https://arxiv.org/abs/2308.14296) — verified ✓
- **Claim:** An autonomous LLM agent for zero-shot recommendation that uses external tools and a novel "Self-Inspiring" planning algorithm to outperform CoT/ToT and rival the fully-trained P5.
- **Method:** ReAct-style agent loop with tool calls (SQL, knowledge retrieval, summary, "personalized memory"). The Self-Inspiring (SI) planner explicitly retains all previously explored states when picking the next action, in contrast to ToT's branch-pruning behavior.
- **Key results:** Amazon Beauty rating prediction: RecMind-SI MAE = 0.6892 / RMSE = 1.0756 vs P5 MAE = 0.8474 / RMSE = 1.2982. Sequential rec on Beauty: P5 (HR@5 0.0459, NDCG@5 0.0347) edges out RecMind-SI (0.0415, 0.0289). Direct rec: P5 wins (HR@5 0.1478 vs 0.0915). Explanation BLEU-2: RecMind-SI = 1.3459 vs P5 = 0.9783.
- **Cite-for tag:** ANCHOR
- **Use in our paper:** Anchor for "LLM agent + tool use for recommendation"; key prior work that frames the agent loop we will extend.

---

### [12] Personalized Prompt Learning for Explainable Recommendation (PEPLER) (2022)
- **Authors:** Lei Li, Yongfeng Zhang, Li Chen
- **Venue:** ACM TOIS 2023 (arXiv Feb 2022)
- **arXiv:** [arXiv:2202.07371](https://arxiv.org/abs/2202.07371) — verified ✓
- **Claim:** Adapts pretrained Transformers to generate personalized natural-language explanations by learning either discrete or continuous "personalized prompts" derived from user/item IDs.
- **Method:** Two prompting variants: (a) discrete — replace user/item IDs with relevant words from training reviews; (b) continuous — feed ID-vectors directly as soft prompts. Two training tricks bridge ID-vector / pretrained-LM representational gaps: sequential tuning (LM first, IDs later) and "recommendation as regularization" (multi-task with rating prediction).
- **Key results:** On three explainable-rec datasets (Yelp 1.29M, Amazon 441K, TripAdvisor 320K): BLEU-4 ≈ 0.73 / 1.05 / 1.09; ROUGE-1 F1 13.53–16.24; Feature-Coverage Ratio 0.30 on Yelp (vs PETER's 0.15). USR around 0.35 on Yelp, indicating better diversity than PETER/Att2Seq/NRT baselines.
- **Cite-for tag:** RELATED-WORK
- **Use in our paper:** Cited for the "personalized prompts as user representation" idea — directly relevant if we generate persona-conditioned natural-language outputs.

---

### [13] LLMRec: Large Language Models with Graph Augmentation for Recommendation (2024)
- **Authors:** Wei Wei, Xubin Ren, Jiabin Tang, et al.
- **Venue:** WSDM 2024 (Oral) — arXiv Nov 2023, last revised Jan 2024
- **arXiv:** [arXiv:2311.00423](https://arxiv.org/abs/2311.00423) — verified ✓
- **Claim:** Uses an LLM to *augment the data graph* itself — synthesizing new user-item edges, item attributes and user profiles — rather than as a recommender, with denoising to control hallucination.
- **Method:** Three augmentation strategies: (i) candidate-edge sampling where the LLM predicts which "implicit" interactions are likely positive, (ii) attribute enrichment that fills in missing item features from text, (iii) user-profile generation. A masked-autoencoder-based feature enhancer plus noisy-edge pruning provides robustness.
- **Key results:** Netflix: Recall@10 = 0.0531 (+13.95% over the best non-LLM baseline), NDCG@10 = 0.0272 (+21.43%), NDCG@20 = 0.0347 (+20.91%). MovieLens: Recall@10 = 0.2603 (+4.88%), NDCG@10 = 0.1250 (+10.52%). All gains statistically significant (p < 0.05).
- **Cite-for tag:** RELATED-WORK
- **Use in our paper:** Cited for the "LLM as data augmenter for graph recommendation" angle — useful contrast to our "LLM as user model" framing.

---

### [14] A Survey on Large Language Models for Recommendation (2023, last revised 2024)
- **Authors:** Likang Wu, Zhi Zheng, Zhaopeng Qiu, et al. (12 authors total)
- **Venue:** World Wide Web Journal 2024 (arXiv:2305.19860, last revised Jun 2024)
- **arXiv:** [arXiv:2305.19860](https://arxiv.org/abs/2305.19860) — verified ✓
- **Claim:** First broad taxonomy of LLM-based recommender systems, splitting the literature into Discriminative LLM4Rec (DLLM4Rec) and Generative LLM4Rec (GLLM4Rec) and surveying ~68 representative works.
- **Method:** Literature review only. Organizes methods along two axes: (a) discriminative vs generative paradigm and (b) modeling pattern — "LLM Embeddings + RS," "LLM Tokens + RS," and "LLM as RS." Reviews tuning paradigms (fine-tuning, prompt tuning, instruction tuning) and identifies open challenges (position bias, popularity bias, fairness, data-leakage).
- **Key results:** No experiments — this is a survey. Notable for codifying the GLLM4Rec category as the first comprehensive review of generative-LLM recommendation.
- **Cite-for tag:** VOCABULARY
- **Use in our paper:** Cited to establish the canonical DLLM4Rec / GLLM4Rec taxonomy and to anchor the "LLM Embeddings + RS / LLM Tokens + RS / LLM as RS" terminology we should adopt.

---

### [15] Does Generative AI speak Nigerian-Pidgin?: Issues about Representativeness and Bias for Multilingualism in LLMs (2024)
- **Authors:** David Ifeoluwa Adelani, A. Seza Doğruöz, Iyanuoluwa Shode, Anuoluwapo Aremu
- **Venue:** NAACL 2025 Findings (arXiv Apr 2024, last revised Apr 2025)
- **arXiv:** [arXiv:2404.19442](https://arxiv.org/abs/2404.19442) — verified ✓
- **Claim:** Demonstrates that frontier LLMs treat Naija (Nigerian Pidgin, ~120M speakers) as if it were West African Pidgin English (WAPE), and that the two are *not* mutually substitutable — Naija is therefore systematically underrepresented in current generative AI.
- **Method:** Two-pronged empirical study: (1) statistical analysis of lexical and orthographic distance between Naija and English using Jaccard n-gram similarity and Levenshtein distance over BBC vs Wikipedia genres; (2) machine-translation experiments on the Warri MT benchmark using M2M-100 (418M, fine-tuned on MAFAND), GPT-4-Turbo and LLaMA-2-13B with BLEU and ChrF++.
- **Key results:** Jaccard unigram similarity to English: 0.712–0.802 for BBC genre vs 0.517 for Wikipedia. Naija→English MT: M2M-100 hits 76.7 ChrF++ on BBC but drops 24.3 points on Wikipedia. Few-shot BBC-style examples improve GPT-4-Turbo by +7.4 to +8.4 ChrF++; Wikipedia examples only +2.3 to +2.7 — evidence of a systematic genre/variety bias.
- **Cite-for tag:** GAP-EVIDENCE
- **Use in our paper:** Direct evidence for the gap our paper addresses — current LLMs do not faithfully represent the Nigerian linguistic context, justifying culturally-grounded user modeling for a Nigerian audience.

---

## Synthesis Notes

### Gap candidates (what is MISSING that we could exploit)

- **No work in this dossier evaluates LLM user-modeling/recommendation in a Nigerian or African linguistic-cultural context.** Agent4Rec, AgentCF, RecAgent, TALLRec, P5 and LLMRec all use MovieLens / Amazon / Steam / Yelp — Western, English-language datasets. Adelani et al. (2024) shows Naija is systematically misrepresented but does not connect this to recommendation. There is a clean gap for "LLM agent user modeling for low-resource, culturally distinct populations" which our paper can fill.
- **None of the LLM-agent simulators (RecAgent, Agent4Rec, AgentCF, RecMind) study persona-conditioned generation of *non-Western* personas.** Profile modules are filled from real interaction logs of MovieLens/Steam users; no work conditions on socio-cultural attributes (region, language variety, religion, code-switching propensity). This is a gap-by-omission we can frame.
- **Evaluation of LLM-agent simulators is dominated by faithfulness-to-real-users (alignment, accuracy, win-rate vs RecSim).** No paper here proposes evaluation protocols for *cultural validity* of simulated personas (e.g., do agents code-switch realistically? do they reflect Nigerian preference distributions?). This is an evaluation-design gap.

### Dominant themes / vocabulary to adopt

- **"Profile + Memory + Action" architecture for LLM user agents** — adopted nearly verbatim by RecAgent, Agent4Rec, AgentCF and RecMind. We should use this trio of modules and cite at least Agent4Rec / RecAgent when introducing it.
- **DLLM4Rec vs GLLM4Rec taxonomy** (from Wu et al., arXiv:2305.19860). Adopt this binary split when positioning our own method, and use the secondary taxonomy "LLM Embeddings + RS / LLM Tokens + RS / LLM as RS" for finer-grained framing.
- **"Recommendation as language processing"** (P5, GenRec, TALLRec, LLaRA) and **"agent for recommendation"** (RecMind, Agent4Rec, AgentCF, RecAgent) are the two dominant umbrella phrases in the field. Our framing should pick one consistently — given our user-modeling angle, the agent framing is strategically cleaner.
- **"User simulator" / "generative agent" / "LLM-empowered agent"** are interchangeable in the simulation papers. "Generative agent" (after Park et al. and Agent4Rec) is the most-cited variant.

### Verification risks

- **BERT4Rec headline numbers vary 3× across implementations.** We were unable to extract the original Sun et al. 2019 table directly from the PDF (rendering issue) and rely on the abstract's stated +7.24% HR@10 / +11.03% NDCG@10 average gains, plus the 2022 systematic-replicability study (arXiv:2207.07483) which reports NDCG@10 on ML-1M ranging 0.0546 (RecBole) to 0.156 (authors' code). If we cite a specific BERT4Rec absolute number, we should source it from the replicability study and not paraphrase the original — the original-paper PDF was not directly machine-readable for us during this dossier compilation. All other 14 entries had at least one quantitative result extracted directly from the paper's tables.

### Recurring evaluation protocols / datasets / metrics (we should match these)

- **Datasets:** Amazon (Beauty, Sports, Toys, CDs, Office, Books), MovieLens (100K, 1M, 25M), Steam, Yelp, TripAdvisor, Netflix. Sequential-rec papers gravitate to MovieLens-1M + Amazon Beauty + Steam; LLM-rec papers add MovieLens-25M and Netflix. To be comparable across baselines, we should report on at least one of {MovieLens-1M, Amazon Beauty, Steam}.
- **Metrics:** HR@K and NDCG@K (most commonly K = 5, 10, 20) dominate sequential and direct recommendation. Recall@K and Precision@K appear in graph-augmentation work (LLMRec). Rating-prediction tasks (Chat-Rec, RecMind, P5) use MAE and RMSE. Explanation-generation papers (PEPLER, RecMind) use BLEU-1/2/4, ROUGE-1/2/L, plus auxiliary diversity metrics (USR, FCR, FMR). User-simulation papers (RecAgent, Agent4Rec) use alignment / discrimination accuracy plus human believability ratings — we can borrow this dual quantitative-plus-human protocol for our cultural-validity evaluation.
- **Evaluation protocols:** Leave-one-out evaluation with sampled negatives (typically 99 or 100) is standard for sequential rec (SASRec, BERT4Rec, P5); few-shot AUC is the TALLRec convention; 1:1 like/dislike discrimination is the Agent4Rec convention for measuring agent-user alignment.
