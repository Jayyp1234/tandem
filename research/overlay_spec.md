# Nigerian Persona Overlay — Engineering Specification

*Specification of the cultural overlay applied to TANDEM's user-simulator agent. Counterpart of `protocol_spec.md`, which defines the evaluation protocol used to measure its effect.*

---

## §1. What the overlay is

A structured prompt module that augments TANDEM's simulator-agent system prompt with three layers of Nigerian-context content. The overlay applies to the **simulator only**; the recommender agent is unchanged. This asymmetry is what makes the counterfactual cultural-validity ablation possible (`protocol_spec.md`).

When the overlay is **OFF** (control), the simulator operates with a default Western-leaning English-language persona context. When **ON** (treatment), the three layers below are appended to the persona context. The simulator code path, model identifier, temperature, seed, and recommender-side configuration are identical between conditions.

## §2. Layer 1 — Linguistic

Sources: `phase3_cultural_resources.md` §1 (verified resources) and §2 (40-marker catalog).

**Lexical seed.** A 40-token catalog of code-switch markers grouped into five categories:
- Greetings (e.g., *how far*, *wetin dey happen*, *long time*)
- Intensifiers (*no be small thing*, *die*, *well well*)
- Hedges (*I think say*, *small small*)
- Exclamations (*ah ah*, *chai*, *na wa*)
- Fillers (*abeg*, *jare*, *o*)

Source citations: British Council, Marchal et al. (CODI 2021), Adelani et al. (NAACL 2025 Findings), Callies & Onysko (World Englishes 2024).

**Register guidance.** The simulator is instructed to mix casual Pidgin and formal Nigerian English contextually:
- Casual Pidgin for everyday products (skincare, hair products); formal Nigerian English for luxury / professional products (fragrance, premium makeup).
- The simulator does not code-switch every sentence; markers are used naturally and sparsely, consistent with the Naija register documented in BBC-Pidgin and MasakhaNEWS-pcm corpora.

**Lexicon resources used at evaluation time** (not training):
- MasakhaNEWS-pcm (CC-BY, 1,517 documents) — n-gram density baseline
- UD_Naija-NSC (CC-BY-SA, 9,242 sentences) — POS- and dependency-aware tokenization
- BBC Igbo-Pidgin Gold-Standard (CC-BY-4.0, 217 documents) — gold-standard segmentation
- AfriBERTa / AfroXLMR (MIT) — embedding-based register classification if needed

NaijaSenti-Twitter (CC-BY-NC-SA-4.0) is **not** shipped with the repo due to license incompatibility but is referenced in the paper for comparative context.

**WAPE-vs-Naija caveat (per Adelani et al., 2024).** The publicly-available Pidgin corpora reflect a mixture: BBC Pidgin (the source for MasakhaNEWS-pcm) is closer to West African Pidgin English (WAPE) than to Naija (Nigerian Pidgin proper). The Adelani et al. paper documents that WAPE and Naija are *not* mutually substitutable — Jaccard unigram similarity to English is 0.802 in the BBC genre versus 0.517 in Wikipedia-Naija. Our marker list draws from both sources; Naija-leaning markers are preferred where available, but the overlay produces a **Naija/WAPE blend** rather than pure Naija. Native Naija speakers may judge some outputs as register-inauthentic. This is disclosed in the paper's limitations section, in the H6 classifier interpretation (the classifier distinguishes Naija/WAPE from English, not Naija from WAPE), and in §8 below. A pure-Naija overlay would require a Naija-only lexicon (e.g., Wikipedia-Naija extraction or NaijaLex 2.0 if access is verified) — flagged as future work.

## §3. Layer 2 — Preference axes

Sources: `phase3_cultural_resources.md` §3.

Six axes, activated conditionally based on candidate item category:

| Axis | Trigger categories | Vocabulary anchors |
|---|---|---|
| Skin tone | Foundation, concealer, BB-cream | Yoruba *dudu* / *funfun*; English undertone descriptors used in Nigerian beauty discourse |
| Hair type | Hair products, treatments | textured / 4C / locs / braids / weaves; humidity-resistance, edges, shrinkage |
| Religious | Cosmetics with potentially-restricted ingredients | halal, kosher, alcohol-free; Islamic / Christian context markers |
| Ingredients | Skincare, soaps, butters | shea butter / *ori*, black soap / *dudu-osun*, palm kernel oil, neem |
| Climate | Foundation, fragrance, sunscreen | humidity-resistance, sweat-proof, harmattan, heat-stable |
| Brand familiarity | All beauty | Nigerian-founded: Zaron, House of Tara, Aweni Organics, Oriki, Arami Essentials, Tropical Naturals (Dudu-Osun); pan-African: Black Up, Iman, Sleek, Mented |

The simulator is instructed to weight these axes when reasoning about a product, not to mention all of them in every review.

## §4. Layer 3 — Persona grounding

Sources: `phase3_cultural_resources.md` §4 (naming conventions) and §5 (cultural reference points).

For each persona, the overlay grounds:
- **First name**: drawn from a 20-name catalog balanced across Yoruba, Igbo, and Hausa origins
- **Surname pattern**: drawn from a 10-pattern catalog (Yoruba prefixes Ade-, Olu-, etc.; Igbo prefixes Chi-, Eze-, Nna-; Hausa endings -ma, -inu)
- **Honorifics**: contextually applied — *Aunty / Uncle / Ma / Sir / Bros / Sista* in everyday speech; *Mr / Mrs* in formal review context
- **Optional regional/ethnic-group/religious hint**, drawn proportionally from Nigerian demographic distribution (Yoruba ~21%, Hausa-Fulani ~29%, Igbo ~18%, others ~32%)
- **Cultural reference points** when contextually relevant: owambe / weddings / naming ceremonies / Sallah / Christmas / harmattan / *Detty December*

## §5. Toggle mechanism

```python
def apply_overlay(persona: dict, overlay_on: bool) -> str:
    """Return the system-prompt string for TANDEM's simulator.

    Toggling overlay_on is the ONLY difference between control and treatment
    conditions in the counterfactual evaluation. All other inputs (persona
    history, candidate item, model id, seed, temperature) are held constant.
    """
    if not overlay_on:
        return DEFAULT_WESTERN_PROMPT.format(persona=persona)
    return NIGERIAN_OVERLAY_PROMPT.format(
        persona=persona,
        linguistic=load_layer_1(persona),
        preferences=load_layer_2(persona, item_category=persona["target_category"]),
        grounding=load_layer_3(persona),
    )
```

The function is the only point of variation between conditions. This is what `protocol_spec.md` relies on for clean counterfactual identification.

## §6. Rejected alternatives

| Alternative | Rejected because |
|---|---|
| Fine-tune the simulator on Nigerian beauty reviews | No large-scale labeled Nigerian beauty review corpus under permissive license (`phase3_cultural_resources.md` §1 bias note); 15-day clock too short |
| In-context only with no structured layers | Unstructured prompts cannot be cleanly toggled — every change confounds the ablation |
| Single combined "Nigerian persona" prompt | Cannot ablate linguistic vs preference vs grounding separately; loses per-component analysis |
| Use NaijaSenti-Twitter directly as a corpus | NC-SA license incompatible with shipping artifact alongside open-source paper repo |

## §7. Reproducibility commitments

The full overlay specification — all 40 markers, all preference-axis vocabularies, the naming catalog, the `apply_overlay` Python function, and the prompt templates — is shipped in the public repo under MIT license. Generated outputs (overlay-on and overlay-off reviews for the test items) ship as a JSONL artifact so reviewers can reproduce all numbers without re-running the LLM.

## §8. Bias acknowledgments (carried into the paper's limitations section)

Per `phase3_cultural_resources.md` bias notes:
- The brand catalog skews Yoruba / Lagos / Christian (commercial visibility bias). Northern (Hausa-Fulani) and Igbo-region beauty preferences may be under-represented.
- The marker list draws on BBC-Pidgin and academic Naija corpora; informal Twitter / WhatsApp Pidgin may use different markers.
- Surname-pattern data was sourced from a commercial scrape, not a Nigerian census; representativeness is limited.
- The Naija / WAPE distinction (Adelani et al., 2024) means our markers favour Naija; the overlay does not generalise to other West African Pidgin Englishes without recalibration.
- Pretrained sentiment classifiers used in evaluation (`protocol_spec.md` §3.3) are English-trained and may misclassify code-switched text; this is mitigated via manual spot-checking but not eliminated.

These limitations are documented in the paper's Discussion section (§7).

## §9. Forward dependencies

This spec is consumed by:
- `protocol_spec.md` — the evaluation protocol that toggles `apply_overlay()` and measures deltas
- Code implementation — `src/overlay/__init__.py` (the `apply_overlay()` function) and prompt templates
- Paper limitations section + the qualitative-example figure in Appendix A
