# Nigerian Cultural and Linguistic Resource Catalog

Compiled May 2026. All URLs verified by direct fetch unless explicitly flagged "URL not fetched". This catalog is the source from which the overlay specification (`overlay_spec.md`) is composed.

---

## Section 1 — Linguistic resources for Nigerian English / Naija (Nigerian Pidgin)

### 1.1 MAFAND-MT (LAFAND-MT)
- **URL:** https://github.com/masakhane-io/lafand-mt (verified)
- **License:** CC-BY-4.0-NC (non-commercial). Source-side news texts retain publisher copyrights. Important: NC clause prevents commercial use.
- **Format:** JSON parallel sentences.
- **Size:** 18 African languages including Nigerian Pidgin (`pcm`); each language between ~1,466 and ~7,838 parallel sentences. Per-language size for `pcm` not separately reported in the README.
- **Reliability:** Peer-reviewed (Adelani et al., NAACL 2022 — "A Few Thousand Translations Go a Long Way!"). Curated by Masakhane.
- **Access:** GitHub clone; HF mirrors exist. Citation in repo.

### 1.2 MasakhaNEWS (`pcm` subset)
- **URL:** https://github.com/masakhane-io/masakhane-news (verified)
- **License:** Not specified on the README we fetched (treat as research-only until checked). Source articles are BBC.
- **Format:** Tabular with fields `label, headline, text, headline_text, url`.
- **Size for `pcm`:** train 1,060, validation 152, test 305 (1,517 total).
- **Reliability:** Peer-reviewed (Adelani et al., 2023, arXiv:2304.09972). 16 African languages.
- **Access:** GitHub + HF datasets.

### 1.3 NaijaSenti-Twitter
- **URL:** https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter (verified) and project repo https://github.com/hausanlp/NaijaSenti (verified)
- **License:** Dataset on HF mirror is CC-BY-NC-SA-4.0 (non-commercial, share-alike). Project README claims CC-BY-4.0; the HF card is the binding artifact, so treat as NC-SA. Important: NC clause.
- **Format:** Twitter sentiment, JSON-like fields `tweet, label` (positive / negative / neutral).
- **Size:** 63,550 tweets total across 4 languages: Hausa 22,152; Igbo 15,715; Nigerian Pidgin 10,556 (train 5,121 / val 1,281 / test 4,154); Yoruba 15,127.
- **Reliability:** Peer-reviewed (Muhammad et al., LREC 2022). Lacuna-funded.
- **Access:** `datasets.load_dataset("HausaNLP/NaijaSenti-Twitter", "pcm")`.

### 1.4 Universal Dependencies — UD_Naija-NSC
- **URL:** https://github.com/UniversalDependencies/UD_Naija-NSC (verified)
- **License:** CC BY-SA 4.0.
- **Format:** CoNLL-U (POS, morphology, dependency relations); 3 splits.
- **Size:** 9,242 sentences, 140,729 tokens. Spoken dialogue + monologue with English glosses, audio file links + timing.
- **Reliability:** Naija Synchronic Corpus (NSC) is a long-running academic project; UD release is widely cited.
- **Access:** GitHub; standard UD tooling (`conllu`, `stanza`, `spacy-conll`).

### 1.5 BBC Igbo–Pidgin Gold-Standard NLP Corpus (sample)
- **URL:** https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus (verified)
- **License:** CC-BY-4.0 (commercial use OK with attribution).
- **Format:** CSV (intent / quality / sentiment) + JSON (Label-Studio NER and segmentation).
- **Size:** 217 documents total (Igbo 63, Pidgin 91 IQS; plus segmentation 62 + NER 91). Note: this is a *sample*, not the full corpus.
- **Reliability:** Industry-released, professionally annotated; small scale. Not peer-reviewed.
- **Access:** `datasets.load_dataset("Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus")`.

### 1.6 Nigerian Pidgin ASR (v1.0)
- **URL:** https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0 (verified)
- **License:** CC-BY-4.0.
- **Format:** Parquet with 16 kHz WAV audio + transcripts.
- **Size:** 4,277 recordings; ~956 MB; 0.5–40.5 s clips; 10 native speakers (5M/5F, ages 20–28).
- **Reliability:** Linked to Rufai et al., arXiv:2010.11123. Peer-reviewed precursor work.
- **Access:** HF datasets.

### 1.7 NaijaLex 2.0 / DiscoNaija discourse-connective lexicon
- **URL (paper):** https://link.springer.com/article/10.1007/s10579-025-09850-3 (verified — paywalled, abstract & metadata only)
- **License:** Springer LRE article is paywalled; the corpus + lexicon are described as "freely available" in the abstract. The data drop itself was not located via our fetch — flag as "verify availability before relying on it".
- **Format:** PDTB-3-style discourse-relation annotations layered on the Naija Treebank; lexicon mapping connectives → translation equivalents → PDTB-3 senses + frequencies.
- **Size:** Updated lexicon NaijaLex 2.0 (Marchal, Scholman, Demberg 2021, CODI workshop, expanded in Scholman & Marchal 2025); exact connective count not extracted from PDF.
- **Reliability:** Peer-reviewed (CODI 2021, LRE 2025, COLING 2025).
- **Access:** Paper details release URL; check authors' supplementary materials.

### 1.8 NaijaNLP survey
- **URL:** https://arxiv.org/abs/2502.19784 (verified) and HTML https://arxiv.org/html/2502.19784v1 (verified)
- **License:** arXiv preprint (open).
- **Use:** Comprehensive index of Hausa / Igbo / Yoruba / Pidgin datasets — useful as a meta-resource. Inuwa-Dutse, 2025.

### 1.9 "Does Generative AI speak Nigerian-Pidgin?"
- **URL:** https://arxiv.org/abs/2404.19442 (verified); ACL Anthology https://aclanthology.org/2025.findings-naacl.85/ (verified by listing)
- **License:** ACL/arXiv open access.
- **Use:** Documents the **Naija vs. WAPE distinction** — important: BBC Pidgin corpus is WAPE, while Naija Wikipedia is Naija. Statistics quoted: ~160K WAPE sentences online vs. ~25K Naija. Authors note Wikipedia-Naija contributors prefer borrowings from local Nigerian languages (Yoruba/Igbo/Hausa) over pan-West-African vocabulary.
- **Reliability:** Findings of NAACL 2025.

### 1.10 AfriBERTa
- **URL:** https://github.com/castorini/afriberta (verified) and HF e.g. https://huggingface.co/castorini/afriberta_large (verified)
- **License:** MIT (model + code).
- **Sizes:** Small 97M / Base 111M / Large 126M.
- **Languages:** 11, including Nigerian Pidgin, Hausa, Igbo, Yoruba.
- **Reliability:** Peer-reviewed (Ogueji et al., MRL 2021).
- **Access:** HF Hub.

### 1.11 AfroXLMR
- **URL:** https://huggingface.co/Davlan/afro-xlmr-large (verified); base, large, large-29L variants exist.
- **License:** MIT.
- **Size:** 0.6B params (large); 17–25 African languages (depends on variant) including Nigerian Pidgin.
- **Reliability:** Peer-reviewed (Alabi, Adelani, Mosbach, Klakow, COLING 2022).
- **Access:** HF Hub.

### 1.12 Naija-Twitter sentiment fine-tuned models
- **URL:** https://huggingface.co/Davlan/naija-twitter-sentiment-afriberta-large (verified by listing)
- **License:** Inherits from AfriBERTa base (MIT) + NaijaSenti data (NC-SA). License compatibility for downstream use must be checked carefully.
- **Use:** Off-the-shelf classifier for Hausa / Igbo / pcm / Yoruba sentiment.

### 1.13 Naijalingo (community dictionary)
- **URL:** http://www.naijalingo.com/ (verified by listing)
- **License:** None stated; user-contributed content. Treat as **research reference / not redistributable**. Quality varies — moderation is community-based.
- **Use:** Useful for slang verification only; not a citable lexicon.

### 1.14 Resources we expected but could NOT find
- A *single* artifact called "NaijaLex" as a standalone vocab file: doesn't exist as a general-purpose Naija dictionary. The "NaijaLex" name is specific to the Marchal et al. 2021 / 2025 **discourse-connective** lexicon (entry 1.7), not a broad lexicon.
- Open Reddit / Twitter Naija dumps with permissive licenses: we found none. The Twitter API change in 2023 means most datasets are **research-only** and not redistributable.
- A dedicated Nigerian English (acrolect) corpus separate from Naija Pidgin: not found as a single packaged resource. ICE-Nigeria and the OED Nigerian-English additions exist as references but are not open-licensed corpora.

---

## Section 2 — Code-switch markers (concrete tokens)

Sources: British Council "Nigerian Pidgin – 20 useful words and phrases" (verified at https://www.britishcouncil.org/voices-magazine/nigerian-pidgin-words-phrases — fetched indirectly); Wikivoyage Nigerian Pidgin phrasebook (verified at https://en.wikivoyage.org/wiki/Nigerian_Pidgin_phrasebook); Marchal et al. (2021) connective lexicon; Adelani et al. 2024 ("Does Generative AI speak Nigerian-Pidgin?"); Wiley World Englishes 2024 article on Nigerian Pidgin proverbs (Callies & Onysko 2024). All entries are in the published sources cited; none invented.

| # | Token | Category | Function (formal Naija / casual / either) | Example |
|---|---|---|---|---|
| 1 | abeg | hedge / softener (request) | either | "Abeg, send me the link." |
| 2 | wahala | noun (trouble) / intensifier | either | "This product no get wahala." |
| 3 | no wahala | reassurance | either | "No wahala, e go work." |
| 4 | how far | greeting | casual | "How far, na you buy am?" |
| 5 | well well | reduplicated intensifier | either | "This cream dey work well well." |
| 6 | sef | discourse particle (even/also) | either | "I no sabi am sef." |
| 7 | na | copula / focus marker | either | "Na better lipstick be this." |
| 8 | dey | progressive aspect marker | either | "E dey shine like mirror." |
| 9 | go | future marker | either | "E go last for harmattan." |
| 10 | don | perfective aspect marker | either | "I don use am for two months." |
| 11 | sabi | verb (know/be skilled at) | either | "She sabi makeup well." |
| 12 | jare | exclamation / softener | casual | "It's not bad jare." |
| 13 | sha | discourse particle (anyway) | either | "E sweet sha." |
| 14 | o (sentence-final) | emphasis particle | either | "Yes o, I like am." |
| 15 | chai | exclamation (regret/surprise) | casual | "Chai, this powder cake my face." |
| 16 | shey | tag question marker | either | "Shey you don try the foundation?" |
| 17 | oga | honorific (boss) | either | "Oga, this product na fire." |
| 18 | madam | honorific (woman of standing) | either | "Madam, the cream too cost." |
| 19 | come | sequencer / "then" | casual | "She come buy am again." |
| 20 | comot | verb (remove / leave) | either | "Comot the smell quick." |
| 21 | yeye | adjective (useless/silly) | casual | "Na yeye product." |
| 22 | yama-yama | noun (rubbish) | casual | "The packaging na yama-yama." |
| 23 | gbam | exclamation (exact / agreement) | casual | "Gbam! That's the right shade." |
| 24 | scatter | intensifier (be excellent) | casual | "Her foundation game dey scatter." |
| 25 | die | intensifier (extremely) | casual | "E sweet die." |
| 26 | small small | reduplicated hedge ("a little") | either | "Apply small small." |
| 27 | now now | reduplicated intensifier ("right now") | either | "I want am now now." |
| 28 | no be small thing | praise intensifier | casual | "This cream no be small thing." |
| 29 | no be lie | agreement intensifier | casual | "No be lie, e dey work." |
| 30 | gist | noun (story / news) | either | "Give me the gist about Zaron's primer." |
| 31 | wetin | wh-pronoun (what) | either | "Wetin make am sweet so?" |
| 32 | shele / wetin dey | greeting (what's up) | casual | "Wetin dey today?" |
| 33 | epp | verb (help) | casual | "This serum no fit epp dry skin." |
| 34 | no vex | apology marker | either | "No vex, the package late." |
| 35 | omo | exclamation (filler / surprise) | casual | "Omo, the price hike sef." |
| 36 | as in | confirmation / emphasis | either | "As in, the lipstick stayed all day." |
| 37 | una | 2pl pronoun (you all) | either | "Una sef try this brand." |
| 38 | wahala dey | idiom ("there is trouble") | casual | "If e no match my undertone, wahala dey." |
| 39 | tear | intensifier ("very") | casual | "The mascara tear my expectation." |
| 40 | sote | connective ("until") | either | "I rub am sote my skin shine." |

Notes:
- Aspect markers (dey, go, don) and copula `na` are **grammatical**, not stylistic; they will appear in any Naija text and are the strongest signals separating Naija from Standard Nigerian English.
- Items 21–24 ("yeye", "scatter", "gbam") are documented in popular slang lists but appear less frequently in academic corpora; use with caution for formal text.
- Reduplication (entries 5, 26, 27) is a *productive* feature; LLMs should be able to generate novel reduplications, not just memorize specific tokens.

---

## Section 3 — Cultural preference axes for cosmetics / beauty

### 3.1 Skin tone vocabulary
- **Tokens:** `caramel`, `chocolate`, `dark chocolate`, `espresso`, `honey`, `bronze`, `deep`, `cool undertone`, `warm undertone`, `neutral undertone`, `yellow undertone`, `red undertone`, `melanin-rich`, `pro-melanin`, `dusky`.
- **Description:** Reviewers describe shades along a depth axis (caramel → chocolate → espresso) crossed with undertone (warm/yellow vs. cool/red vs. neutral). Dissatisfaction is most often "shade too light", "too pink/grey on me", or "ashy", not depth alone. Mented Cosmetics, Type Beauty Inc, and Shea Tribe explicitly market along this two-axis system.
- **Cite:** Mented Cosmetics blog "How to Find the Perfect Foundation for Dark Skin"; Type Beauty Inc "How to Find Your Best Foundation Match for Brown Skin"; Shea Tribe brand description in The Culture Custodian (https://culturecustodian.com/10-nigerian-skincare-brands-revolutionizing-the-skincare-industry/, verified).

### 3.2 Hair type vocabulary
- **Tokens:** `4A`, `4B`, `4C`, `kinky`, `coily`, `Z-pattern`, `shrinkage`, `protective style`, `braids`, `cornrows`, `twists`, `bantu knots`, `locs`, `wig`, `weave`, `relaxer`, `texturizer`, `wash-and-go`, `low porosity`, `high porosity`, `edge control`, `pre-poo`, `LCO method`.
- **Description:** Most Nigerian consumers identify as 4B/4C; descriptors emphasise *protection* (braids, wigs) and *moisture retention* (LCO, pre-poo). Reviewer concerns differ from Eurocentric review datasets — slip / detangling / shrinkage replace bounce / volume / shine.
- **Cite:** Wikipedia "Kinky hair" (https://en.wikipedia.org/wiki/Kinky_hair, verified by listing); Avrupa Hair guide "Afro Hair Types: A Complete Guide to 4A, 4B, and 4C Hair"; Walker (2007) for the Andre Walker hair-typing system, on which 4A/4B/4C terminology is built.

### 3.3 Religious considerations (halal / kosher / Muslim ingredient prohibitions)
- **Tokens:** `halal-certified`, `alcohol-free`, `wudhu-friendly`, `vegan` (often used as halal-adjacent), `no pork derivatives`, `no carmine`, `permeable`, `breathable nail polish`.
- **Description:** ~50% of Nigeria identifies as Muslim, concentrated in the North. Halal-relevant concerns: (a) ingredient sourcing — no pig-derived gelatin, no carmine (insect-derived); (b) ethanol — products with ≥1% ethanol are typically deemed haram per the Halal Products Research Institute; (c) wudhu-friendliness — nail polish must be permeable to water (`breathable`) so daily ablutions remain valid. NAFDAC (Nigerian regulator) does not currently issue a halal certification — that is provided by NSCIA (Nigerian Supreme Council for Islamic Affairs). We could not verify a specific NAFDAC halal-cosmetic ruling — flag this if needed.
- **Cite:** Inolex "Why Halal Cosmetics" (https://www.inolex.com/learn/beautyfood/why-halal-understanding); BeautyMatter "The Increasing Demand for Halal Beauty Brands"; Claudia Nour Cosmetics product positioning. NAFDAC ruling: NOT FOUND in our search — do not assert one exists.

### 3.4 Ingredient priorities
- **Tokens:** `shea butter` / `ori` (Yoruba), `African black soap` / `ose dudu` (Yoruba) / `ncha nkota` (Igbo) / `sabulun salo` (Hausa), `palm kernel oil`, `neem` / `dogonyaro` (Hausa), `camwood` / `osun`, `baobab`, `cocoa pod ash`, `aloe vera`, `coconut oil`, `cocoa butter`, `honey`, `lemon`, `kuli kuli` (groundnut residue, occasional skincare use).
- **Description:** Five ingredients dominate Nigerian beauty marketing: **shea butter (ori)**, **black soap (ose dudu / dudu osun)**, **palm kernel oil**, **neem (dogonyaro)**, **camwood (osun)**. Brands explicitly pitch these as ancestral / heritage. The presence of a Yoruba (or Hausa) ingredient name in a review is a strong cultural signal.
- **Cite:** Wikipedia "African black soap / Dudu-Osun" (https://en.wikipedia.org/wiki/Dudu-Osun, verified by listing); Modara Naturals "Ori Shea Butter" (https://www.modaranaturals.com/product/ori-shea-body-butter/); George (2006) "Cutaneous adornment in the Yoruba" Int. J. Dermatology — peer-reviewed source confirming shea butter's role in the Yoruba beauty tradition.

### 3.5 Climate concerns
- **Tokens:** `harmattan`, `dry season`, `rainy season`, `humidity`, `sweat-proof`, `non-comedogenic`, `lightweight`, `gel-based`, `water-resistant SPF`, `melt off`, `cake`, `oxidize`, `sun-resistant`, `transepidermal water loss`, `crackling lips`, `ashy`.
- **Description:** Two climate cycles drive product needs: (1) **harmattan** (Nov–Mar) — dry, dusty, cool — drives demand for heavy moisturisers, lip balms, occlusives; (2) **rainy/humid season** — drives demand for non-comedogenic, sweat-resistant, long-wear, mattifying. "Cake" and "oxidize" are the dominant complaint terms for foundations under heat.
- **Cite:** Hush Magazine "Top 10 Skincare Tips for Nigerian Women" (https://www.hushdng.com/post/toptenskincaretipsfornigerianwomen-combatingheatandhumidity); Guardian Nigeria "Me vs Harmattan: Your Beauty Guide" (https://guardian.ng/life/you-vs-harmattan-your-beauty-guide/); MyLab Africa "Best Korean Skincare for Harmattan Season" (https://mylabafrica.com/skincare-for-harmattan-season/).

### 3.6 Brand familiarity (Nigerian + Pan-African beauty brands)
- **Nigerian-founded brands:**
  - **House of Tara** (1998, Tara Durotoye) — https://houseoftara.com/ (verified by listing) — full-line makeup, eyeshadow palettes named for African royals.
  - **Zaron Cosmetics** (2010, Oke Maduewesi) — full-line makeup formulated for African complexions.
  - **Aweni Organics**, **Oriki Skincare**, **Arami Essentials**, **Narganics**, **House of Coco**, **Shea Tribe**, **Bathkandy**, **Olaedo Naturals**, **Orma Skincare** — all listed in the Culture Custodian survey (verified URL above).
  - **Tropical Naturals (Dudu-Osun)** — https://www.duduosun.com/ — black-soap incumbent.
- **Pan-African brands marketed in Nigeria:**
  - **Black Up** (French-founded 2000, deep-tone focus) — sold in Nigeria via Montaigne Place per Lux Afrique.
  - **Iman Cosmetics** (founded 1994 by Iman Abdulmajid; Somali-American — pan-African in positioning).
  - **Sleek MakeUp** (UK-based but heavily marketed to women of colour).
  - **Mented Cosmetics** (US-based but Nigerian co-founder; pro-melanin positioning).
- **Citation:** The Culture Custodian "10 Nigerian Skincare Brands Revolutionizing the Skincare Industry" (verified); Lux Afrique "Top 6 Makeup Brands by Nigerians" (https://luxafrique.net/top-6-makeup-brands-by-nigerians/); Pulse Nigeria "Top 5 Nigerian cosmetics brands".

---

## Section 4 — Naming conventions for synthetic Nigerian personas

### 4.1 Twenty common first names (mixed origins)

| # | Name | Sex | Origin | Meaning |
|---|---|---|---|---|
| 1 | Adebayo | M | Yoruba | "the crown meets joy" |
| 2 | Tunde | M | Yoruba | "(king) has returned" (often clipped from Babatunde) |
| 3 | Oluwaseun | M/F | Yoruba | "God has done [this]" |
| 4 | Ayomide | F | Yoruba | "my joy has come" |
| 5 | Folake | F | Yoruba | "honour with wealth" |
| 6 | Tayo | M/F | Yoruba | "joy is enough" |
| 7 | Chinedu | M | Igbo | "God leads/guides" |
| 8 | Chukwuemeka | M | Igbo | "God has done well" |
| 9 | Obinna | M | Igbo | "father's heart" |
| 10 | Ifeoma | F | Igbo | "good thing" |
| 11 | Adaeze | F | Igbo | "first daughter / princess" |
| 12 | Chiamaka | F | Igbo | "God is beautiful" |
| 13 | Ibrahim | M | Hausa (Arabic) | "father of nations" |
| 14 | Suleiman | M | Hausa (Arabic) | "man of peace" |
| 15 | Aminat / Aminah | F | Hausa (Arabic) | "trustworthy / faithful" |
| 16 | Hadiza | F | Hausa (Arabic) | "early / premature" |
| 17 | Fatima | F | Hausa (Arabic) | "captivating" |
| 18 | Musa | M | Hausa (Arabic) | "Moses" |
| 19 | Aisha | F | Hausa (Arabic) | "alive" |
| 20 | Halima | F | Hausa (Arabic) | "patient / gentle" |

### 4.2 Surname patterns (drawn from Surnam.es Nigerian frequency table — verified, ~75M registered persons)

| # | Pattern | Example surnames |
|---|---|---|
| 1 | Arabic / Islamic given-name surname (Hausa-Fulani) | Ibrahim (1.89M), Mohammed (1.30M), Abdullahi (1.22M), Yusuf (837K) |
| 2 | "Abu-/Abdu-" prefix (servant-of-Allah pattern, Hausa) | Abubakar (1.70M), Abdullahi (1.22M) |
| 3 | Yoruba "Ade-" / "Olu-" prefix (royalty / God) | Adebayo (601K), Adeyemi, Oluwole, Oluwaseun |
| 4 | Yoruba theophoric Ọlọrun- / Olu- contractions | Oluwole, Olusegun |
| 5 | Igbo "Chi-" / "Chukwu-" prefix (God) | Chinedu, Chukwuemeka |
| 6 | Igbo "Nwa-" / "-eke / -orie / -nkwo" market-day | Nwankwo, Okonkwo, Okorie |
| 7 | Day-of-week Anglo names (often Igbo / Niger Delta) | Sunday (804K), Monday, Friday |
| 8 | Patronymic single-name in genitive form (general) | Akpan (1.06M), Udo (604K), Okon (500K) — Ibibio/Efik |
| 9 | Place- / clan-derived (Hausa / Yoruba) | Kano, Sokoto, Lawal (667K), Bello (925K) |
| 10 | Christian first-name surnames | John (711K), Emmanuel (484K) |

Source: Surnam.es Nigerian list (https://surnam.es/nigeria, verified), drawn from registered-population data (the methodology page does not name an official Nigerian census source — flag as commercial scrape).

### 4.3 Honorifics

| Honorific | Typical use |
|---|---|
| Mr / Mrs / Miss | Standard, conservative; common in writing. |
| Sir / Ma | Universal in workplaces; "Ma" universally used for women regardless of marital status. |
| Aunty / Auntie | Any older woman, blood-related or not — fictive kinship. |
| Uncle | Same, for older men. |
| Bros (Bro) | Slightly older or peer male (urban). |
| Sis / Sista | Slightly older or peer female. |
| Oga | Boss / superior / older male; can be respectful or sarcastic. |
| Madam | Older woman of standing (often paired with name). |
| Chief | Bearer of a chieftaincy title; do not use casually. |
| Alhaji / Alhaja | Muslim pilgrim; common in the North as everyday respect. |
| Pastor / Reverend / Imam / Sheikh / Mallam | Religious titles, used in everyday address. |
| Dr / Engr / Barr / Arc | Professional titles — Nigerians use these heavily in everyday speech. |

Sources: Kperogi "Patience Jonathan's 'Son' and Other Fictive Kinship Terms in Nigerian English" (linguistics blog, https://www.farooqkperogi.com/2013/07/patience-jonathans-son-and-other.html, verified by listing — Kperogi is a published linguist); Kperogi "Evolution of 'Sir' in Nigerian English Usage" (2025); The Guardian Nigeria editorial pieces.

---

## Section 5 — Cultural reference points relevant to product reviews

| # | Reference | Brief context |
|---|---|---|
| 1 | **Owambe** (Yoruba: "it's there") | Lavish Yoruba-origin party (weddings, birthdays, anniversaries, naming ceremonies, funerals, chieftaincies). Drives demand for long-wear makeup, perfume layering, hair extensions, aso ebi tailoring. Wikipedia "Owanbe" (verified by listing). |
| 2 | **Aso Ebi** | Coordinated "family cloth" worn by guests; couples specify the fabric. Often paired with gele (head-tie), lace, ankara. Aso Ebi events are when premium beauty products are tested. Wikipedia "Aso ebi" (verified by listing). |
| 3 | **Aso Oke** (Yoruba: "top cloth") | Hand-woven cloth (Sayan, Alarri, Etu) for high-end traditional dress. |
| 4 | **Igba nkwu / igbeyawo** | Igbo wine-carrying ceremony / Yoruba traditional wedding — distinct from white wedding; reviewers often own products *for both*. |
| 5 | **Naming ceremony** (Yoruba *iso omo loruko*; Igbo *iputa nwa*) | Held 7–8 days after birth; a bridal-quality look is expected. Source: Marmalade Collective, Patriot.ng "Owambe culture". |
| 6 | **Sallah / Eid** | Eid-al-Fitr ("Small Sallah") and Eid-al-Adha ("Big Sallah" / *Ileya* in Yoruba) are the main Muslim festivals; they drive demand for halal beauty, perfume oils (*attar*), henna, modest fashion. Greeting: "Barka da Sallah". Wikipedia "Eid al-Adha in Nigeria" (verified by listing); Zikoko "8 Nigerian Muslims on How to Celebrate Eid al-Adha". |
| 7 | **Christmas / New Year (Detty December)** | Major beauty / hair / fashion spending peak — Lagos is at peak occupancy with returnees. |
| 8 | **Harmattan** (Nov–Mar) | Dry north-easterly wind — drives shea butter and lip-balm demand; reviewers frequently mention "this saved me during harmattan". Already cited in §3.5. |
| 9 | **Naija / Naijá** (self-name) | Self-referential ethnonym for the country and its culture; common in product reviews ("naija weather", "naija skin"). |
| 10 | **Detty December** | Lagos diaspora-return party season (weddings, concerts). |
| 11 | **Praise expression: "X dey work well well"** | Reduplicated intensifier for product praise. Source: monoed.africa, verified search; British Council list. |
| 12 | **Praise: "no be small thing"** | Strong endorsement, lit. "this is no small matter". Same source. |
| 13 | **Praise: "e sweet die"** | Pidgin intensifier "extremely good" — *die* as adverbial intensifier, common in product / food reviews. |
| 14 | **Praise: "scatter / dey scatter"** | Slang for "killing it / excellent". |
| 15 | **Complaint: "na yeye product"** | "[This is] a useless product." `yeye` = useless. Wikivoyage; Naijalingo. |
| 16 | **Complaint: "e too cost"** | "[It's] too expensive." Standard complaint. |
| 17 | **Complaint: "wahala dey"** | "[There's] trouble" — used metaphorically in reviews to flag a problem. World Englishes 2024 article on NPE proverbs (Callies & Onysko, verified by listing). |
| 18 | **Religious markers in reviews** | "By God's grace", "to God be the glory", "Alhamdulillah", "Insha Allah", "Chineke!" (Igbo: God!), "Olorun a gba e" (Yoruba: God will accept). Religious framings are common in everyday Nigerian speech and would appear in review text. |
| 19 | **Geography references** | Lagos (Lekki, Ikeja, VI, Surulere), Abuja (Maitama, Wuse), Kano, Port Harcourt — district names cue urban / coastal / northern climates and price tiers. |
| 20 | **Gele** | Elaborate head-tie worn for owambe; a specific beauty sub-economy (gele tutorials, gele tyers as a profession). Reviewers may rate edge-controls, foundation, and lip products by "gele compatibility". |

---

## Bias / risk note

1. **WAPE-vs-Naija conflation is the single biggest risk.** Most "Nigerian Pidgin" online resources (BBC Pidgin, MasakhaNEWS-pcm) are West-African Pidgin English, NOT Naija proper. Adelani et al. 2024 explicitly document that LLMs trained on "Pidgin" data primarily learn WAPE, and Naija speakers find WAPE output unidiomatic. Treat MasakhaNEWS / BBC Pidgin as **WAPE-style** and Wikipedia-Naija / NSC-UD as **Naija-style**. Mixing them without acknowledging the variety distinction will produce inauthentic generations.
2. **NaijaSenti is Twitter, urban, and code-mixed.** Skewed toward Lagos / Abuja diaspora register — under-represents rural, older, monolingual Naija speakers. The dataset's NC-SA license also blocks commercial reuse.
3. **Beauty-blog data is aspirational.** "Top Nigerian skincare brands" lists (Culture Custodian, Pulse, Lux Afrique) are PR / lifestyle journalism; they oversample Lagos high-income consumers and undersell mass-market products like Mama Lola or NIVEA Naturally Even (which are likely the top sellers by volume).
4. **Naijalingo and similar community dictionaries are unmoderated.** They include slang of variable currency; some entries are jokes or dated. Cross-check before lexicon inclusion.
5. **Brand list is metropolitan-skewed.** Northern (Hausa) beauty brands and indigenous Muslim-marketed lines are under-represented in our English-language sources. The catalog as written is Yoruba-leaning.
6. **Naming data (Surnam.es) is a commercial scrape**, not a Nigerian government census. We did not verify the underlying source. Treat frequencies as indicative, not authoritative.
7. **Religious diversity is collapsed.** ~50% Muslim, ~45% Christian, ~5% traditional; our brand and ingredient sources skew Christian / urban-southern. Halal-cosmetic cultural axes are under-resourced in our search; the NAFDAC ruling we expected does not exist (or we could not surface one).
8. **No verified Nigerian Twitter / Reddit dump exists with permissive license.** Plan around derivative resources (NaijaSenti tweets, BBC Pidgin) rather than raw social-media corpora.
9. **NaijaLex (the connective lexicon) availability needs verification** before relying on it — the Springer 2025 paper claims free release, but we did not locate the data drop URL through search.
10. **Yoruba dominance.** Owambe, aso ebi, dudu osun, ori — all primary cultural anchors in this catalog are Yoruba in origin. Igbo (igba nkwu, ofe nsala, omugwo) and Hausa (sallah, attar, henna, kayan ado) cultural anchors are present but less detailed in our English-language sources, and that imbalance will propagate into anything built on this catalog without deliberate counterweighting.

---

Sources index (key URLs verified during compilation): https://github.com/masakhane-io/lafand-mt; https://github.com/hausanlp/NaijaSenti; https://github.com/UniversalDependencies/UD_Naija-NSC; https://github.com/masakhane-io/masakhane-news; https://github.com/castorini/afriberta; https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter; https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0; https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus; https://huggingface.co/Davlan/afro-xlmr-large; https://arxiv.org/abs/2404.19442; https://arxiv.org/abs/2502.19784; https://link.springer.com/article/10.1007/s10579-025-09850-3; https://aclanthology.org/2021.codi-main.8.pdf; https://www.familysearch.org/en/wiki/Nigeria_Naming_Customs; https://surnam.es/nigeria; https://culturecustodian.com/10-nigerian-skincare-brands-revolutionizing-the-skincare-industry/; https://en.wikipedia.org/wiki/Owanbe; https://en.wikipedia.org/wiki/Aso_ebi; https://en.wikipedia.org/wiki/Dudu-Osun; https://en.wikipedia.org/wiki/Eid_al-Adha_in_Nigeria; https://www.britishcouncil.org/voices-magazine/nigerian-pidgin-words-phrases; https://en.wikivoyage.org/wiki/Nigerian_Pidgin_phrasebook.
