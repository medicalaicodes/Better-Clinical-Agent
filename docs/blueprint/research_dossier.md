# HERALD — Research Dossier (cached literature findings, July 2026)

Companion evidence file for `HERALD_blueprint.md`. Everything the blueprint asserts from the
literature is recorded here with enough detail to re-check without re-searching. Figures are the
cited authors' reported numbers, not independently reproduced.

---

# Research Notes — Batch 1: Benchmarks & Backbones

# MedAgentBench (Stanford, arXiv 2501.14654; NEJM AI 2025)
- First benchmark requiring autonomous agent interaction with a **FHIR-compliant virtual EHR**.
- 300 clinician-written, verifiable tasks across 10 categories; 100 synthetic patients, >700k data elements.
- Docker-deployable HAPI-FHIR-style server; agent interacts via HTTP GET/POST on FHIR resources
  (Observation, MedicationRequest, Procedure...).
- Best model at publication: Claude 3.5 Sonnet v2 @ 69.67% success. Unsaturated.
- MedAgentBench v2 (Biocomputing/PSB 2026): prompt+tool redesign, memory-from-failure component,
  tools to avoid raw HTTP construction, explicit plan-before-act, few-shot good/bad examples.
- KEY FOR BLUEPRINT: this is the single most usable *free, local, agentic* EHR environment for an MVP.

# FHIR-AgentBench (arXiv 2509.19319)
- LLM agents for interoperable EHR QA over Synthea-generated FHIR data.
- Comparison table of EHR QA benchmarks: MIMICSQL (MIMIC-III), EHRSQL (eICU+MIMIC-III),
  EHRSQL-2024 (MIMIC-IV), FHIR_DATA (Synthea), ICU_DATA (Synthea).
- Synthea = fully synthetic, no DUA → ideal for a student MVP.

# HealthAdminBench (arXiv 2604.09937) — computer-use agents on healthcare admin tasks.

# Medmarks (arXiv 2605.01417) — open-source LLM benchmark suite for medical tasks; aggregates many.

# The Illusion of Readiness in Health AI (Microsoft Research; arXiv 2509.18234, Sept 2025)
CRITICAL EVIDENCE for benchmark-to-reality mismatch. Six flagship models (GPT-4o, o3, GPT-5,
Gemini 2.5 Pro, DeepSeek-VL2, LLaVA-Med 1.5) x six medical benchmarks.
Stress tests used (DIRECTLY REUSABLE AS OUR STRESS SUITE):
1. **Modality sensitivity / input ablation** — remove the image, see if accuracy holds.
   Models frequently still answer correctly → evidence they never used the image (shortcut).
2. **Distractor perturbation** — shuffle/alter answer choices; answers flip on trivial changes.
3. **Visual substitution** — swap in a different/irrelevant image.
4. **Clinician-guided rubrics** to profile *what a benchmark actually measures*
   (image perception vs. knowledge recall vs. reasoning).
Findings: high leaderboard scores hide brittleness + shortcut learning; models fabricate fluent
rationales citing non-existent image features; "fluency in explanation does not equate to valid
inference"; benchmarks vary widely in what they measure yet are treated interchangeably;
gains on VQA do NOT transfer to report generation.
Recommendation from authors: stress tests, clinician-guided benchmark profiling, robustness &
trustworthiness reporting rather than leaderboard scores.

# Related robustness/validity literature to cite
- Hager et al., *Nature Medicine* 2024, 30(9):2613-2622 — "Evaluation and mitigation of the
  limitations of LLMs in clinical decision-making" (MIMIC-CDM; LLMs fail on real sequential
  clinical decision-making, poor at following guidelines, sensitive to info order/amount).
- MedFuzz (2024) — adversarial robustness of medical QA.
- "Medical LLM benchmarks should prioritize construct validity" (2025).
- "Hidden flaws behind expert-level accuracy of multimodal GPT-4V in medicine", npj Digital
  Medicine 7:190 (2024) — correct answers with wrong rationales.
- Bedi et al., *JAMA* 2025 — systematic review of testing/evaluation of LLM health applications.
- Bedi et al., *JAMA Netw Open* 2025;8(8):e2526021 — "Fidelity of medical reasoning in LLMs".

# Backbones (open, local-first)
# MedGemma (Google; arXiv 2507.05201, tech report rev. 2026)
- Built on Gemma 3. Variants: 4B multimodal, 27B multimodal, 27B text-only.
- Image encoder = **MedSigLIP** (400M, SigLIP tuned on CXR, histopath patches, dermatology, fundus)
  — released separately, usable as a standalone medical image encoder / retrieval backbone.
- MedGemma 27B text: **87.7% MedQA**, within 3 pts of DeepSeek R1 at ~1/10 inference cost.
- MedGemma 4B fine-tuned: SOTA CXR report generation, **RadGraph F1 30.3**.
- ~500x compute-cost difference between MedGemma 4B and most expensive comparator.
- KEY: this is the right MVP backbone — 4B runs on a single consumer/lab GPU, is multimodal,
  and MedSigLIP gives a free cross-modal retrieval tool.
# Others
- Lingshu (arXiv 2506.07044) — generalist multimodal medical foundation model.
- LLaVA-Med, CheXagent, RadFM, LLaVA-Rad — CXR/VLM specialists.
- QoQ-Med — domain-aware GRPO multimodal clinical foundation model.
- Qwen2.5-VL / Qwen3-VL — strong open general VLM backbone, used by MACRO(3B)/Meissa(4B).

# Known VLM fragility to design against
- PSF-Med paraphrase-sensitivity benchmark: MedGemma-4B baseline **flip rate 14.6%** on binary
  yes/no MIMIC-CXR questions under paraphrase; mean margin difference 1.63 logits.
  → Justifies a *paraphrase-invariance* stress test and self-consistency-based reliability score.

# Abstention / conformal
- Conformal prediction (Vovk 2005; Angelopoulos & Bates 2023): distribution-free, finite-sample
  coverage under exchangeability. Split conformal is the practical variant.
- **Conformal risk control** (Angelopoulos et al., ICLR 2024) and **Learn-then-Test**
  (Angelopoulos et al., Ann. Appl. Stat. 2025, 19(2):1641-1662) — guarantee an error rate on the
  *accepted* set, not just coverage. This is exactly what an abstention layer needs.
- Mitigating LLM Hallucinations via Conformal Abstention (Abbasi-Yadkori et al., NeurIPS 2024 WS).
- Conformal Factuality (Mohri & Hashimoto 2024) — back-off chain, emit the most specific claim
  that still satisfies a correctness guarantee. Very relevant to "hedged answer instead of refusal".
- CARE: A Conformal Safety Layer for Medical Summarization (arXiv 2606.08969, som-shahlab)
  — conformal safety layer bolted onto a medical NLG system. Direct precedent for our design.
- "Conformal selective prediction with cost-aware deferral for safe clinical triage under
  distribution shift" — *Scientific Reports* 2026 (s41598-026-40637-w). Sepsis prediction;
  temporally-separated ID vs OOD splits; risk–coverage trade-off. Good methodological template.
- Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty (arXiv 2601.12471).
- RadFlag (arXiv 2411.00299) — black-box hallucination detection for medical VLMs: sample extra
  reports at higher temperature, check sentence-level support across samples, flag unsupported
  sentences. CHEAP, IMPLEMENTABLE, no training. Strong candidate for the Reliability Estimator.
- "Entropy Alone is Insufficient for Safe Selective Prediction in LLMs" (arXiv 2603.21172)
  → do NOT use entropy as the only reliability signal.
- ICML 2026 Workshop: Statistical Frameworks for Uncertainty in Agentic Systems — active venue,
  explicitly lists "uncertainty for routing to tools and subagents" as a topic. GOOD TARGET VENUE.


---

# Research Notes — Batch 2: Systems, Tools, Governance

# Differential diagnosis (DDx) paradigm — evidence it matters
- AMIE / "Towards Accurate Differential Diagnosis with LLMs" (arXiv 2312.00164; *Nature* 2025).
  302 NEJM CPC cases. Standalone LLM **top-10 59.1% vs unassisted clinicians 33.6%** (p=0.04).
  Clinicians + AMIE: top-10 51.7% vs 36.1% without, vs 44.4% with search. Also produced *more
  comprehensive* differentials. Authors explicitly caution: CPCs ≠ general competence, and the
  study did NOT test **goal-directed information acquisition under uncertainty** — the hard part.
- Kanjee et al.: GPT-4 on 70 NEJM CPCs — exact final dx 39%, correct dx anywhere in DDx list 64%.
- OpenMedicine AI (medRxiv 2025.10.28.25338335): **deterministic controller** + LLM on 302 CPCs:
  46.0% top-1 / 79.1% top-10, vs AMIE 32.5%/68.9%, vs physicians 15.6%/20.9%.
  Introduces **Capture@K** — "actionability" metric: is the Top-n differential sufficient to make a
  clinician order the *diagnostic test of record* (DToR) or its immediate precursor?
  → EXCELLENT metric idea to adopt: judges the DDx by the *next action it licenses*, not just
    string match against a label.
- Lab-result ablation study (npj Digital Medicine 2025, s41746-025-01556-8): adding lab data
  improves DDx accuracy **by up to 30%** across GPT-4/GPT-3.5/Llama-2-70B/Claude-2/Mixtral.
  → Direct empirical justification for a *modality-value* / evidence-acquisition module.
- HeartAgent (arXiv 2603.10764): cardiology DDx agent. CoT baseline top-3 on MIMIC 0.377–0.421;
  HeartAgent +36%. On NEJM subset top-3 ≈0.556–0.567 across Llama-3.3-70B, Qwen-2.5-32B,
  MedGemma-27B — i.e. **agent scaffolding matters more than backbone choice**. Strong support for
  a backbone-agnostic blueprint.

# Report / evidence evaluation metrics (needed for the Traceability axis)
- Surface NLG (BLEU/ROUGE/METEOR/CIDEr/BERTScore) — inadequate, misses negation and factuality.
- **CheXbert-F1** (Smit 2020): 14 CheXpert observation labels. Narrow; ICU-biased; confounds
  findings (consolidation) with diagnoses (pneumonia).
- **RadGraph-F1** (Jain 2021; Delbrouck 2022): entity–relation graph (ANAT, OBS-DP...). Preserves
  anatomical context; no entity normalization; hard to port beyond CXR.
- **RadCliQ** (Yu 2023): regression combining BLEU + BERTScore + CheXbert vector sim + RadGraph-F1.
- **GREEN** (Ostmeier 2024): LLM judge extracts findings from both reference and candidate, scores
  matched findings weighted by clinical significance, and *explains* errors by category.
  Known issue: **pronounced length bias** (reported by RadAgent, arXiv 2604.15231).
- **RadFact** (Bannur et al. 2024, MAIRA-2): entailment-based, sentence-level; designed for
  *grounded* report generation (claims tied to boxes). CLOSEST INSTRUMENT TO "EVIDENCE FIDELITY".
- **FineRadScore**: per-line error identification + severity.
- **TBFact** (Microsoft healthcare blog): factual inclusion / distortion / omission; κ=0.760 with
  experts. Referenced in user's own ERC_survey.
- **RadFlag** (arXiv 2411.00299): black-box hallucination detection — resample reports at higher
  temperature, use an LLM to count how many samples support each sentence, flag unsupported ones.
  Cheap, training-free → **use directly as a Reliability Estimator signal in v1.**

# Open specialist tools available for a CXR MVP (all pip-installable, local)
- **TorchXRayVision** (`pip install torchxrayvision`, MIDL 2022, arXiv 2111.00595):
  - Multiple *heterogeneous* pretrained DenseNet classifiers: weights `densenet121-res224-nih`,
    `-chex`, `-mimic_ch`, `-mimic_nb`, `-pc` (PadChest), `-rsna`, `-all`. 18 pathology labels.
  - Also segmentation (PSPNet lung/heart) and autoencoders.
  - Unified dataset wrappers for NIH/CheXpert/MIMIC/PadChest/RSNA/VinDr + **built-in utilities to
    construct covariate shift and spurious-correlation splits**.
  - CRITICAL PROPERTY FOR THIS BLUEPRINT: several models trained on *different* source hospitals
    give a natural pool of specialists with genuinely different, measurable per-domain reliability.
    That is exactly the substrate a Tool Router + Reliability Estimator needs. No new training.
  - An MCP-compliant server wrapper for TorchXRayVision classification/segmentation already exists.
- **MedRAX** (ICML 2025, arXiv 2502.02673, github.com/bowang-lab/MedRAX): first general CXR agent;
  ReAct loop over **8 specialized CXR tools** (classification, segmentation, grounding, VQA, report
  gen); *no additional training*. **63.1% on ChestAgentBench** (+6.7 over GPT-4o), 90.35% SLAKE VQA,
  79.1% mF1-14 MIMIC-CXR report gen. **ChestAgentBench**: 2,500 six-choice queries over 675
  expert-curated Eurorad cases, 7 categories (detection, classification, localization, comparison,
  relationship, characterization, diagnosis).
  → This is the correct MVP *baseline to beat*, and a legitimate code starting point.
- **CXReasonAgent** (KAIST, arXiv 2602.23276, Feb 2026): LLM + clinically grounded tools; restricts
  itself to **12 predefined diagnostic tasks whose image-derived evidence can be reliably extracted**
  (cardiomegaly/cardiac size, mediastinal widening, aortic knob, ascending/descending aorta
  enlargement, aortic tortuosity, trachea deviation, carina angle, plus image-quality checks:
  inspiration, rotation, projection, inclusion). Tools built on **CheXStruct** — rule-based
  geometric computations from radiologist-defined criteria → **deterministic, reproducible evidence**.
  Ships **CXReasonDial**: 1,946 multi-turn dialogues over 12 tasks.
  → KEY DESIGN LESSON: constrain the agent's task space to what tools can *verifiably* measure.
    Deterministic geometric tools are the cheapest possible source of high-reliability evidence.
- **RadFabric** (arXiv 2506.14142): multi-agent CXR system built **on MCP**; pathology agents +
  anatomical interpretation agent + reasoning agent. Precedent for MCP as the tool bus.
- **RadAgent** (arXiv 2604.15231): tool-using agent for stepwise chest **CT** interpretation.
- **GAZE** (arXiv 2605.00876): notes that CXR-Agent / MedRAX / AURA all use *AI-derived* tools
  rather than the plain viewer controls (zoom, crop, windowing) radiologists actually use.
- Curated lists to mine: github.com/Nanboy-Ronan/awesome-medical-imaging-agents ;
  github.com/AgenticHealthAI/Awesome-AI-Agents-for-Healthcare.
  Notable entries: "An Explainable Agentic AI Framework for Uncertainty-Aware and
  Abstention-Enabled Acute Ischemic Stroke Imaging Decisions" (2026);
  "DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents" (2026);
  "OPGAgent: An Agent for Auditable Dental Panoramic X-ray Interpretation" (2026).

# Datasets — practical access reality (2026)
- **MIMIC-CXR / MIMIC-IV**: PhysioNet **Credentialed** Health Data License 1.5.0. Requires CITI
  human-subjects training + credentialed account + DUA.
   As of mid-2026 PhysioNet warns that **credentialed-access applications face significant delays**
    due to staffing changes. Do NOT put a student internship's critical path on this.
   The CDUA **prohibits sending MIMIC data to third-party LLM APIs** (OpenAI, etc.) or online
    platforms; Azure OpenAI with human-review opt-out is the named exception; "MIMIC data must not
    be stored or retained by third-party LLM services."
    → THIS IS THE STRONGEST SINGLE ARGUMENT FOR A LOCAL, OPEN-WEIGHT BACKBONE (MedGemma-4B / Qwen-VL).
      It is a compliance constraint, not a preference.
- **Openly downloadable alternatives** (no credentialing, or lightweight registration):
  - **VinDr-CXR** (Sci Data 2022): 18,000 images, **radiologist bounding-box local labels** (train
    split fully released). Best free source of *localized* ground truth for grounding metrics.
  - **PadChest** (160,868 imgs / 67,625 patients / 109,931 reports; 174 findings, 19 differential
    diagnoses, 104 anatomical locations, hierarchical taxonomy **mapped to UMLS**; ~27% manually
    validated by physicians; Spanish reports). Ontology mapping already done → great for the
    ontology-grounding requirement.
  - **NIH ChestX-ray14** (112,120 imgs, 30,805 patients; noisy text-mined labels).
  - **RSNA Pneumonia Detection** (Kaggle), **SIIM-ACR Pneumothorax**.
  - **CheXpert / CheXpert Plus** (224,316 CXRs; CheXpert Plus adds 36M report tokens).
  - **CheXmask** — anatomical segmentation masks across ChestX-ray8/CheXpert/MIMIC/PadChest/VinDr.
  - **CXR-LT 2026 challenge** (arXiv 2602.22092): 145,000+ images from **PadChest + NIH**, chosen
    explicitly "addressing recent access restrictions on MIMIC-CXR". Task 1 = 30 known classes
    (winning mAP 0.5854), Task 2 = **6 unseen OOD rare classes** (winning mAP 0.4315).
    → Ready-made *open-world / OOD* stress split with published SOTA numbers.
  - **Eurorad** cases (used by ChestAgentBench) — publicly browsable teaching cases.
  - **Synthea** — fully synthetic FHIR patients, no DUA at all. Basis of FHIR-AgentBench.
  - **SMART Health IT sandbox** (r4.smarthealthit.org) — free live FHIR R4 test server.
  - **MedAgentBench** Docker environment — 100 synthetic patients, 700k elements, 300 tasks.
  - VQA: VQA-RAD, SLAKE, PathVQA, OmniMedVQA, PMC-VQA, MMMU-Health.

# MedHELM (Bedi et al., *Nature Medicine* 2026; arXiv 2505.23802; Apache-2.0)
- Clinician-validated taxonomy: **5 categories / 22 subcategories / 121 tasks**, built with 29
  clinicians across 14 specialties; 96.7% clinician agreement on category mapping.
- 35–37 benchmarks (17 existing, 5 reformulated, 13 new); 14 datasets private for privacy.
- **LLM-jury** (3 LLMs + task rubrics) agreement with clinicians **ICC = 0.47**, beating
  clinician–clinician ICC 0.43, ROUGE-L 0.36, BERTScore-F1 0.44.
  → Legitimizes LLM-jury as the open-ended scorer for our evaluation, WITH the caveat that it only
    just matches human inter-rater noise.
- Normalized accuracy by category: Clinical Note Generation 0.73–0.85; Patient Communication
  0.78–0.83; Medical Research Assistance 0.65–0.75; **Clinical Decision Support 0.56–0.72**;
  **Administration & Workflow 0.53–0.63**. The two lowest are exactly the agentic ones.
- Runs on HELM (`crfm-helm`, Python 3.10). In 2026 became a community-led project.

# Multi-agent failure evidence (why NOT to build a debate swarm)
- **MAST** (Cemri et al., arXiv 2503.13657; NeurIPS 2025): MAST-Data = 1,600+ annotated traces
  across 7 MAS frameworks (AutoGen, ChatDev, CrewAI, ...). Taxonomy of **14 failure modes** in 3
  categories: (i) system design / specification issues (~41.8%), (ii) inter-agent misalignment
  (~36.9%), (iii) task verification (~21.3%). κ=0.88 inter-annotator. LLM-as-judge annotator
  ~94% accuracy, Cohen's κ 0.77.
  Key quote-idea: **improvements in base model capability alone will not fix the taxonomy.**
  Sub-modes worth designing against: 1.1 disobey task spec, 1.3 step repetition,
  1.5 unaware of termination conditions, loss of conversation history.
- "Silent Failure in LLM Agent Systems" (arXiv 2606.08162) names **Behavior Routing Deficiency**
  — task allocation failing to match subtasks to agent capability under normal operation.
  → This is precisely our Tool Router risk; it now has a name in the literature.

# Tool-trust / reliability literature (the core research hook)
- **"The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents"**
  (ACL 2026 Long, arXiv 2601.07264). FINDING: **evidence tools (e.g. web search) systematically
  induce severe overconfidence** because of noise in retrieved information, while **verification
  tools (e.g. code interpreters) ground reasoning via deterministic feedback and mitigate
  miscalibration.** Proposes RL fine-tuning jointly optimizing accuracy + calibration.
  → Maps perfectly onto medicine: a classifier's softmax is an "evidence tool" (overconfidence
    inducer); a rule-based geometric measurement (CheXStruct-style) is a "verification tool".
    Design implication: pair every evidence tool with a verification tool where possible.
- **Robust-TO / "Confidence-Aware Tool Orchestration for Robust Video Understanding"**
  (arXiv 2606.26904). Names the **"Blind Trust Problem"**: every perception output treated as
  equally reliable; model confidence decoupled from the conditions that produced it. Their design:
  each tool returns evidence in a **shared format** = {concrete prediction, temporal/spatial
  grounding, **calibrated reliability score**}; three-tier (high/med/low) synthesis; confidence-cost
  GRPO reward. Result 56.4% avg, +10.6pp over best open baseline, beats Gemini-2.5-Pro (46.2%).
  Notes frontier models lose 15–30pp under corruption **while self-reported confidence barely moves**.
  → THIS IS THE EXACT ARCHITECTURE WE WANT, IN A DIFFERENT DOMAIN. The medical instantiation is open.
    Strong evidence the research gap is real and the solution shape is validated elsewhere.
- **ATTC — "When to Trust Tools?"** (arXiv 2604.08281): model's confidence in the *code block it
  generated before calling a tool* predicts whether the prerequisite reasoning was sound; converts
  implicit awareness into an explicit control signal. → cheap pre-call reliability prior.
- ChromaFlow (arXiv 2605.14102): negative ablation — adding a recovery/orchestration layer
  *degraded* accuracy and increased operational noise. Orchestration is not free.

# Governance / regulatory anchors (for §10)
- **EU AI Act (Reg. 2024/1689)**. Diagnosis, therapy planning, clinical decision support and remote
  monitoring fall in the high-risk bracket.
  - Art. 9 risk management; Art. 10 data governance; Art. 11 technical documentation (Annex IV);
    **Art. 12 record-keeping — automatic event logging over the system lifetime**, enabling
    (a) identification of risk situations / substantial modification, (b) post-market monitoring
    (Art. 72), (c) monitoring of operation (Art. 26(5));
    Art. 14 human oversight incl. ability to intervene/interrupt; Art. 19 / 26(6) **≥6-month log
    retention**; Arts. 43/47/48/49 conformity assessment, DoC, CE marking, registration.
  - **Timeline (as of July 2026):** entered into force 1 Aug 2024. Digital Omnibus on AI proposed
    19 Nov 2025; provisional political agreement 7 May 2026; Parliament endorsed 16 June 2026;
    **Council final approval 29 June 2026**. Result: **Annex III stand-alone high-risk obligations
    deferred to 2 December 2027**; Annex I product-embedded high-risk to **2 August 2028**.
    NOT deferred: Art. 50 transparency obligations still apply **2 August 2026**
    (Art. 50(2) machine-readable marking moved to 2 Dec 2026 for systems already on market).
    New prohibition (NCII/CSAM) from 2 Dec 2026. Penalties up to 7% global turnover / €35M.
  - Related standards: ISO/IEC 42001 (AI management system), IEC 62304 (software safety class A/B/C),
    forthcoming **prEN ISO/IEC 24970 on AI system event logging** (log format/granularity).
- **HIPAA / MCP**: MCP is a transport protocol, NOT a compliance guarantee. Compliance must be
  implemented at the server layer: OAuth 2.1 / SSO, RBAC, audit logging of every tool call,
  TLS 1.2+ in transit, AES-256 at rest, PHI minimization, signed BAA. Narrow workflow-specific
  tools, not general API/database access; wildcard FHIR scopes are a real over-permission hazard;
  read and write operations need different controls.
- **OpenTelemetry GenAI semantic conventions** (SIG formed Apr 2024; semconv v1.41 as of ~May 2026,
  still **Development** stability — attribute names can change).
  Span types: `invoke_agent` (root), `chat` (per LLM call), `execute_tool` (per tool), plus workflow
  and MCP spans. Attributes: `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
  `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`; content captured as span *events*
  (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`) rather than
  indexed attributes — **because attributes are always indexed, size-limited, and would expose PHI**;
  events can be dropped/redacted at the Collector before leaving the network.
  → This is the single most important practical governance decision: emit OTel GenAI spans from day
    one, put PHI in droppable events, and the audit trail + redaction layer is 80% free later.
  Backends: Langfuse (self-host), Grafana Tempo, OpenSearch, Phoenix/Arize.
- Emerging: PROV-AGENT (W3C PROV extension for agent provenance); "Decision Trace Schema for
  Governance Evidence" (arXiv 2604.09296); DEMM-Bench (agent-runtime governance-evidence
  sufficiency, arXiv 2606.20634); "Runtime Governance for AI Agents: Policies on Paths"
  (arXiv 2603.16586).

# Missing modality
- Reconstruction/imputation of missing modalities is a **liability under heavy missingness** —
  a generated profile "may look plausible while adding unsupported evidence to the pipeline"
  (Meta-Modal Agent, arXiv 2605.25007). → argues for *disclose-and-degrade*, not *impute-and-hide*.
- Practical graceful-degradation pattern that works (arXiv 2511.15847, ICU mortality): mask missing
  branches out of the gating distribution, renormalize over observed experts, fall back to a
  **calibrated single-specialist prediction**. Reported cost ≈ AUPRC −0.09 to −0.11 while
  **calibration metrics stay near branch-level values**. → Degrade accuracy, preserve calibration.
- MREF-AD (arXiv 2512.10966): mask missing experts from gating, renormalize remaining gate weights.
- Gong et al. 2026: quantify uni-modal aleatoric uncertainty to handle missingness (MIMIC, eICU).
- Cerebra (user's survey): halts entirely if MRI missing — the negative example to design against.


---

# Research Notes — Batch 3: Evaluation Methodology & Confirmed Gaps

# MedCTA: A Benchmark for Clinical Tool Agents (arXiv 2606.11702, 10 Jun 2026)
- 107 real-world clinician-verified clinical tasks, executable trajectories over **5 deployed tools**,
  multimodal inputs (radiology images, pathology slides, reports). "Step-implicit" tasks — the
  agent is not told which steps to take.
- **Process-aware** evaluation axes: tool selection, argument validity, execution stability,
  trajectory fidelity, outcome quality.
- 18 open + closed multimodal models benchmarked. Findings:
  - reliable clinical tool use is **unsolved**;
  - **gold-standard (oracle) tool routing exposes a large "controller gap"**;
  - most failures come from **protocol instability, under-calling, and weak local grounding**,
    NOT from final synthesis.
  → Confirms: (1) the router is the bottleneck, not the writer; (2) "controller gap vs oracle
    routing" is an accepted metric; (3) under-calling (failing to invoke a needed tool) is a
    first-class failure mode we must measure.

# AgentProp-Bench (arXiv 2604.16706) — error propagation, quantified
- 2,000 tasks / 2,300 traces / 4 domains / 9 production LLMs / 100 human-validated labels.
- Method: **controlled injection** of errors (rather than waiting for natural ones) → enables
  causal measurement of propagation.
- RO1 Judge reliability: **substring matching agrees with humans at κ = 0.049 (chance level)**;
  a **three-LLM ensemble judge reaches κ = 0.432** with conservative bias.
  → MANDATE: validate any automatic judge against a human-labelled subset before reporting numbers.
- RO2 Propagation: a **parameter-level injection propagates to a wrong final answer with
  human-calibrated probability ≈ 0.62 (range 0.46–0.73 across models)**.
  → Baseline expectation for our "corrupted tool" stress test.
- RO2b: **rejection rate and recovery rate are statistically independent** (Spearman ρ = 0.126,
  p = 0.747). Robustness to tool errors is **at least two-dimensional**.
  → MANDATE: report *detection* and *recovery* as separate metrics. A system that notices a bad
    tool but can't route around it is a different (and commoner) failure than one that never notices.

# "How to benchmark medical AI agents" — PLOS Medicine, 9 July 2026 (pmed.1005170)
- Very recent methodological framing piece. Notes agents will act in **clinician-supporting roles
  rather than replacing them** for the foreseeable future.
- Cites: AgentClinic (now *npj Digital Medicine* 2026, s41746-026-02674-7); Bean et al.,
  *Nature Medicine* 2026;32(2):609–15 — randomized preregistered study of LLM reliability as
  medical assistants for the general public; Hager et al. *Nat Med* 2024;30(9):2613–22;
  Chiu, Pitis & van der Schaar, NeurIPS 2025 D&B — simulated viva voce exams for clinical reasoning.

# Orchestration framework landscape (mid-2026)
- **LangGraph**: orchestration as an explicit state machine — pure nodes, typed edges, shared
  typed state (TypedDict + reducers), durable checkpointing, time-travel debugging, native
  human-in-the-loop interrupts, LangSmith/OTel tracing. Consistently described as the default for
  regulated / auditable / stateful production workflows. Learning curve ~10–14 engineer-days.
- **CrewAI**: role/goal/backstory hierarchy; fastest prototype; murkier debugging. Built on LangChain.
- **AutoGen**: group-chat paradigm; **Microsoft has moved it to maintenance** in favour of the
  unified Microsoft Agent Framework (AutoGen + Semantic Kernel). Do not start new projects on it.
- **OpenAI Agents SDK**: lowest friction if OpenAI-locked; no model flexibility → unusable under
  PhysioNet DUA constraints.
- Repeated practitioner warning: many "agent" problems are better solved by a well-structured chain
  with a few tool calls; agents add non-determinism, debugging cost, and money. Also:
  **set termination conditions tightly** (e.g. max_consecutive_auto_reply=3) — matches MAST failure
  mode 1.5 "unaware of termination conditions".

# Consolidated statement of the gap this blueprint targets
Confirmed by independent sources, none of which solve it together for clinical multimodal agents:
1. Tools are trusted blindly (Robust-TO "Blind Trust Problem"; MedAgent-Pro's −16.5% F1 from one
   weak tool; CARE's over-editing coordinator).
2. Tool augmentation *changes* calibration in tool-type-dependent ways — evidence tools induce
   overconfidence, verification tools reduce it (Confidence Dichotomy, ACL 2026). Nobody exploits
   this deliberately in a clinical pipeline.
3. Routing is the empirical bottleneck and there is a large measured oracle-vs-actual controller
   gap (MedCTA 2026; ToolSelect 72% vs 96% oracle from the user's own survey).
4. Errors propagate at ≈0.62 once injected; detection and recovery are independent capabilities
   (AgentProp-Bench). No medical agent reports either.
5. Benchmarks reward shortcuts; models answer correctly with the image removed (Illusion of
   Readiness). No medical agent reports input-ablation sensitivity.
6. Missingness is handled by halting (Cerebra) or by imputation that fabricates unsupported
   evidence (Meta-Modal Agent warning). "Disclose and degrade with preserved calibration" is
   demonstrated in supervised fusion (arXiv 2511.15847) but not in an agent.
7. Abstention in medicine is studied for single models (conformal abstention, CARE conformal safety
   layer, Knowing When to Abstain) but **not at the coordination level**, and not as a *typed,
   actionable request for a specific missing datum*.
8. Governance: Art. 12-style logging is a legal requirement with no engineering schema; OTel GenAI
   semconv is the natural carrier but is still Development-stability and covers telemetry, not
   clinical evidence semantics.

→ The blueprint's contribution is the *integration*: a typed evidence ledger with calibrated
  per-packet reliability, a hypothesis-level fusion layer that can preserve conflict, a typed
  disposition (answer / caveat / request / escalate), and an entailment-gated narrative — all
  emitted as OTel-compatible spans so the governance layer is a later attachment, not a rewrite.
