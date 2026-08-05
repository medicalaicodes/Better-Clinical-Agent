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
