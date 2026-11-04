# Clinical AI Agent: Master Implementation Log & Research Notes

Personal daily development log and research notebook tracking architecture, math formulations, model integrations, bug hunts, architectural audits, and benchmark results for the clinical diagnostic agent project throughout August 2026.

---

## Aug 5, 2026
- Setting up the repo and planning the core architecture.
- Spent the whole morning going through the latest literature:
  - MedAgent-Pro (ICLR 2026) — great two-level separation between guideline-level medical rules and patient-level quantitative tools.
  - MedRAX (ICML 2025) — clean ReAct tool execution with real PyTorch vision backbones.
  - SDBench / MAI-DxO (Microsoft Research, 2025) — sequential diagnostic workup framing where the agent must actively inquire and request tests step-by-step rather than receiving the full patient chart in a single prompt dump.
  - AMIE (Nature 2024/2025) — conversational clinical interview metrics and diagnostic accuracy scoring.
  - Stanford RadFact — factuality and evidence grounding verification.
- Goal: build a modular clinical diagnostic agent that simulates hospital emergency workups. Initial vitals -> initial differential -> sequential test ordering -> vision/lab/RAG tool execution -> grounded clinical report.
- Setup: Python 3.11 venv, verified PyTorch 2.x CUDA support on my local RTX GPU. Initialized git repository.

## Aug 6, 2026
- Built foundational Pydantic v2 data models in `src/clinical_agent/schemas.py`.
- Defined:
  - `PatientCaseInput` / `PatientCase`: age, sex, chief complaint, triage vitals, doctor triage notes, image file paths, lab dictionaries.
  - `FindingItem`: structured clinical finding (present/absent status, confidence score, quantitative value, unit, reference interval).
  - `EvidencePacket`: wrapper with UUID, timestamp, modality tag (`IMAGE_CXR`, `LAB_VALUE`, `CLINICAL_NOTE`, `GUIDELINE`), reliability score (0.0 to 1.0).
  - `DifferentialHypothesis`: condition name, ICD-10 code, suspicion tier (`HIGH_SUSPICION`, `MODERATE_SUSPICION`, `RULE_OUT`), clinical rationale.
  - `DiagnosticOrder`: tool name, display name, clinical indication, parameters.
  - `ClinicalReport`: primary impression, ranked differential list, key findings gathered, actionable recommendations, and disposition (`ADMIT_INPATIENT`, `DISCHARGE_WITH_FOLLOWUP`, `ESCALATE_TO_PHYSICIAN`).
- Wrote initial unit tests in `tests/test_schemas.py` to verify validation boundaries, enum serialization, and immutable dict conversions.

## Aug 7, 2026
- Worked on mathematical foundations for diagnostic uncertainty in `src/clinical_agent/evidence_math.py`.
- Formulated:
  - Shannon Information Entropy: `H(P) = -sum(P(Hi) * log2(P(Hi)))` to track differential uncertainty in bits across hypotheses.
  - Qualitative suspicion tiers mapped to log-odds:
    - High suspicion -> 4:1 odds (logit = ln(4) = +1.386)
    - Moderate suspicion -> 1:1 odds (logit = ln(1) = 0.000)
    - Rule out -> 1:4 odds (logit = ln(0.25) = -1.386)
  - Softmax log-odds normalization: `softmax_tiered_priors` across candidate conditions.
  - Basic belief assignment for Dempster-Shafer cross-modal conflict detection: `K = sum(m1(B) * m2(C))` where B and C are disjoint hypothesis sets.
- Tested helper functions on synthetic vectors. Initial math looks neat on paper, but I need to see how it behaves once real model outputs hit it.

## Aug 8, 2026
- Built the initial LangGraph state machine in `src/clinical_agent/graph.py` and state container in `src/clinical_agent/state.py`.
- Defined `AgentClinicalState` TypedDict with reducers:
  - `evidence_ledger`: append-only list accumulating findings across turns.
  - `active_differential`: current ranked list of hypotheses.
  - `conflict_log`: tracked multi-modal discrepancies.
  - `iteration_count` & `max_iterations`.
- Connected nodes: `intake_node` -> `differential_formulation_node` -> `planner_node` -> `tool_runner_node` -> `evidence_synthesis_node` -> `grounding_gate` -> conditional routing.
- Ran a dummy pass through the graph to verify state passing and transitions work without runtime errors.

## Aug 9, 2026
- Implemented `src/clinical_agent/grounding.py` based on Stanford RadFact factuality methodology.
- Core problem: LLMs generate confident, fluent reports that hallucinate clinical findings that were never actually observed in the tests.
- Built `GroundingVerifier`:
  - Parses every asserted claim in the generated clinical impression.
  - Cross-references each claim against the UUIDs and content in the active `evidence_ledger`.
  - Calculates grounding rate: `grounding_rate = len(supported_claims) / len(total_claims)`.
  - Applies a penalty discount if an asserted differential hypothesis lacks supporting evidence packets.
- Wrote unit tests in `tests/test_grounding.py`.

## Aug 10, 2026
- Integrated TorchXRayVision DenseNet-121 classifier in `src/clinical_agent/tools/cxr_classifier.py` and `src/clinical_agent/models/densenet_cxr.py`.
- Downloaded `densenet121-res224-all` pretrained weights from TorchXRayVision.
- The model outputs raw probabilities across 18 acute thoracic pathologies: Consolidation, Cardiomegaly, Effusion, Pneumothorax, Atelectasis, Infiltration, Edema, Nodule, Mass, Pneumoperitoneum, etc.
- Added operating threshold margins: instead of a blunt 0.5 cutoff, evaluated `Margin = |p_k - theta_k|` using the checkpoint's calibrated per-pathology operating thresholds (`op_threshs`).
- Benchmarked local PyTorch inference: ~42ms per 224x224 chest radiograph on CUDA.

## Aug 11, 2026
- Added Image Quality Assessment (IQA) in `cxr_classifier.py`.
- Idea: if an X-ray is blurry, underexposed, or noisy, the classifier shouldn't claim 100% confidence.
- Implemented:
  - Sharpness metric: Laplacian gradient variance across the image matrix.
  - Exposure & contrast score: normalized histogram spread and mean intensity checks.
- Formulated a 3-factor composite reliability: `r = q_quality * c_auroc * (1.0 - u_entropy)`.
- Tested with artificially blurred and darkened images to confirm `r` drops accordingly.

## Aug 12, 2026
- Built quantitative emergency laboratory panel validator in `src/clinical_agent/tools/laboratory.py`.
- Coded standard physiological reference intervals from the Tietz Textbook of Clinical Chemistry:
  - CBC: WBC (4.5 - 11.0 x 10^9/L), Platelets (150 - 450), Hemoglobin (12.0 - 17.5 g/dL).
  - Inflammatory: Procalcitonin (< 0.10 ug/L), hs-CRP (< 3.0 mg/L).
  - Cardiac: Troponin I (< 0.04 ng/mL), BNP (< 100 pg/mL), NT-proBNP (< 300 pg/mL).
  - Coagulation: D-Dimer (< 0.50 ug/mL FEU).
  - BMP: Sodium (135 - 145 mEq/L), Potassium (3.5 - 5.0), Chloride, Bicarbonate, BUN, Creatinine, Glucose.
- Added quantitative percentage deviation: `Delta = |value - reference_midpoint| / reference_midpoint` to measure severity of abnormal lab results.

## Aug 13, 2026
- Developed clinical triage notes parser in `src/clinical_agent/tools/clinical_notes.py`.
- Triage notes contain unstructured clinical signs and doctor observations.
- Built regex extraction for key physical signs: crackles/rales, wheezes, jugular venous distension (JVD), chest wall point tenderness, relative bradycardia (Faget sign), currant-jelly sputum, unilateral calf edema.
- Implemented negation detection ("denies fever", "no JVD", "without calf swelling") so negated symptoms aren't misclassified as positive findings.

## Aug 14, 2026
- Built empirical Evidence-Based Medicine (EBM) likelihood ratio database in `src/clinical_agent/tools/ebm_database.py` and `evidence_math.py`.
- Sourced positive and negative likelihood ratios (LR+, LR-) from JAMA Rational Clinical Examination literature:
  - Focal Consolidation for Pneumonia: LR+ = 8.33, LR- = 0.27
  - Elevated Procalcitonin (> 0.50 ug/L): LR+ = 7.08, LR- = 0.17
  - Elevated BNP (> 400 pg/mL) for Heart Failure: LR+ = 7.15, LR- = 0.08
  - Elevated Troponin (> 0.04 ng/mL) for ACS: LR+ = 13.14, LR- = 0.09
  - Chest wall point tenderness for Costochondritis: LR+ = 4.50, LR- = 0.30
- Added log-odds likelihood updating: `logit(P(H|E)) = logit(P(H)) + sum ln(LR_bounded)`.

## Aug 15, 2026
- Built NCBI StatPearls and guideline retrieval tool in `src/clinical_agent/tools/statpearls_client.py` and `src/clinical_agent/tools/knowledge_rag.py`.
- Connected to NCBI Entrez Bookshelf REST API (`esearch.fcgi`, `efetch.fcgi`).
- Added persistent local disk caching in `~/.cache/statpearls/` to avoid network latency during benchmark runs.
- Added structured risk calculators: CURB-65 (Pneumonia Severity), Wells Score (PE risk), and Framingham Diagnostic Heart Failure Criteria.

## Aug 16, 2026
- Integrated MedSAM (Medical Segment Anything Model) ViT-Base checkpoint (`wanglab/medsam-vit-base`) via HuggingFace `transformers` in `src/clinical_agent/models/medsam_wrapper.py` and `src/clinical_agent/tools/medsam.py`.
- **First big realization & pivot during model testing**:
  - Tried box prompting MedSAM on the central mediastinum to measure Cardiothoracic Ratio (CTR).
  - Inspected the visual segmentation masks: MedSAM consistently snapped to the sharp bony edges of the thoracic spine instead of the soft-tissue cardiac silhouette!
  - Tried inferring cardiac width from the gap between lung fields: produced crazy CTR numbers (0.86, 0.002, 0.77).
  - However, box prompting on the left and right hemithorax segmented lung fields cleanly (0.85 to 0.95 IoU).
  - Decision: Use MedSAM for dual hemithorax lung-field area segmentation and compute the Thoracic Lung Symmetry Ratio:
    `S = min(Area_left, Area_right) / max(Area_left, Area_right)`
    Symmetry under 0.85 indicates dense unilateral consolidation, large pleural effusion, or lobar atelectasis.

## Aug 17, 2026
- Built LiteLLM interface in `src/clinical_agent/llm_client.py`.
- Added real-time token chunk streaming (`stream=True`) directly to terminal stdout so the clinician can see the agent's diagnostic thought process live.
- Created root `settings.json` and `src/clinical_agent/config.py` using `pydantic-settings` to manage model providers, endpoints, and temperature centrally.
- Tested against local Ollama (`medgemma`, `llama3`) and cloud APIs (Gemini, OpenAI, Claude).
- Added `test_litellm_connection()` diagnostic healthcheck.

## Aug 18, 2026
- Worked on multi-class Bayesian odds compounding in `src/clinical_agent/evidence_math.py`.
- Fixed a compression paradox where evaluating odds across 3+ hypotheses artificially squashed posteriors.
- Formulated exact Softmax log-odds compounding:
  `log W_k = log P_0(H_k) + sum log LR_eff(e_m | H_k)`
  `P(H_k | E) = exp(log W_k - max_j log W_j) / sum_i exp(log W_i - max_j log W_j)`
- Tested with multiple concordant findings; posteriors now properly update without artificial squashing.

## Aug 19, 2026
- Enforced a strict **Zero-Simulation Policy** across the codebase:
  - Deleted mock file fallbacks and dummy string paths (`sample_pneumonia.png`).
  - Curated genuine clinical radiographs from NIH ChestX-ray14 and IEEE open hospital datasets into `data/sample_images/`:
    - `streptococcus_pneumonia.jpg`: dense RUL consolidation.
    - `nih_cardiomegaly.png`: marked cardiomegaly with pulmonary vascular congestion.
    - `normal_cxr.jpg`: clear lung fields, sharp costophrenic angles.
    - `effusion_cxr.png`: dense right pleural effusion with meniscus blunting.
    - `covid_viral_pneumonia.jpg`: bilateral peripheral ground-glass opacities.
    - `klebsiella_pneumonia.jpg`: upper lobe cavitary necrosis.
    - `legionella_pneumonia.png`: dense consolidation.
    - `pneumocystis_pneumonia.png`: perihilar ground-glass haziness.
    - `chest_ct_scan.dcm`: native DICOM cross-sectional volume.

## Aug 20, 2026
- Built `AttendingPhysicianOrchestrator` in `src/clinical_agent/agent.py`.
- Implemented Expected Value of Information (VoI) ranking across available diagnostic orders:
  `VoI = H_prior - E[H_post]`
- The agent calculates VoI across candidate tests (CXR, Labs, MedSAM, Guideline RAG) to select the single most informative test for the next turn.

## Aug 21, 2026
- Built multi-turn interactive EHR simulation environment in `src/clinical_agent/evaluation/simulated_ehr.py`.
- The simulation blinds the agent: only triage vitals and complaint are given initially. The agent must explicitly order an investigation to unmask its result.
- Added cost tracking (CXR $120, Labs $150, MedSAM $85, Notes $0) to measure workup cost-efficiency.
- Benchmark run on initial cases: 80% accuracy, average cost $339.

## Aug 22, 2026 — First Major Cracks in the Math Machinery
- Started noticing severe bugs during live end-to-end testing with real Ollama, real DenseNet, and real labs:
  1. **Bayesian Posterior Overconfidence (100% vs 0% saturation)**:
     - On a live pneumonia case, the Bayesian posterior outputted literally `1.000 / 0.000 / 0.000 / 0.000`.
     - Root cause: multiplying likelihood ratios from the same modality (e.g. CXR reporting Consolidation, Lung Opacity, and Infiltration simultaneously) assumes conditional independence, which is completely false! Correlated findings compounded into an astronomical log-LR (~380,000x product), saturating the softmax.
     - Added a temporary log-LR magnitude cap (`max_log_lr_magnitude = 5.0`) to enforce Cromwell's rule (never be 100% certain).
  2. **Grounding Renormalization Bug**:
     - Found that when `GroundingVerifier` discounted an ungrounded hypothesis by 0.4x, it didn't renormalize the remaining distribution, causing probabilities to sum to 0.72 instead of 1.0! Fixed renormalization.

## Aug 23, 2026 — Dempster-Shafer Conflict & RAG Failure Modes Exposed
- Ran into two more deep architectural issues:
  1. **Dempster-Shafer Conflict False Alarms**:
     - On 4-condition differentials, the Dempster conflict mass `K` was inflating to 0.58–0.65 even when both CXR and Labs agreed on Pneumonia as #1!
     - Why? Any small residual probability mass spread across runner-up hypotheses creates non-zero cross-modal conflict in Dempster's orthogonal sum. Concordant cases were falsely escalating to human physicians.
     - Added a patch requiring modalities to actually disagree on their top hypothesis and clear a `min_confident_margin = 0.15` before considering conflict.
  2. **StatPearls Keyword Search Was Breaking**:
     - Watched `statpearls_client.py` return "Staphylococcal Pneumonia" for a "Community-Acquired Pneumonia" query due to crude regex title matching.
     - When network calls timed out, the fallback was generating LLM-hallucinated guidelines — but assigning them a reliability score of 1.0 (higher than real neural networks or lab values)!
  3. **Discovered Fake Code in Old Files**:
     - Audited `tools/cxr_segmenter.py` and found it was determining CTR by matching the image filename string `if "cardiomegaly" in lower_path:` rather than running any model. Immediately deleted that file.

## Aug 24, 2026 — The Architectural Audit: Realizing v1 Was Flawed
- Spent the day doing a deep audit of `evidence_math.py` and the entire v1 system (documented in `backup/REDESIGN_PROPOSAL.md` and `backup/TODO_later.md`):
  - **The Core Fallacy of v1**: We were trying to compute a "shadow quantitative Bayesian differential" using hand-picked formulas (`q * c * (1-u)` reliability, 4:1 prior odds, static EBM tables, Dempster-Shafer belief fusion) and forcing the LLM to rubber-stamp the math.
  - But trust computed about an LLM by an external likelihood formula is structurally wrong. The formulas were overconfident, brittle, and constantly needed patches (capping log-LR, margin gates, synonym expanders).
  - Furthermore, having two parallel orchestrators (`graph.py` vs `orchestrator.py`) meant logic was duplicated and drifting.
  - Literature check (MedRAX, MedAgent-Pro, MAI-DxO, ClinicBot, SEMA-RAG):
    - MedRAX doesn't do shadow math — it uses clean ReAct tool calling with real vision models.
    - MAI-DxO showed that LLMs reasoning sequentially over evidence beat physicians (85% vs 20%) without arithmetic fusion.
    - SEMA-RAG / ClinicBot showed that RAG needs semantic vector embeddings and citation entailment verification, not regex web scraping.
- **Decision: Ditch v1's brittle shadow math. Build a clean-start v2 architecture** where:
  - The Attending LLM is the primary clinical reasoner in an active, sequential ReAct loop.
  - The LLM's claims are strictly constrained and gated by evidence ledger citations (anti-hallucination via RadFact).
  - All specialist tools use real PyTorch models and standard clinical reference intervals.
  - RAG uses semantic cosine similarity over indexed society guidelines.

## Aug 25, 2026 — The Clean-Start v2 Refactoring
- Archived the legacy v1 codebase into `backup/v1/`.
- Modularized the clean repository structure:
  ```
  src/clinical_agent/
  ├── schemas.py              # Pydantic v2 data models
  ├── config.py               # Centralized settings
  ├── llm_client.py           # LiteLLM streaming client
  ├── agent.py                # Sequential Attending Clinical Agent ReAct loop
  ├── graph.py                # Compiled LangGraph state machine
  ├── models/
  │   ├── densenet_cxr.py     # TorchXRayVision DenseNet-121 wrapper
  │   └── medsam_wrapper.py   # MedSAM ViT-B wrapper
  ├── tools/
  │   ├── base.py             # Tool base class
  │   ├── chest_xray.py       # DenseNet-121 classifier with threshold margins
  │   ├── medsam.py           # MedSAM thoracic symmetry segmenter
  │   ├── laboratory.py       # Tietz lab panel validator
  │   ├── medical_knowledge.py# Guideline RAG with semantic cosine matching
  │   └── clinical_notes.py   # Triage notes parser
  ├── prompts/
  │   └── __init__.py         # Differential, action, and report prompts
  └── evaluation/
      ├── sample_cases.py     # 8 benchmark cases
      ├── simulated_ehr.py    # Multi-turn blind EHR simulator
      └── run_benchmark.py    # Benchmark runner & metrics
  ```
- Implemented pure sequential loop in `agent.py`: at each turn, the Attending LLM sees previous findings, reasons about what to order next, executes the specialist tool, and loops until sufficient evidence is collected or turn limit is reached.

## Aug 26, 2026 — 8-Case Clinical Benchmark Validation
- Expanded the benchmark suite in `src/clinical_agent/evaluation/sample_cases.py` to 8 comprehensive emergency presentations:
  1. `PT_STREP_65M`: Typical Streptococcus pneumoniae Lobar CAP (RUL consolidation, PCT 1.85, WBC 17.8) -> Bacterial Pneumonia (`ADMIT_INPATIENT`)
  2. `PT_ADHF_71M`: Acute Decompensated Heart Failure (BNP 1680, Cardiomegaly, JVD) -> Decompensated HF (`ADMIT_INPATIENT`)
  3. `PT_PE_48F`: Acute Pulmonary Embolism (D-Dimer 3.85, clear CXR, calf swelling) -> Pulmonary Embolism (`ADMIT_INPATIENT`)
  4. `PT_COSTO_34M`: Costochondritis / Musculoskeletal Chest Wall Strain (point tenderness, normal labs & CXR) -> Costochondritis (`DISCHARGE_WITH_FOLLOWUP`)
  5. `PT_PJP_HIV_40M`: PJP Pneumonia in HIV Host (SpO2 84%, CD4 85, perihilar haziness) -> PJP Pneumonia (`ADMIT_INPATIENT`)
  6. `PT_KLEB_DM_62F`: Klebsiella Necrotizing Pneumonia (currant-jelly sputum, upper lobe cavitation, WBC 21.4) -> Klebsiella Pneumonia (`ADMIT_INPATIENT`)
  7. `PT_LEGIONELLA_60M`: Legionnaires' Disease (hyponatremia Na 126, diarrhea, Faget sign) -> Legionnaires' Disease (`ADMIT_INPATIENT`)
  8. `PT_COVID_VIRAL_58M`: COVID-19 Acute Viral Infection (anosmia, bilateral GGOs, CRP 78) -> COVID-19 (`ADMIT_INPATIENT`)
- Ran full evaluation:
  - **Top-1 Accuracy:** 8/8 (100.0%)
  - **RadFact Grounding Rate:** 100.0% (zero hallucinated claims)
  - **Average Diagnostic Turns:** 2.8 turns
  - **Average Investigation Cost:** $312.50
- Every single case converged on the correct clinical diagnosis and appropriate disposition.

## Aug 27, 2026 — Final Polish, 8-Page LaTeX Paper & Release
- Wrote full 8-page academic technical report in LaTeX: `docs/CLINICAL_AGENT_REPORT.tex` and compiled to `docs/CLINICAL_AGENT_REPORT.pdf`.
  - Detailed system architecture, mathematical formulations, tool specifications, benchmark tables, and end-to-end Case 0 radiograph trace.
- Created interactive CLI demo script `demo_execution.py`:
  - `python demo_execution.py --case 0` (runs interactive sequential workup with live token streaming)
  - `python demo_execution.py --case 0 --graph` (runs LangGraph state machine)
  - `python demo_execution.py --benchmark` (runs full 8-case benchmark)
- Updated `README.md` with system overview, architecture flowchart, benchmark results, and academic citations.
- Ran full unit test suite: `pytest tests/` -> 14/14 tests passing in 5.8s.
- All code clean, verified, documented, and ready for deployment.

---

## Nov 2, 2026 — Audit of v2: what is actually broken
- Re-read every file and checked library sources instead of trusting my own comments. Findings:
  - **Double sigmoid in the chest X-ray wrapper.** For `densenet121-res224-all`, TorchXRayVision's `forward()` already applies `sigmoid` + `op_norm` (checked in `torchxrayvision/models.py`). My wrapper applied a second sigmoid, so every "probability" sat in [0.50, 0.73] and "certainty" could never be negative. The 0.65 cut-off was only accidentally close to the model's real operating point.
  - The calibrated operating thresholds (`op_threshs`) were loaded but never used.
  - MedSAM CTR comes from fixed prompt boxes (heart box = 25–75% of the width), so CTR is capped at about 0.54 by geometry, not anatomy.
  - `get_intake_view()` was never called: the "sequential" workup saw everything from turn 1, and `ground_truth` was never compared.
  - **The guideline RAG was circular:** 9 hand-written summaries, keywords repeated twice per chunk, and a synonym map pointing words straight at diseases. The 8 test vignettes used the same keywords. **The Aug 26 "8/8 = 100%" is a smoke test, not a result.** With n=8 the 95% CI is roughly 63–100% even before the circularity.
  - "Grounding rate" only checked that evidence exists; top-k matching used substrings both ways.
  - A missing disposition defaulted to `ADMIT_INPATIENT`.
  - An API key was committed in `settings.json` (rotated, file removed).

## Nov 3, 2026 — Literature pass and redesign of retrieval
- Read the current medical RAG work and pulled exact numbers:
  - **MIRAGE/MedRAG** (ACL Findings 2024): the benchmark plus the Textbooks/StatPearls corpora; RRF of several retrievers was best.
  - **i-MedRAG** (PSB 2025), Llama-3.1-8B: MedQA 64.73 → 66.54 (MedRAG) → 75.02 (i-MedRAG); MMLU-Med 77.23 → 78.05 → 79.61.
  - **RAG²** (NAACL 2025), Llama-3-8B MedQA: no RAG 57.7, MedRAG 56.4 (*worse*), RAG² 64.6.
  - **"Rethinking RAG for Medicine"** (arXiv 2511.06738): only 22% of the top-16 passages relevant; standard RAG often lowered factuality.
  - MedRAG-KG (WWW 2025) discriminative features; MEDDxAgent (ACL 2025) interactive DDXPlus; ACTMED / MedClarify information-gain question selection.
- Takeaway: **retrieval quality and evidence selection are the bottleneck**, and RAG needs a way to *not* hurt.
- Designed **DDx-RAG**: an option-free differential → SUPPORT/DISCRIMINATE/REFUTE queries → hybrid retrieval with quotas → cited fact-check → evidence-justified adjudication against closed-book votes, with verification depth gated by vote agreement. Full design: `docs/RAG_PIPELINE.md`.

## Nov 4, 2026 — Phase 1 implementation (RAG engine + MIRAGE harness)
- **Removed dead v2 code:**
  - `tools/clinical_notes.py` (unused regex)
  - `evaluation/metrics.py`, `simulated_ehr.py`, `run_benchmark.py` (misleading or no-op)
  - `prompts/orchestrator_skills.md`, the old `scratch/test_rag*.py`, `settings.json`
  - `litellm` (one OpenAI-compatible client now serves everything)
- **New modules:**
  - `retrieval/corpus.py`: Textbooks (HF) + StatPearls (NCBI, chunked with a port of MedRAG's chunker so snippet boundaries match the baselines).
  - `retrieval/indexes.py`: BM25 (`bm25s`), MedCPT bi-encoder + exact FAISS inner product, MedCPT cross-encoder.
  - `retrieval/search.py`: RRF fusion, rerank, per-query quotas, near-duplicate removal, neighbour expansion, noise injection for robustness tests.
  - `llm.py`: one client for vLLM/Ollama/llama.cpp with per-thread call/token accounting and robust JSON/answer parsing.
  - `rag/methods.py`: `cot`, `cot_sc`, `medrag`, `rationale_rag`, `imedrag` (baselines) and `ddx_rag` (ours), each with every switch needed for ablations.
  - `evaluation/mirage.py` + `stats.py`: resumable runner, Wilson CIs, exact McNemar, fixed/broke counts, cost per question, option-shuffle robustness, MedQA-train "dev" split for tuning.
- **Fixed:** double sigmoid; operating-point thresholds; unknown disposition now escalates; the knowledge tool now queries the real retriever.
- **Bug found during the smoke test:** servers that ignore `n` (llama.cpp, Ollama) produced identical "independent" votes because each top-up call reused the same seed. Each extra sample now gets `seed + i`.
- **Tests:** 16 new CPU-only tests (fake LLM + toy corpus). They also check that the PLAN prompt never sees the answer options (question-only retrieval).
- **Serving config:** the server now has a dedicated 40 GB A100, so Llama-3.1-8B runs in **bf16** (the same precision as the published baselines, with no quantization caveat). vLLM uses 85% of the card with prefix caching (the votes and later DDx-RAG steps share a prompt prefix), 128 concurrent sequences and 48 benchmark workers. FP8 is kept as a one-flag fallback for a shared GPU. MedGemma-27B (FP8) is an optional second model.

## Nov 4, 2026 (evening) — First GPU runs and server lessons
- **Server setup problems, each now guarded in code or docs:**
  - Driver 535 = CUDA 12.2. Default PyPI torch (CUDA 13.2) silently reported no GPU, so MedCPT started encoding on CPU. `build_index.py` now refuses to run without a visible GPU unless `--device cpu` is passed.
  - Latest vLLM wheels need `libcudart.so.13`. Pinned `vllm==0.11.0` (CUDA 12 line) in `requirements-vllm.txt`.
  - vLLM 0.11 + transformers 5.x breaks the tokenizer (`all_special_tokens_extended`). Pinned `transformers<5`.
  - Serving by repo id offline failed because we skip Meta's `original/` weights. `serve_vllm.sh` now serves the local snapshot folder automatically.
- **Sanity run (50 MedQA questions, Llama-3.1-8B bf16):**
  - CoT 64.0% (published: 64.73%), so the baseline reproduces.
  - MedRAG 70.0%, DDx-RAG 70.0%. With n=50 the 95% CIs are about ±13 points; this is not a result.
  - DDx-RAG made 3.3 calls per question (the votes are one request) and had 0 unparsed answers vs 4 for MedRAG.
  - Fixed/broke vs CoT: DDx-RAG 6/3, MedRAG 9/6. That is a first hint of less retrieval harm, still to be confirmed on the full set.
- **Added `scripts/analyze_results.py`** for trace-level error analysis: does the gate signal predict correctness, light vs full path, good vs bad overrules, evidence by intent, per-subject accuracy, fixed/broke question ids.
