# Implementation & Workflow Plan
# Composite Blueprint for Governed Multimodal Agentic AI in Healthcare

**Document type:** Sequential engineering workflow (research implementation plan).
**Status:** Pre-build planning document. No production code — architecture, contracts, evaluation protocol, and phase gates only.
**Related artifacts:** `idea1_evidence_reliable_coordination_concept_note.md` (evidence-reliability governance layer), `Workflow.md` (HIPAA privacy/compliance governance layer), `internship_presentation.md` (source gap analysis, 27-paper survey).

---

# 0. Objective

Build and evaluate a **composite agentic architecture** for multimodal clinical AI by:

1. Decomposing five architecturally distinct SOTA systems into a common layer taxonomy.
2. Empirically ablating each candidate module on a fixed backbone and fixed task to identify the best-performing module per pipeline slot.
3. Assembling a "best-of-breed" composite pipeline from the empirically winning modules — not from assumption.
4. Wrapping the composite pipeline in **two independent governance layers**: privacy/compliance (HIPAA-style) and evidence-reliability/coordination.
5. Benchmarking the final system against a strong single-agent baseline at every stage, so the project does not silently reproduce the "coordination overhead without gain" failure already documented in MAM, CoMMa, and AgentRx.

**What this project is not:** It is not a claim that multi-agent orchestration is automatically better than a single strong agent. Every stage below requires a single-agent control condition. If the composite pipeline does not beat that control on accuracy, calibration, robustness, or auditability, that is a valid and reportable finding.

---

# 1. The five source architectures and their role

AgentClinic is not a peer competitor — it has no fixed accuracy of its own; it is an **environment**, not a diagnostic architecture. It is used as the evaluation harness in Stage 4, not as a module donor in Stage 1–3.

| # | System | Core mechanism | Blueprint role | Native reference |
|---|---|---|---|---|
| 1 | **MMedAgent** | LLaVA-Med orchestrator trained to route across 6 specialized tools spanning 5 imaging modalities | Donor for the **routing/tool-selection layer** | EMNLP 2024 Findings |
| 2 | **CARE** | Entity proposal → entity-conditioned segmentation → evidence-grounded reasoning → coordinator; entropy-filtered masks discard unreliable segmentation before reasoning | Donor for the **evidence-provenance/grounding layer** | ICLR 2026 (Microsoft Research Asia + Yale) |
| 3 | **MedAgent-Pro** | Indicator-specific tool agents (e.g. vCDR, disc-haemorrhage) feed quantitative measurements to a decision agent | Donor for the **interpretable indicator-extraction layer**; also the primary case study for tool-error cascade risk | arXiv 2025 |
| 4 | **CoMMa** | Deterministic per-agent evidence streams combined via Shapley-inspired contribution-aware weighting, not naïve averaging | Donor for the **aggregation/fusion layer** | arXiv 2026 |
| 5 | **AgentClinic** | Simulated doctor-patient dialogue with embedded cognitive/implicit bias injection, tool use, multimodal NEJM cases | **Evaluation harness**, not a module donor | AgentClinic benchmark |

---

# 2. Common layer taxonomy (Stage-1 deliverable)

Before any code is written, map all five systems onto one shared taxonomy. This table is the actual output of Stage 1 — it is what makes "best module per slot" a falsifiable claim rather than an assumption.

| Layer | AgentClinic (harness) | MMedAgent | CARE | MedAgent-Pro | CoMMa |
|---|---|---|---|---|---|
| Input / triage | Simulated patient presents symptoms | Multimodal query received | Image + question | Case + modality | Partitioned evidence streams |
| Tool/specialist selection | N/A | Trained router over 6-tool registry | Entity proposal model | Indicator-specific tool agents | Fixed per-stream deterministic agents |
| Evidence extraction | Doctor agent requests tests/images | Tool output (mask, classification) | Confidence-filtered segmentation mask | Quantitative clinical indicator | Per-agent embedding/logit |
| Aggregation/fusion | Doctor agent reasons over dialogue | Orchestrator LLM synthesizes tool outputs | Coordinator checks thought-answer consistency | Decision agent combines indicators | Contribution-weighted (Shapley-inspired) mixture |
| Uncertainty/reliability handling | Bias-injection stress test only | Not specified | Entropy-derived confidence filter | Not specified (cascade risk here) | Learned contribution weight = implicit reliability signal |
| Output/explanation | Final diagnosis + transcript | Report synthesis | Answer traceable to explicit ROI/entity | Diagnosis + cited indicator | Prediction + per-agent attribution |

**Known gap across all four donor systems:** none has a validated mechanism for detecting a malfunctioning *tool* before its output reaches aggregation. This is the exact MedAgent-Pro disc-haemorrhage cascade risk, and it is the reason Stage 5 (evidence-reliability governance) exists.

---

# 3. Phase-by-phase workflow

```
Phase 0   Scaffold, environment, fixed backbone selection
Phase 1   Common layer taxonomy + module inventory        (Section 2 above)
Phase 2   Reimplementation of candidate modules on fixed backbone
Phase 3   Module-level ablation (fixed task, fixed dataset, one slot varies at a time)
Phase 4   Composite assembly (best module per slot, empirically chosen)
Phase 5a  Privacy/compliance governance overlay (HIPAA blueprint + modality-aware gate)
Phase 5b  Evidence-reliability governance overlay (Idea 1 coordinator)
Phase 6   Joint evaluation harness (AgentClinic + static benchmark + privacy/adversarial suite)
Phase 7   Reporting: four independent scorecards
```

---

# Phase 0 — Scaffold, environment, fixed backbone

**Goal:** Remove backbone/dataset confounds before any module comparison is meaningful.

- Select **one** open-weight multimodal backbone used consistently across all reimplemented modules (e.g. a Qwen2.5-VL-class model), matching what recent multimodal clinical-agent benchmarks already use for comparability.
- Select **one** primary task and dataset family for Stage 3 ablation. Recommended starting point: an ophthalmology or chest-imaging task where MedAgent-Pro-style indicator tools and MMedAgent-style routing both apply, so all four donor mechanisms can be exercised on the same cases.
- Freeze compute budget, decoding temperature, and prompt templates.
- Log every model version, tool version, and library version into a run manifest (this manifest is reused by the audit layer in Phase 5a).

**Gate:** one fixed backbone + one fixed task/dataset + reproducible environment before Phase 2 begins.

---

# Phase 1 — Common layer taxonomy + module inventory

**Goal:** Produce the table in Section 2 with full traceability: every cell must cite the specific section/algorithm/figure of the source paper it is drawn from.

**Deliverable:** `taxonomy.md` + `module_inventory.yaml` listing, per system, per layer: input contract, output contract, and dependencies.

**Gate:** every layer has exactly one candidate module from at least one of the four donor systems; gaps (e.g. "no system has a tool-reliability check") are explicitly flagged, not silently left blank.

---

# Phase 2 — Reimplementation of candidate modules

**Goal:** Reimplement the *mechanism*, not the full paper, for each donor module, wired to the Phase-0 fixed backbone.

| Module to reimplement | Source | Contract |
|---|---|---|
| Tool router | MMedAgent | Input: multimodal query + tool registry. Output: selected tool id(s) + routing confidence |
| Entity-conditioned segmentation + confidence filter | CARE | Input: image + proposed entities. Output: filtered mask set + per-mask confidence |
| Indicator-extraction agents | MedAgent-Pro | Input: image/case. Output: named clinical indicator + value + tool-confidence |
| Contribution-aware aggregator | CoMMa | Input: set of per-agent embeddings/logits. Output: fused prediction + per-agent contribution weight |

**Gate:** each reimplemented module passes a standalone unit test reproducing the qualitative behavior reported in its source paper (e.g. CoMMa's aggregator must show naïve-mixture AUC below contribution-aware AUC on a toy input, matching the direction of its own ablation).

---

# Phase 3 — Module-level ablation

**Goal:** Hold three layers fixed, vary one layer at a time, using the fixed backbone and fixed task from Phase 0. This directly follows CoMMa's own ablation logic, generalized across all four layers.

**Ablation grid:**

| Fixed layers | Varying layer | Candidates swapped in |
|---|---|---|
| Evidence extraction, aggregation, output | Tool/specialist selection | MMedAgent router vs. fixed rule-based router vs. no router (single agent sees everything) |
| Tool selection, aggregation, output | Evidence extraction | CARE grounding vs. MedAgent-Pro indicator extraction vs. raw unstructured tool output |
| Tool selection, evidence extraction, output | Aggregation | CoMMa contribution-weighted vs. naïve majority vote vs. naïve averaging vs. single-agent (no aggregation needed) |
| All agentic layers | — | Full single-agent baseline with entire case in one context window (mandatory control) |

**Metrics per cell:** task accuracy/AUROC/F1 as appropriate, expected calibration error, latency, token/tool-call cost, and — critically — delta versus the single-agent control.

**Gate:** for every layer, at least one candidate must be shown to outperform the naïve baseline for that slot on the fixed task before it is eligible for Phase 4 assembly. A module that cannot beat a naïve baseline in isolation should not be promoted into the composite pipeline regardless of its reputation in its source paper.

---

# Phase 4 — Composite assembly

**Goal:** Assemble the pipeline using the empirically best module per slot from Phase 3, not the module assumed best from reading the papers.

```
[Input] → [Tool/Specialist Selection: winning module]
        → [Evidence Extraction: winning module]
        → [Aggregation/Fusion: winning module]
        → [Output/Explanation]
```

- Re-run the full composite against the single-agent control and against each "all-naive" baseline combination on the same fixed task.
- Document any interaction effects — a module that won in isolation (Phase 3) may not compose well with another winning module. If this occurs, run a small combinatorial sweep restricted to the top-2 candidates per slot.

**Gate:** the composite pipeline's accuracy/calibration/cost profile is fully characterized and compared against the single-agent control before any governance layer is added. Governance must never be used to compensate for a composite pipeline that is already worse than the single-agent baseline on raw task performance.

---

# Phase 5a — Privacy/compliance governance overlay

**Source:** `Workflow.md` (HIPAA-Compliant Architecture for Agentic Clinical AI Systems, reproducing Tripathi et al., CAIS '26).

**Required modification before attaching to the composite pipeline:** the original blueprint is text-only; it explicitly documents that image/DICOM payloads pass through its regex+BERT sanitizer as a no-op and still receive a logged ALLOW/compliant certificate. This gap must be closed, not inherited, because the composite pipeline from Phase 4 is inherently multimodal.

**Extension required:**

1. Implement the `input_gate` seam (already specified in `Workflow.md` Phase 9) as a real modality classifier, not a text-only no-op.
2. Add a parallel image/tool-output policy path: any non-text payload (raw image, segmentation mask, tool output) must be routed to its own authorization + audit path before it reaches the Tool Selection or Evidence Extraction layer of the composite pipeline.
3. Reuse the existing six-agent chain (Policy Decision → Sanitization → Middleware → Inference → Post-Redaction → Audit) for the text/report channel exactly as specified in `Workflow.md` Phases 3–8.
4. Extend the audit tuple schema to record which composite-pipeline module produced each output, so the audit log is traceable to a specific tool/module version, not just the final response.

**Gate (from Workflow.md Phase 11, extended):**
- All eight original validation checks from `Workflow.md` §11 pass.
- New check 9: a non-text payload (image, mask, tool output) fed into the composite pipeline is intercepted by the modality-aware gate and produces its own explicit ALLOW/DENY decision — it must never silently inherit a text-channel "compliant" certificate.

---

# Phase 5b — Evidence-reliability governance overlay

**Source:** `idea1_evidence_reliable_coordination_concept_note.md`.

**Goal:** Address the failure class privacy governance cannot see — a system can be fully HIPAA-compliant while still making a clinically wrong decision because it blindly trusted a faulty tool or dropped a modality.

**Required components, attached at the Aggregation/Fusion layer of the composite pipeline:**

1. Each module upstream of aggregation (tool selector, evidence extractor) must emit a structured evidence packet: claim, confidence, evidence reference, data-quality status, supporting/contradicting observations, recommended action (accept / re-check / abstain).
2. The aggregator (Phase 4 winning module, e.g. CoMMa-style contribution weighting) is extended to consume these structured packets rather than raw scores alone, so contribution weight can be conditioned on declared data quality and cross-modal consistency.
3. A missing-modality and tool-corruption test suite (per the concept note's stress-test table: complete data, missing modality, low-quality modality, tool error, cross-modal conflict, communication bottleneck, increasing agent count) is run against the composite pipeline with this overlay attached, and without it, to isolate its marginal effect.

**Gate:** the evidence-reliability overlay must show a measurable improvement in at least one of {calibration, error-cascade resistance, missing-modality robustness} relative to the composite pipeline without the overlay, on the same fixed task from Phase 0. If it shows no improvement, that is a valid, reportable negative result — do not adjust the test suite until it produces a positive result.

---

# Phase 6 — Joint evaluation harness

**Goal:** Evaluate the fully governed composite pipeline (Phase 4 + 5a + 5b) across four independent axes, using three complementary evaluation settings.

| Evaluation setting | Purpose | Source |
|---|---|---|
| Fixed static task/dataset (from Phase 0) | Primary accuracy/calibration comparison against single-agent control | Internal |
| AgentClinic (sequential, biased, multimodal NEJM cases) | Stress-test under interactive dialogue, limited information, and injected cognitive bias | AgentClinic benchmark |
| Privacy/adversarial suite (`Workflow.md` §10.4) | Ten adversarial attacks (Debug Mode, Translation, Emergency Bypass, etc.) mapped to STRIDE categories | `Workflow.md` |
| Missing/corrupted-modality stress suite (Phase 5b) | Coordination robustness under incomplete or unreliable evidence | Idea 1 concept note |

**Gate:** all four settings are run against (a) single-agent control, (b) composite pipeline with no governance, (c) composite pipeline with privacy governance only, (d) composite pipeline with both governance layers. This 4-way comparison is mandatory — it is the only way to attribute any observed improvement or regression to a specific layer.

---

# Phase 7 — Reporting

**Deliverable:** four independent scorecards per system variant (single-agent control, ungoverned composite, privacy-only composite, dual-governed composite):

1. **Clinical performance:** accuracy/AUROC/F1/C-index as task-appropriate, plus expected calibration error.
2. **Privacy/compliance:** Resid, Weighted Leak Score, severity-weighted attack success rate (per `Workflow.md` §10.2–10.4).
3. **Coordination robustness:** performance delta under missing modality, tool corruption, and cross-modal conflict (per Idea 1 concept note stress tests).
4. **Operational cost:** latency, token/tool-call count, GPU/API cost per case.

**Do not collapse these into one composite score.** A reviewer or supervisor needs to see, separately, whether governance improved safety at the cost of accuracy, latency, or both — collapsing the four axes into one number would hide exactly the trade-offs this project exists to characterize.

---

# 4. Mandatory safeguards throughout

- **Single-agent control is present in every phase from Phase 3 onward.** No claim of "composite pipeline improvement" is valid without it.
- **No governance layer may be used to mask a composite pipeline that underperforms the single-agent baseline on raw task accuracy.** Fix or abandon the pipeline first.
- **Privacy governance and evidence-reliability governance are evaluated and reported separately**, never merged into a single "trust score."
- **All five source-system claims used in module selection must be re-verified against the module's own reimplementation results (Phase 3), not cited directly from the original papers**, since backbone, dataset, and task differ across the five sources.
- **Patient-level data splits are mandatory** for any dataset used in Phase 0/3/4 to prevent leakage, consistent with standard practice for multimodal clinical benchmarks.

---

# 5. Dependency-respecting build order

```
Phase 0  → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5a (parallel) 
                                                  → Phase 5b (parallel) → Phase 6 → Phase 7
```

Phase 5a and 5b depend only on Phase 4 (the composite pipeline) and can be developed in parallel, since they are independent governance axes. Both must be complete before Phase 6's 4-way comparison.

**Fastest path to a first meaningful result:** Phase 0 → 1 → 2 (single donor module per slot, no ablation yet) → 4 (naive assembly) → 6 (single-agent control comparison only) — this gives an early sanity check before committing to the full Phase-3 ablation sweep.
