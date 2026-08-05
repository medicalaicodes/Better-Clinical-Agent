# Evidence-Reliable Coordination for Multimodal Clinical Agents

# Concept Note and Benchmark Plan

**Working title:** *When Specialist Agents Fail to Collaborate: Benchmarking Evidence-Reliable Coordination in Multimodal Clinical AI*

**Research objective:** Determine when splitting clinical information across specialist AI agents helps, when it harms, and whether a coordinator that can inspect source evidence, represent uncertainty, and handle missing inputs can make multi-agent systems safer and more useful.

**Scope:** This is a research benchmark and method-development project. It is not a diagnostic product and it should not be presented as ready for clinical use.

---

# 1. The story begins with a reasonable idea

Modern healthcare data is naturally heterogeneous. A clinician may combine the patient history, vital signs, laboratory trends, imaging, radiology reports, pathology slides, molecular profiles, and prior treatment information before reaching a decision. It seems natural to mimic this workflow with a team of AI specialists: one agent reads the image, another interprets clinical text, another examines longitudinal EHR data, and another handles molecular or genomic information.

The coordinator would then combine the specialists' opinions. This architecture is attractive for practical reasons: data are often stored in separate hospital systems, each modality may need a different model or tool, and specialists might make explanations easier to audit.

However, this intuitive design has an important unresolved problem. A multi-agent pipeline does not automatically create better multimodal understanding. It can instead turn rich evidence into short, incomplete summaries, and then ask a coordinator to make a high-stakes decision from those summaries.

---

# 2. The open gap: coordination is the weak link

Our survey of multimodal clinical-agent work found repeated warning signs.

- In the cited MAM analysis, extra debate could overwrite initially correct answers and reduce accuracy from approximately 70 to 64.
- In the cited CoMMa result, increasing from one to five agents improved AUC by only 0.5 points, raising a serious cost-versus-benefit question.
- In MedAgent-Pro, adding a weak disc-haemorrhage tool reportedly reduced F1 by 16.5 points. This is a concrete example of an error from one component harming the entire pipeline.
- Cerebra reportedly stops when MRI is unavailable, despite missing data being normal in real clinical workflows.
- CARE reportedly shows that a coordinator can modify a correct specialist output into an incorrect final response.

These results do **not** prove that all multi-agent designs are bad. They do show that naïve coordination—simple voting, fixed debate, averaging scores, or exchanging only free-text summaries—cannot be assumed to be reliable. The scientific gap is not merely “do more agents help?” It is:

> How can a multi-agent clinical system preserve, check, and appropriately use heterogeneous evidence when a modality is missing, a tool is unreliable, or specialists disagree?

# Related external evidence

AgentRx is especially important because it directly evaluates this question in a real-world ICU prediction setting. It compares single-agent and multi-agent approaches across patient summaries, structured EHR, chest X-rays, and radiology reports. Its reported result is that a unified single agent outperformed the naïve multi-agent approaches and was better calibrated; its strongest multi-agent configuration retained raw multimodal input for the final decision-maker rather than relying only on decentralized summaries. Source: [AgentRx](https://arxiv.org/html/2605.10286v1).

AgentRx also found that a dedicated supervised multimodal model outperformed the evaluated agent designs. This means the project must not claim that an agent is automatically the best predictive model. Instead, it should evaluate whether agentic coordination earns its added cost through robustness, provenance, safe uncertainty handling, and operational flexibility.

---

# 3. Why this matters clinically

A clinical AI system can fail even when every individual component looks strong in isolation.

Imagine a cancer case. A pathology model sees patterns compatible with aggressive disease. A gene-expression model identifies a molecular profile that suggests a different risk pattern. The clinical record is incomplete because treatment details are missing. A weak coordinator may average the agents' confidence scores, force a consensus, or silently ignore the difficult modality. It may then give a confident conclusion without showing which evidence it trusted or why.

A safer system should do something different. It should preserve the disagreement, identify that the evidence is incomplete or conflicting, inspect relevant source evidence when needed, and communicate uncertainty or request review rather than fabricate agreement.

This is not only an accuracy problem. It is a **reliability, calibration, provenance, and graceful-degradation** problem.

---

# 4. Proposed idea

We propose an **Evidence-Reliable Coordinator (ERC)** for multimodal clinical agents.

The ERC is not a new foundation model. It is a coordination layer placed between modality-specific specialists and the final clinical decision. It is designed to prevent the coordinator from treating every specialist message as equally trustworthy or equally complete.

# Core design principles

1. **Keep evidence connected to claims.** Each specialist must return a structured claim and point to the source evidence that supports it. For example, the pathology specialist should identify slide regions or extracted features; the genomics specialist should identify pathways, genes, or molecular features; the clinical specialist should identify the relevant fields and timestamps.

2. **Represent uncertainty explicitly.** A specialist should be allowed to say that evidence is insufficient, low quality, unavailable, or contradictory. The coordinator should not convert uncertainty into a forced diagnosis.

3. **Allow source-evidence re-checking.** The final coordinator should be able to inspect selected original evidence or verified feature summaries. It should not receive only an untraceable narrative from specialist agents.

4. **Treat disagreement as a signal.** Specialists need not agree. The coordinator should distinguish between an explainable difference, an unresolved conflict, and a likely unreliable output.

5. **Degrade gracefully.** When pathology, genomic, or clinical data are absent, corrupted, or delayed, the system should continue with the available information while reducing confidence, documenting the missing input, or abstaining when necessary.

6. **Separate prediction from explanation.** An explanation is useful only if it remains tied to evidence that the system actually used. The benchmark should test this connection rather than reward persuasive prose.

---

# 5. What each agent returns

Instead of allowing agents to exchange unconstrained prose, each modality specialist returns a small, inspectable evidence packet:

- **Clinical claim:** what the modality suggests for the task
- **Confidence and uncertainty:** how certain the specialist is and why
- **Evidence references:** identifiers for source fields, image patches, reports, features, or pathways
- **Data-quality status:** complete, incomplete, out-of-distribution, missing, or suspected corruption
- **Supporting and contradicting observations:** evidence that supports the claim and evidence that weakens it
- **Recommended next action:** accept, re-check, request another modality, or abstain

The coordinator reviews these packets. It may accept a claim, request a targeted re-check, down-weight an unreliable component, preserve a conflict in the final report, or abstain/escalate.

The research contribution is not that this structure resembles a human multidisciplinary team. The contribution is whether it produces measurable improvements under controlled stress tests.

---

# 6. Research questions

# Primary question

Can evidence-reliable coordination make multimodal multi-agent systems more robust than naïve voting or debate when data are incomplete, noisy, contradictory, or distributed across modalities?

# Secondary questions

- Does a coordinator that can access source evidence outperform one that sees only specialist summaries?
- Can the system identify and discount a weak or corrupted tool output without discarding genuinely useful modality-specific evidence?
- Does the system maintain useful calibration as modalities are added or removed?
- Can it state when it lacks enough evidence instead of emitting an overconfident prediction?
- What accuracy, calibration, cost, and latency trade-off is introduced by more reliable coordination?
- Does multi-agent specialization outperform a strong unified agent, and if so, under which data conditions?

---

# 7. Benchmark philosophy

The benchmark should not reward a system only for its score on clean, complete cases. It should evaluate the behavior that makes multimodal coordination difficult in real settings.

# Required test conditions

| Stress condition | What changes | What a reliable system should do |
|---|---|---|
| Complete data | All modalities available | Use complementary information without unnecessary overhead |
| Missing modality | Remove pathology, omics, image, report, or clinical fields | Continue safely, disclose absence, reduce confidence or abstain when needed |
| Low-quality modality | Blur/mask image regions, reduce slide quality, introduce incomplete clinical data | Detect reduced trustworthiness and avoid false certainty |
| Tool error | Inject known incorrect, noisy, delayed, or low-confidence specialist outputs | Re-check or discount the unreliable output rather than propagate the error |
| Cross-modal conflict | Make two modalities provide plausible but conflicting signals | Preserve and resolve conflict with evidence; do not simply average or suppress it |
| Communication bottleneck | Give coordinator only summaries versus summaries plus retrievable source evidence | Quantify information loss from decentralized communication |
| Increasing agent count | Add specialists or debate rounds | Measure marginal utility against accuracy, calibration, latency, token use, and cost |

# Metrics to report

- **Task performance:** AUROC and AUPRC for binary prediction; C-index and time-dependent survival metrics for survival tasks; appropriate pathology classification metrics when relevant.
- **Calibration:** Expected calibration error and Brier score, with reliability plots.
- **Selective safety:** performance after allowing the system to abstain or request review on uncertain cases.
- **Missing-data robustness:** performance and calibration as each modality is removed or degraded.
- **Error-cascade resistance:** how much final performance drops when one specialist or tool is deliberately corrupted.
- **Evidence fidelity:** whether the final answer references source evidence that was actually provided and relevant to the decision.
- **Efficiency:** end-to-end latency, number of specialist calls, token consumption, GPU time, and API cost.

All comparisons should use identical cases, tasks, prompts where possible, model backbones, and evaluation splits. The experiment must include confidence intervals or repeated seeds where feasible.

---

# 8. Benchmark arms

The project should compare architectures rather than evaluate only the proposed method.

| System | Role in study |
|---|---|
| Supervised multimodal predictor | Strong non-agent reference; shows whether agentic complexity is justified |
| Unified multimodal agent | Strong agent baseline with all available information in one context |
| Specialist agents + majority vote | Naïve decentralized baseline |
| Specialist agents + free-text debate | Tests whether discussion alone solves coordination |
| Specialist agents + summary-only coordinator | Direct communication-bottleneck baseline |
| Specialist agents + ERC | Proposed evidence-reliable method |

The most important comparison is not “ERC versus a weak baseline.” It is **ERC versus a unified agent and a strong supervised multimodal model under the same cases and perturbations.**

---

# 9. Available datasets and benchmarks

# A. AgentRx: immediate reproducibility baseline

**What it is:** An open evaluation framework for multimodal clinical risk prediction with agents. It was created specifically to compare unimodal, unified multimodal, and specialized multi-agent settings.

**Source datasets:** MIMIC-IV, MIMIC-CXR, and MIMIC-IV-Note. AgentRx aligns these sources at patient, admission, and ICU-stay level. Its four modalities are patient summaries, structured EHR time series, chest X-ray images, and radiology reports. Source: [AgentRx paper and code](https://arxiv.org/html/2605.10286v1), [GitHub repository](https://github.com/nyuad-cai/AgentRX).

**Tasks:**

- In-hospital mortality prediction using information from the first 48 hours in the ICU.
- Long length-of-stay prediction, defined as a stay longer than seven days, again using the first 48 hours.

**Scale reported by AgentRx:** For mortality, it reports 17,773 training and 4,925 test patient-summary/EHR samples, 16,128 training and 4,454 test radiology-report samples, and 4,259 training and 1,174 test CXR samples. For length of stay, it reports 17,476 training and 4,845 test patient-summary/EHR samples, 15,865 training and 4,380 test radiology-report samples, and 4,171 training and 1,153 test CXR samples. The lower CXR count creates real-world partial-modality conditions.

**Reported baselines:** AgentRx evaluates supervised models, single agents, majority voting, debate, meta-prompting, Traj-CoA, MDAgents, and MedAgents. On its four-modality mortality task, it reports MedPatch at AUROC 0.877; the best reported agent configuration at 0.763; Qwen single-agent zero-shot at 0.756; Qwen debate at 0.631; and Qwen majority voting at 0.748. These results make AgentRx the most direct baseline for the proposed project.

**Strengths for this project:**

- Exact match to the single-agent versus multi-agent research question.
- Includes image, text, time-series EHR, and report modalities.
- Already provides modality-addition ablations, calibration reporting, public code, and multiple agent baselines.
- Naturally supports communication-bottleneck and missing-modality experiments.

**Limitations:**

- ICU risk prediction, not oncology or pathology-genomics.
- MIMIC access requires credentialed PhysioNet access and completion of the applicable data-use training/agreement.
- Its CXR subset is much smaller than the EHR/text cohort, so pairing strategy and missingness must be reported carefully.
- It does not provide an omics modality.

**Recommended role:** Phase 1. Reproduce a small AgentRx subset first, then add controlled tool-noise, source-evidence access, and missing-modality coordination tests.

---

# B. TCGA / GDC: pathology, genomics, and clinical information

**What it is:** The Cancer Genome Atlas (TCGA), accessible through the NCI Genomic Data Commons (GDC), is the most appropriate public resource for extending the project to pathology–genomics–clinical coordination.

**Available data types:** TCGA contains clinical and biospecimen information; diagnostic and tissue whole-slide images; gene expression; somatic mutation calls; copy-number data; methylation; miRNA; and protein-expression measurements. The NCI notes that what is available varies by cancer type, sample quality, quantity, and the technology used. Source: [NCI TCGA data-type documentation](https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data/types).

**Image formats and access:** Diagnostic and tissue whole-slide images are available as SVS files through GDC and are open access. Selected pre-surgical radiology scans are available through The Cancer Imaging Archive in DICOM format. The GDC portal supports browsing and downloading, and larger collections can be downloaded through a manifest and the GDC Data Transfer Tool. Source: [GDC Data Portal](https://gdc.cancer.gov/access-data/gdc-data-portal).

**Scale and linkage:** NCI describes TCGA as containing data from more than 20,000 tumour and normal samples. ISB-CGC documentation reports more than 30,000 TCGA tissue slide images in an open Google Cloud bucket, with metadata that includes TCGA case and sample barcodes that can be joined to clinical, biospecimen, and molecular tables. Source: [ISB-CGC TCGA images documentation](https://isb-cancer-genomics-cloud.readthedocs.io/en/latest/sections/data/TCGA-images.html).

**Good initial cohorts:**

- **TCGA-BRCA:** large and clinically familiar, but biologically heterogeneous; useful for subtype and survival/risk studies.
- **TCGA-LUAD and TCGA-LUSC:** a practical lung-cancer pair for histology, clinical variables, and molecular profiles.
- **TCGA-COAD and TCGA-READ:** useful related colorectal cohorts, with prior pathology–genomics survival work.
- **TCGA-KIRC:** commonly used in computational pathology and survival modelling.

Final cohort selection should be based on a pre-analysis count of *patients with all target modalities*, not on nominal cohort size.

**Strengths for this project:**

- Directly addresses the pathology + genomics extension that AgentRx lacks.
- Supports realistic modality absence because not every case has every assay or usable slide.
- Enables clinically meaningful source provenance: slide patches, pathway/gene features, and clinical fields can be logged separately.
- Well aligned with pathology and medical-imaging expertise.

**Limitations and risks:**

- TCGA is retrospective and multi-institutional, not a live clinical workflow.
- Survival labels involve censoring and require survival-aware analysis, not a simplistic binary label unless clearly justified.
- Whole-slide images are very large; do not send raw slides to a language model. Use a pretrained pathology encoder to produce tile/region embeddings and retain tile identifiers for provenance.
- Omics has very high dimensionality. Start with biologically meaningful pathway scores or a fixed, predeclared gene panel rather than thousands of raw genes in an agent prompt.
- Sample pairing, slide selection, assay preprocessing, batch effects, and leakage prevention are major engineering tasks.
- Some raw or sensitive genomic files are controlled access; derived data are generally open access, but every selected file type must be checked in GDC before project planning.

**Recommended role:** Phase 2. Use TCGA to test whether evidence-reliable coordination transfers from ICU data to pathology–omics settings.

---

# C. AgentClinic: sequential clinical-agent stress testing

**What it is:** An open simulated clinical environment in which a doctor agent obtains information through dialogue and active data collection rather than receiving a fully summarized case at once.

**Available suites:** The public repository describes extended MedQA cases (215), extended NEJM cases (120), and an AgentClinic-MIMIC-IV suite based on real MIMIC-IV cases; MIMIC-IV access requires approval. Source: [AgentClinic GitHub](https://github.com/SamuelSchmidgall/AgentClinic).

**Modalities and behavior:** AgentClinic includes dialogue-based diagnosis, a multimodal image-and-dialogue environment for NEJM cases, and simulation of cognitive and implicit biases in doctor/patient agents. Its central value is sequential interaction, tool use, limited information, and the possibility of testing how a model revises decisions over time. Source: [AgentClinic website](https://agentclinic.github.io/).

**Strengths for this project:**

- Useful for testing whether the coordinator behaves safely when evidence is revealed in stages.
- Supports early misleading cues, later corrective evidence, limited interactions, and tool-use conditions.
- Provides a complementary operational test: a method that looks good on static TCGA or AgentRx cases may fail in a sequential setting.

**Limitations:**

- It is mainly a simulated dialogue environment, not a pathology–genomics cohort.
- It is best used as a stress-test extension, not as the primary dataset for the pathology–omics version of the project.
- Simulated patients and dialogues may introduce model-dependent artifacts; report the patient-agent model and simulation configuration.

**Recommended role:** Optional Phase 3. Test temporal evidence reconciliation after the core static multimodal benchmark is complete.

---

# D. Supporting supervised baselines and resources

| Resource | Use in this project | Caveat |
|---|---|---|
| MedPatch / AgentRx baselines | Supervised and agentic reference implementations for ICU multimodal prediction | Verify code, licensing, and exact preprocessing before comparison |
| PathOmics / PONET-style TCGA studies | Design references for pathology–genomics survival tasks and feature preparation | They are task-specific deep-learning approaches, not agent coordination benchmarks |
| GDC Data Portal | Discovery, manifests, metadata, clinical/molecular download, SVS slides | Confirm access level and use patient-level joins to prevent leakage |
| ISB-CGC / Google Cloud resources | Efficient metadata exploration and scalable TCGA image access | Requires cloud/data-governance planning |
| PhysioNet MIMIC-IV, MIMIC-CXR, MIMIC-IV-Note | AgentRx reproduction data sources | Credentialing, data-use agreement, and secure handling are required |

---

# 10. Recommended phased implementation

# Phase 0: lock the claim

Write a one-page protocol before coding:

- Define one primary task and one primary metric.
- State that the project tests coordination reliability, not autonomous diagnosis.
- Pre-register the architecture arms, perturbations, stopping rule, and intended metrics.
- Build a source audit table for every baseline number cited in the proposal.

# Phase 1: reproduce the known gap on AgentRx

- Run a manageable subset with one open multimodal backbone.
- Compare unified single agent, majority-vote specialists, and a summary-only coordinator.
- Verify whether the single-agent / naïve multi-agent gap is reproduced under your compute budget.
- Add modality removal and a controlled wrong-tool-output experiment.

**Success condition:** reproduce or clearly explain differences from AgentRx before proposing a new method.

# Phase 2: implement ERC

- Replace free-text exchange with structured evidence packets.
- Give the final coordinator controlled retrieval of source evidence or verified modality features.
- Add explicit missingness, low-quality, and uncertainty fields.
- Compare ERC with the exact same backbone and prompts as the naïve baselines.

**Success condition:** ERC improves at least one robustness property without an unacceptable reduction in clean-case performance or an excessive latency increase.

# Phase 3: transfer to TCGA

- Start with a single cancer cohort and a single well-defined endpoint.
- Use pathology-slide embeddings, pathway/gene-expression representations, and a small clinical feature set.
- Build patient-level matching and split before feature generation.
- Run complete-data, missing-modality, corrupted-modality, and conflict conditions.

**Success condition:** show whether the method generalizes beyond ICU data, not merely that it achieves a high score on one cancer dataset.

# Phase 4: optional sequential validation

- Use AgentClinic to reveal evidence in stages.
- Test whether ERC preserves uncertainty and revises decisions after corrective evidence.
- Do not treat a sudden opinion change as inherently unsafe; evaluate whether the revision is supported by new evidence.

---

# 11. Critical safeguards

# Avoid forced agreement

Pathology, molecular data, and clinical history can legitimately point in different directions. The system should report unresolved discordance rather than being rewarded for producing a single smooth narrative.

# Avoid attention-based claims

Do not claim that raw attention weights prove model reliance. Evaluate behavior with controlled evidence removal, corruption, and source retrieval instead.

# Avoid cross-paper leaderboard claims

Dice, AUC, F1, diagnostic accuracy, latency, and tool-use success are not interchangeable. Use only matched in-benchmark comparisons for primary claims.

# Control data leakage

Patient-level splits are mandatory. For TCGA, slides from the same patient must never cross train/test boundaries; survival endpoint definitions must respect time and censoring; feature-selection choices must be made using training data only.

# Keep a non-agent baseline

A high-quality supervised multimodal model must remain in the study. If it is more accurate, faster, and better calibrated, report that honestly. The agentic system may still add value through provenance, flexible modality handling, or safe escalation—but those benefits must be measured rather than assumed.

---

# 12. Expected contribution

A successful project would deliver:

1. A controlled benchmark for testing multi-agent clinical coordination under missing, noisy, conflicting, and compressed multimodal evidence.
2. A fair comparison among unified agents, naïve multi-agent systems, supervised fusion models, and evidence-reliable coordination.
3. A reproducible intervention—the structured evidence packet plus source-evidence retrieval and uncertainty handling—not a claim of a new medical foundation model.
4. A pathology–genomics extension using TCGA that complements AgentRx's ICU EHR/CXR/report setting.
5. Public evaluation scripts, configuration files, perturbation generators, data manifests where permitted, and full accuracy/calibration/cost/latency reporting.

The central contribution is a falsifiable claim: **multi-agent specialization is useful only when the system can preserve evidence, recognize unreliable inputs, handle missing modalities, and expose unresolved disagreement.**

---

# References and access links

1. AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks — https://arxiv.org/html/2605.10286v1
2. AgentRx code — https://github.com/nyuad-cai/AgentRX
3. NCI: TCGA Data Types Collected — https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data/types
4. NCI Genomic Data Commons Data Portal — https://gdc.cancer.gov/access-data/gdc-data-portal
5. ISB-CGC: TCGA Radiology and Pathology Image Data Set — https://isb-cancer-genomics-cloud.readthedocs.io/en/latest/sections/data/TCGA-images.html
6. AgentClinic benchmark — https://agentclinic.github.io/
7. AgentClinic code and release information — https://github.com/SamuelSchmidgall/AgentClinic
8. User-provided internship presentation, *AI Agents in Healthcare Internship Research Presentation*, IIT Mandi, July 2026.
