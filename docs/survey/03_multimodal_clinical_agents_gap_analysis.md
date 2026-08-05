# Critical Gap Analysis: Category 1.1 — Multi-Modal Clinical Agents
*Synthesized from 27 papers across 2024–2026 | Survey: Healthcare AI Agents*

> **Scope:** This analysis critically synthesizes the research/engineering gaps and future directions identified across all 27 papers catalogued in Category 1.1. Papers span multi-modal clinical diagnosis agents, agentic segmentation frameworks, multi-agent orchestration systems, automated ML pipelines, research scientists, and simulation environments. The goal is to expose the *systematic*, *recurring*, and *structurally deep* gaps that no single paper has solved — and to identify where the field must go next.

---

# GAP 1 — The Benchmark Validity Crisis: Static Proxies ≠ Clinical Reality

**Severity: Critical | Papers: MedOpenClaw, MDAgents, AgentDS, MAM, CARE, MedAgentSim, MedXIAOHE, Meissa**

The most pervasive and structurally dangerous gap across all 27 papers is the systematic mismatch between evaluation benchmarks and actual clinical deployment conditions.

# Evidence
- **MDAgents (NeurIPS Oral):** Evaluated on MedQA, PubMedQA — single-turn multiple-choice question answering designed for medical school examinations. Real clinical workflows require multi-turn, multi-modal, time-sensitive, incomplete-information decision making.
- **MedOpenClaw (arXiv 2603):** The most damning paper in the corpus. Agents scoring **90%+ on static 2D benchmarks collapse to ~42.1% 3D coordinate precision** when forced to navigate full clinical imaging software (3D Slicer). This is not a marginal drop — it is a complete failure of generalization.
- **MedAgentSim (arXiv 2503):** Maximum diagnostic accuracy on NEJM-derived benchmarks is **26–29%**, with interactions frequently left "unresolved" or "incomplete" within turn limits. This is clinical failure territory.
- **MAM (ACL Findings 2025):** DeepLesion accuracy of **35.1%**, PathVQA **47.6%**, MedQA **40.0%** — far below any clinically actionable threshold. Retrieved context was correct only **46.2%** of the time, and NIH dataset recall was **12.1%**.
- **AgentDS (arXiv 2602):** Fully automated agentic feature selection **degraded performance** by -0.014 F1 on average compared to human-curated features. Deep learning (LSTM/BERT) was *entirely unusable* on real datasets with N=1,000–5,000; the team had to fall back to gradient-boosted trees.
- **MedXIAOHE (arXiv 2602):** On sanitized benchmarks (MMMU_val-Med), scores 87.53%. On real-world in-house OCR with natural distortion, performance degrades significantly. On HLE-Med, scores **25.77%** vs. Gemini 3.0 Pro's **36.91%**.

# Root Cause
Current benchmarks reward **pattern matching on curated, well-cropped, label-clean, complete-modality, single-turn data** — which describes no real hospital system. Researchers optimize for leaderboard positions, not for clinical utility.

# Future Research Directions
1. **Prospective Clinical Trial Benchmarks:** Design benchmarks that simulate real clinical workflows — including missing modalities, noisy EHR data, partial results, time pressure, and multi-turn follow-up. MedFlowBench (MedOpenClaw) is the first step; the field needs a standardized version.
2. **Adversarial and Distribution-Shift Benchmarks:** Test agents on out-of-distribution hospital imaging systems, scanner vendors, patient demographics, and institutional reporting styles.
3. **Human-Physician Comparative Studies:** Benchmark clinical agents against primary care physicians, specialists, and MDTs on *prospective* patient cohorts — not retrospective curated datasets.
4. **Multi-turn, Incomplete-Information Diagnostics:** Benchmark suites where the agent must ask clarifying questions, order additional tests, and arrive at a diagnosis over multiple steps, mirroring real clinical practice.

---

# GAP 2 — The 2D→3D Chasm: Medical Imaging is Inherently Volumetric

**Severity: Critical | Papers: MedOpenClaw, MemSeg-Agent, IBISAgent, MedSAM-Agent, MedSAM3, 3DMedAgent, MACRO, MedVR**

With the exception of Cerebra (MRI), almost every imaging agent in this corpus operates in 2D slice space — despite the fact that clinical imaging (CT, MRI, 3D Echo, PET) is fundamentally 3D.

# Evidence
- **MemSeg-Agent (arXiv 2603):** Operates purely in 2D slice-wise manner, **explicitly discarding the Z-axis**. The authors acknowledge this as a core limitation yet the evaluation metrics do not penalize for it.
- **MedSAM3 (arXiv 2511):** On 3D volumetric LiTS (liver tumor segmentation), achieves a **Dice of 0.1374 vs. U-Mamba's 0.7910** — a catastrophic ~0.65 Dice absolute gap. On ISLES2024: **0.3033 vs. nnU-Net's 0.7718**.
- **IBISAgent (arXiv 2601):** Achieves Dice ~0.74–0.85 on hepatic vessel tumors with low-contrast boundaries vs. ~0.97 on simple organ (liver) segmentation. 3D reasoning is absent.
- **MedSAM-Agent (arXiv 2602):** Without fine-tuning, on CT: **Dice 0.122**; on MRI: **Dice 0.549** — both completely unusable. The agent "wanders" in volumetric search spaces.
- **3DMedAgent (arXiv 2602):** Despite the "3D" in the name, the Think-with-1-Slice (T1S) loop forces a **2D MLLM to reason from isolated slices**, failing "adjacent organ" relational questions that require holistic 3D context.
- **MedOpenClaw (arXiv 2603):** Agents entirely fail to maintain spatial context across 3D software GUI navigation — they get "lost" scrolling through slices.

# Root Cause
VLM architectures are image-patch-based and naturally operate on 2D tokens. Extending attention mechanisms to 3D volumes (hundreds of slices × full resolution) is computationally prohibitive. Most 3D approaches use slice sampling, which introduces blind spots and destroys inter-slice relationships critical for lesion boundary delineation and anatomical reasoning.

# Future Research Directions
1. **Native 3D Medical Foundation Models:** Architectures that process volumetric data natively — not as 2D slices — using 3D attention, 3D positional embeddings, or hierarchical 3D encoders (nnU-Net, SwinUNETR variants as backbones, extended to reasoning).
2. **Efficient Volumetric Transformers:** Memory-efficient 3D attention (linear attention, sparse attention, sliding window 3D) that allows VLMs to process full CT/MRI volumes without slice sampling.
3. **Agentic Slice Navigation Policies:** RL-trained agents that learn to navigate volumetric stacks by actively choosing which slices to examine based on their clinical relevance — rather than uniform slice sampling.
4. **Multi-View Aggregation with 3D-Aware Reasoning:** Fuse orthogonal 2D views (axial, coronal, sagittal) with explicit 3D positional encoding to recover volumetric context without full 3D computation.
5. **3D Grounding Benchmarks:** New benchmarks that test volumetric lesion localization with 3D coordinate precision — MedFlowBench is a start, but needs to become the standard.

---

# GAP 3 — Error Cascade in Tool-Orchestrated Architectures

**Severity: Critical | Papers: AURA, MedAgent-Pro, CARE, MAM, ToolSelect (085), MMedAgent, M^3Builder**

Multi-modal agents that delegate perception to specialized tools inherit all errors from those tools — without any mechanism to detect, correct, or compensate for tool failures.

# Evidence
- **MedAgent-Pro (arXiv 2503):** The Disc Hemorrhage (DH) indicator achieves only **66.8% mACC and 29.6% F1**. Including this weak indicator in the pipeline **actively harms** overall performance: -5.1% F1 for the MoE decider and **-16.5% F1** for the LLM decider. When the critical vCDR indicator is missing, F1 **plummets to 14.3%**.
- **AURA (arXiv 2507):** Explicitly acknowledges that if underlying tools (MedSAM, TorchXRayVision, PRISM, RadEdit) misclassify, AURA will "confidently propagate these errors." The counterfactual selection metrics (CPG, SIP, SSIM) are all worse than the best individual baseline tool.
- **CARE (arXiv 2603):** GPT-4o as coordinator was found to "over-edit" correct answers provided by the expert EG-VQA tool agent, introducing errors. BiomedParse as segmentation tool caused a **3.4% drop in downstream VQA performance**.
- **MAM (ACL 2025):** The retrieval agent achieves **12.1% recall on NIH** datasets. Even when context was correctly retrieved, the downstream answer accuracy was only **46.2%** — the pipeline cannot compensate for retrieval failure.
- **ToolSelect / Paper 085 (arXiv 2602):** The VQA routing gap vs. Oracle is **72.01% vs. 96.49%** — a **~24.5% absolute gap**. The model has no way to know its routing decision is wrong.
- **M^3Builder (arXiv 2502):** Completely agent-LLM-sensitive: Llama 3.3 or Gemini 2.0 reduce success rate to **4.29%** vs. Claude 3.7's ~90%.

# Root Cause
Tool-orchestrated agents treat each tool as a black box. There is no:
- **Tool reliability estimation** (does this tool work well for *this* input distribution?)
- **Failure detection** (has the tool returned a plausible but wrong output?)
- **Graceful degradation** (what to do when a critical tool fails or produces low-confidence output?)
- **Inter-tool consistency checks** (do tool outputs agree with each other?)

# Future Research Directions
1. **Tool Reliability Scoring in Context:** Learn a meta-model that estimates tool performance for a given input, routing to alternative tools when the primary tool is likely to fail (addressing the ToolSelect Oracle gap).
2. **Confidence-Calibrated Tool Outputs:** Require all tools to output calibrated uncertainty estimates. The orchestrator should down-weight or exclude tools with low confidence.
3. **Multi-Tool Cross-Verification:** When multiple tools analyze the same data (e.g., a diagnosis from a VQA tool and a segmentation tool), use agreement/disagreement as a signal for escalation to human review.
4. **Self-Consistency Loops with Revision:** Agents should run iterative consistency checks between tool outputs and the overall reasoning state, with targeted re-querying of failed tools.
5. **Graceful Degradation Protocols:** Agents must define explicit degradation strategies when critical tool outputs are unavailable or unreliable, rather than either confidently propagating errors or halting the pipeline.

---

# GAP 4 — Multi-Agent Orchestration: Coordination Overhead vs. Accuracy Gain

**Severity: High | Papers: MDAgents, MAM, CoMMa, Cerebra, MDAgents, CARE, CoMMa**

A fundamental assumption underlying multi-agent systems — that more agents produce better outcomes — is repeatedly contradicted by the empirical evidence in this corpus.

# Evidence
- **MDAgents (NeurIPS Oral):** Only outperforms baselines in **7/10 benchmarks**. The complexity classifier shows **low correlation with physician ratings** — the entire adaptive routing strategy rests on a flawed premise.
- **MAM (ACL Findings 2025):** Increasing expert roles beyond 3 **actively degrades performance** across datasets. Running debate rounds beyond 3 dilutes initially accurate diagnoses. The framework **overrides correct baseline predictions with incorrect ones** (Heartbeat drops from 70.2% to 64.0%).
- **CoMMa (arXiv 2602):** Transitioning from 1 deterministic agent to the full 5-agent system only improves HCC accuracy from **0.826 to 0.831** — a gain of 0.5%. On MTBBench, AUC is **0.595 ± 0.061** — barely above chance. Tumor recurrence AUC has **±0.204 standard deviation** across subsets.
- **CARE (arXiv 2603):** The coordinator (GPT-4o) actively harms performance by over-editing correct tool outputs. Using weaker models as coordinators breaks the entire pipeline.
- **Cerebra (arXiv 2603):** If a single modality (MRI) is missing, the **entire pipeline halts** — rigid multi-agent architecture with no graceful fallback.

# Root Cause
Multi-agent systems assume that agent roles are: (a) clearly separable, (b) independent, and (c) individually reliable enough that coordination adds value. None of these hold in practice for medical reasoning tasks, where:
- Clinical knowledge is highly interdisciplinary and not cleanly partitioned
- Individual agents hallucinate, and majority voting or debate amplifies rather than corrects errors
- The "orchestrator" or "coordinator" must itself be highly capable — often more capable than the individual agents it oversees

# Future Research Directions
1. **Dynamic Agent Pruning:** Research mechanisms to determine when multi-agent collaboration actually improves outcomes vs. when a single capable agent is superior, with automatic fallback.
2. **Game-Theoretic Debate with Adversarial Roles:** Rather than cooperative consensus, assign explicitly adversarial roles (one agent argues *against* each diagnosis) to stress-test reasoning, similar to clinical morbidity and mortality conferences.
3. **Evidence-Based Role Assignment:** Move away from domain-based role assignment (Neurologist Agent, Radiologist Agent) toward evidence-based task assignment, where the agent with the *most relevant training data* for the specific input handles that input.
4. **Missing Modality Robustness:** Mandatory robustness testing for partial modality scenarios. No production medical agent should halt when a modality is missing — it should re-route with explicit uncertainty flags.
5. **Agent Contribution Attribution at Inference:** Runtime tracking of which agent contributed to each claim, enabling post-hoc auditing of multi-agent failures (CoMMa proposes Shapley values, but only for training-time attribution).

---

# GAP 5 — Clinical Safety, Interpretability, and Trust Deficit

**Severity: Critical | Papers: Meissa, MDAgents, VERITAS, AURA, MedAgentSim, CoMMa, AgentDS, CARE**

Not a single paper in this corpus provides clinical-grade uncertainty quantification, principled abstention, or regulatory-aligned interpretability. This is the single most important gap for real-world deployment.

# Evidence
- **Meissa (arXiv 2603):** Explicitly lacks an abstention mechanism. When faced with out-of-distribution queries, the 4B model cannot "defer" to human experts — it will produce a confident-sounding but potentially wrong answer. Scores only **57.2% on MedQA** (vs. GPT-4V's 88.7%), meaning errors occur frequently with no mechanism to flag them.
- **MDAgents (NeurIPS Oral):** Multi-agent debate is a "black box" — the entire reasoning chain is hidden. Clinicians cannot trace *why* a diagnosis was made.
- **AgentDS (arXiv 2602):** No uncertainty quantification, no demographic stratification, no fairness analysis — the model may be systematically biased by race, age, or insurance status without any means of detection.
- **AURA (arXiv 2507):** Proxy metrics (CPG, SIP) for self-evaluation can be "gamed" — selecting images that satisfy mathematical criteria but introduce clinically unrealistic artifacts. No radiologist-in-the-loop validation.
- **CoMMa (arXiv 2602):** Provides quantitative attribution (Shapley values on logits) but no textual rationales. Clinicians require narratives, not percentile contributions, to trust AI decisions.
- **MedAgentSim (arXiv 2503):** Self-evolving memory buffers may silently accumulate hallucinations and biases — with no detection or correction mechanism.
- **VERITAS (arXiv 2604):** Addresses this partially — the explicit hypothesis-testing framework reduces hallucination rates from 19.8% to 4.2% — but only for 2D chest X-rays.

# Root Cause
This is a fundamental research gap, not an engineering oversight. The field lacks:
- Theoretical frameworks for what "safe medical AI" means in agentic settings
- Loss functions that penalize overconfident wrong answers
- Regulatory guidelines (FDA, CE, UKCA) for agentic AI in clinical decision support
- Evaluation metrics that measure *clinical safety*, not just accuracy

# Future Research Directions
1. **Calibrated Uncertainty in Agentic Medical AI:** Develop training procedures that produce well-calibrated confidence scores — where a confidence of 0.7 actually corresponds to 70% accuracy. Apply temperature scaling, conformal prediction, or Bayesian deep learning approaches.
2. **Principled Abstention Policies:** Design agents that can identify when they are operating outside their competency and explicitly refuse to answer, escalating to human clinicians. Define abstention thresholds based on case complexity and confidence.
3. **Regulatory-Aligned Interpretability:** Move beyond saliency maps to causal, contrastive, and counterfactual explanations that satisfy regulatory requirements. VERITAS's hypothesis-testing approach is a direction, but must be extended to all modalities.
4. **Fairness and Bias Auditing Infrastructure:** Standard benchmarks must include demographic subgroup performance reporting, and agents must be tested for performance parity across race, sex, age, and socioeconomic status.
5. **Clinical-Grade Audit Trails:** Every agentic reasoning step must be logged with sufficient granularity for post-hoc error investigation — similar to how surgical procedures are documented.

---

# GAP 6 — The Annotation Bottleneck vs. Data-Hungry Agents

**Severity: High | Papers: MedVR, Camyla, MemSeg-Agent, MedXIAOHE, MedSAM3, MACRO**

Medical AI systems require vast amounts of expert-annotated data. Agents cannot scale without solving the fundamental annotation problem — yet current solutions (RL-based weak supervision, synthetic data) introduce new failure modes.

# Evidence
- **MedVR (arXiv 2604):** RL-based weakly supervised localization achieves **68.4% IoU (>0.5)** vs. **71.2% for fully supervised** — a 2.8% gap that costs zero annotation budget. But RL training is "highly unstable and computationally expensive."
- **Camyla (arXiv 2604):** The autonomous AI scientist burns "massive amounts of GPU compute running failed experiments" and the coding agent sometimes introduces "bugs that accidentally improve the loss through data leakage" — a critical reproducibility concern.
- **MedSAM3 (arXiv 2511):** Constrained by "a lack of large-scale medical datasets with rich, high-quality clinical text descriptions" — the dataset limitation is the ceiling.
- **MedXIAOHE (arXiv 2602):** Mid-training and post-training pipelines rely heavily on LLM-synthesized data. In high-stakes environments, "reasoning derived from synthetic loops rather than ground-truth clinical reality remains susceptible to hallucination."
- **MACRO (arXiv 2603):** Composite tool quality is bounded by trajectory verification fidelity. **Noisy or partially correct traces introduce flawed diagnostic skills into the library** — bad data compounds over time.
- **Meissa (arXiv 2603):** ~40K filtered agentic trajectories synthesized using frontier models — essentially distilling knowledge from GPT-4V/Gemini into a 4B model, which caps performance at the teacher model's ceiling.

# Root Cause
Expert medical annotation is expensive (radiologists earn $300K+/year), slow (hours per case), and subject to inter-rater variability. Automated annotation generation using LLMs introduces synthetic distribution shift. RL-based learning without annotations requires computationally intensive training that is unstable in medical domains.

# Future Research Directions
1. **Active Learning for Medical Agents:** Agents should identify the specific data points where labeling would maximally improve performance and request targeted expert annotation — minimizing annotation cost per accuracy gain.
2. **Semi-Supervised Agentic Training:** Leverage the massive volumes of unlabeled medical images through self-supervised pretraining (e.g., DINO-med, masked image modeling on volumetric data), reducing labeled data requirements.
3. **Annotation-Free RL Scaling:** Improve stability of reward-based learning (MedVR approach) through better reward shaping, curriculum learning, and initialization from supervised pretraining. Reward model training from small expert datasets.
4. **Synthetic-to-Real Domain Adaptation:** Research robust methods for adapting agents trained on LLM-synthesized data to real clinical distributions, with explicit evaluation of synthetic-real gap.
5. **Foundation Model Fine-Tuning with Minimal Data:** Parameter-efficient fine-tuning (LoRA, adapters) on very small annotated datasets — the MemSeg-Agent's 2M parameter memory approach points toward this direction.

---

# GAP 7 — Computational Latency: Clinical Agents Are Too Slow

**Severity: High | Papers: IBISAgent, AURA, MedSAM-Agent, CARE, Meissa**

Multiple papers report inference latencies that are incompatible with real-time clinical workflows, particularly in emergency, surgical, or screening contexts.

# Evidence
- **IBISAgent (arXiv 2601):** Multi-step agentic loop takes **28.70 seconds per case** — unacceptable for radiological workflows where a radiologist reads 30–50 cases per day.
- **AURA (arXiv 2507):** Counterfactual generation loop (generate 5 images with diffusion models, pass through classifiers) introduces "significant latency" — practically impractical for time-sensitive workflows.
- **MedSAM-Agent (arXiv 2602):** Multi-turn RL loops cause "cumulative latency" that makes the agent "too slow for real-time surgical or emergency workflows."
- **CARE (arXiv 2603):** "Full CARE-Coord agent pipeline drastically increases inference time compared to the static CARE-Flow."
- **Meissa (arXiv 2603):** Standard Gemini-3-flash agents take **87.2 seconds** per query. Meissa reduces this to **4.1 seconds** — but achieves this by routing 72% of queries to direct inference (non-agentic), which limits its clinical generality.

# Root Cause
Agentic loops are fundamentally sequential: think → act → observe → think. Each tool call introduces latency, and multi-turn loops multiply this. Diffusion model calls, LLM reasoning chains, and remote API calls each add hundreds of milliseconds to seconds.

# Future Research Directions
1. **Speculative Agentic Execution:** Parallelize tool calls that are likely to be needed (based on query type) before they are explicitly invoked, amortizing latency.
2. **Adaptive Agentic Depth:** Route queries by complexity — simple queries get single-step inference (Meissa does this, at 72% direct routing), complex queries get full multi-step agentic reasoning. Define clinical thresholds for when agentic depth is justified.
3. **Distilled Fast-Inference Medical Agents:** Distill multi-step agentic reasoning into a single forward pass for common clinical scenarios (similar to chain-of-thought distillation), with full agentic loop reserved for novel or ambiguous cases.
4. **Edge-Deployable Medical Agents:** Design agents specifically for edge deployment in clinical settings (e.g., on hospital workstations) using quantization, pruning, and hardware-aware architecture search.
5. **Asynchronous Human-AI Collaborative Workflows:** For complex cases, design asynchronous agent workflows where the agent pre-analyzes a case in the background and presents findings when the clinician is ready, rather than requiring synchronous real-time inference.

---

# GAP 8 — Human-in-the-Loop: A Crutch, Not a Solution

**Severity: High | Papers: MemSeg-Agent, AgentDS, ToolSelect, CARE, MedOpenClaw, MDAgents**

Multiple papers implicitly or explicitly require human intervention as a core part of the pipeline — but present this as a feature rather than a limitation. In reality, if a clinical agent constantly requires expert input, it provides marginal value over existing human-only workflows.

# Evidence
- **MemSeg-Agent (arXiv 2603):** "Test-time working memory requires an oracle or human clinician to provide corrected annotations ($y_t^*$) when prediction fails." This fundamentally undermines the autonomous clinical agent premise.
- **AgentDS (arXiv 2602):** Required **10–15 hours of human effort per challenge** for feature engineering. Ranked **5th overall** even with this effort — the automated agent consistently underperformed with full automation.
- **ToolSelect / Paper 085 (arXiv 2602):** Authors explicitly state the system "is not capable of autonomous clinical decision-making" and "human-in-the-loop oversight is strictly required."
- **CARE (arXiv 2603):** Coordinator-based system fails with weaker models, requiring an expensive (GPT-4o-class) model to play the coordinator role — which itself introduces over-editing errors.
- **VERITAS (arXiv 2604):** "Requires the underlying VLM to perfectly understand spatial coordinates" — effectively requiring careful human-curated prompting.

# Root Cause
Current agents have not solved the core problem: **they do not know what they don't know**. Without calibrated uncertainty estimation and abstention capabilities, agents cannot reliably identify when to escalate to humans — leading to either over-reliance on human input (requiring expert intervention too frequently) or under-escalation (failing silently on cases they should defer).

# Future Research Directions
1. **Intelligent Escalation Frameworks:** Design clinical agents that autonomously determine when they need human input and provide the clinician with a structured, targeted query (not just "I'm not sure") — minimizing escalation frequency while maximizing the value of each human interaction.
2. **Workload-Aware Human-AI Teaming:** Model the clinician's cognitive load and current workload, escalating only when clinician capacity exists and deferring or auto-flagging otherwise.
3. **Human Feedback as Learning Signal:** Design agents that treat every human correction as a training signal, continuously improving their performance on the specific case types that require frequent escalation.
4. **Prospective Evaluation of Human-AI Team Performance:** Evaluate the combined human-AI team performance (not just AI-alone), measuring whether AI assistance actually improves physician accuracy, efficiency, and error detection vs. physician-alone.

---

# 🟡 GAP 9 — Semantic and Ontological Grounding Failures

**Severity: Medium-High | Papers: MedSAM3, MedXIAOHE, VERITAS, MemSeg-Agent**

Medical language is highly precise and structured. Models that treat medical terminology as approximate natural language fail in critical ways.

# Evidence
- **MedSAM3 (arXiv 2511):** Prompt change from 'cell' to 'nuclei' causes Dice to **drop from 0.6786 to exactly 0.0**. The model has no understanding that 'nuclei' is a sub-structure of 'cell.'
- **MedSAM3 (arXiv 2511):** Semantic misalignment causes the model to segment **lungs when prompted for 'liver'** — demonstrating fundamental anatomical ontology failures.
- **MedXIAOHE (arXiv 2602):** The Medical Entity Taxonomy (MET) has a **"Backward Gap" of 0.68–0.79** — the proprietary taxonomy captures many entities missing from public datasets, but risks poor transferability to standardized healthcare IT systems (ICD-10, SNOMED-CT).
- **MemSeg-Agent (arXiv 2603):** DINOv3 (trained on natural images) is used to measure domain distance for memory routing — fundamentally sub-optimal for medical image domains where visual semantics differ radically.

# Future Research Directions
1. **Medical Ontology-Grounded Reasoning:** Ground agent reasoning in standardized medical ontologies (SNOMED-CT, ICD-10, RadLex, NCI Thesaurus) so that the agent understands *semantic hierarchies* between medical terms, not just token similarity.
2. **Medical-Domain-Specific Embedding Spaces:** Replace natural-image-pretrained embeddings (DINO, CLIP) with medical-domain-specific embeddings trained on clinical data (CT, MRI, pathology slides) for routing, retrieval, and similarity computations.
3. **Compositional Medical Concept Understanding:** Train agents to decompose composite medical terms into their component concepts (e.g., 'perifocal edema around a ring-enhancing lesion' → spatial relation + lesion type + edema) and reason compositionally.

---

# 🟡 GAP 10 — The Research-to-Bedside Translation Gap

**Severity: Medium | Papers: Camyla, Towards Medical AI Scientist, AgentDS, M^3Builder, MedVR**

An emerging category — "Research Agents" that autonomously discover architectures, generate hypotheses, or design experiments — shows impressive results on controlled benchmarks but has zero validated translation to actual clinical impact.

# Evidence
- **Camyla (arXiv 2604):** Autonomously discovered an architecture with 0.932 Dice on cardiac MRI (vs. nnU-Net's 0.918). But: burns "massive GPU compute," introduces potential data leakage bugs, and the discovered architecture has not been validated in a clinical setting.
- **Towards Medical AI Scientist (arXiv 2603):** The agent "cannot physically run wet-lab experiments" — all hypotheses are purely theoretical with no validation pathway.
- **MedVR (arXiv 2604):** RL training is "highly unstable and computationally expensive compared to standard supervised learning" — not practically deployable for annotation-free learning at scale.
- **M^3Builder (arXiv 2502):** Requires carefully pre-configured "medical imaging ML workspaces" — real hospital data ecosystems are far messier and cannot simply be structured to fit the agent's requirements.

# Future Research Directions
1. **Automated Clinical Validation Pipelines:** Research agents should be coupled with automated validation frameworks (IRB-approved retrospective study designs, simulation environments, or synthetic patient cohorts) to evaluate discovered solutions before any human sees them.
2. **Reproducibility and Leakage Detection:** Research scientist agents must include automated data leakage detection, reproducibility verification, and ablation study execution as first-class components.
3. **Hospital-Realistic Data Environments:** Develop simulation environments that replicate the messiness of real hospital data (missing values, inconsistent formatting, DICOM metadata corruption) for evaluating research and deployment agents.

---

# 🟡 GAP 11 — Modality Completeness and Missing Data Handling

**Severity: Medium | Papers: Cerebra, AgentDS, MedAgent-Pro, MedXIAOHE**

Clinical patients frequently present with incomplete or missing modalities. Current multi-modal agents are brittle when expected inputs are absent.

# Evidence
- **Cerebra (arXiv 2603):** "If a patient is missing the MRI modality, the Neuroradiologist agent halts the entire pipeline rather than adapting."
- **AgentDS (arXiv 2602):** "TF-IDF worked for long discharge notes but completely failed for short daily progress notes" — a single feature engineering strategy cannot handle all clinical text types.
- **MedAgent-Pro (arXiv 2503):** When vCDR indicator is missing: F1 **plummets to 14.3%**. The LLM decider "fails to make a definitive diagnosis" when critical markers are unavailable.
- **MedXIAOHE (arXiv 2602):** Performance degrades significantly when encountering in-house data with "real-world distortion, blur, and clutter."

# Future Research Directions
1. **Missing-Modality Resilient Architectures:** Train multi-modal agents to gracefully degrade when modalities are absent, using marginalization, imputation, or uncertainty propagation to produce calibrated outputs even with incomplete inputs.
2. **Modality-Agnostic Patient Representations:** Develop patient state representations that are robust to missing modalities — encoding "what we know" rather than assuming "what we expect."

---

# Synthesis: Recurring Failure Mode Taxonomy

The following table cross-maps the 11 gaps to the 27 papers, revealing which failure modes are most endemic:

| Gap | Core Problem | # Papers Affected | Worst Quantitative Evidence |
|---|---|---|---|
| G1: Benchmark Validity | Static proxies ≠ clinical reality | 8 | MedOpenClaw: 90%→42.1% when moving to 3D GUI |
| G2: 2D→3D | Slice-based agents in volumetric domain | 8 | MedSAM3: LiTS Dice 0.14 vs. 0.79 |
| G3: Error Cascade | Tool failure propagation | 7 | MedAgent-Pro: DH indicator drops F1 by -16.5% |
| G4: Multi-Agent | Coordination overhead vs. accuracy | 6 | MAM: overrides correct answers (70.2%→64.0%) |
| G5: Safety/Trust | No uncertainty quantification | 7 | Meissa: no abstention, 57.2% MedQA |
| G6: Annotation | Data-hungry, annotation-scarce | 6 | MedVR: RL training unstable; Camyla: data leakage |
| G7: Latency | Too slow for clinical workflows | 5 | IBISAgent: 28.70s/case |
| G8: Human-in-Loop | Requires excessive human input | 5 | AgentDS: 10-15 hrs human effort, still ranked 5th |
| G9: Ontology | Semantic grounding failures | 4 | MedSAM3: 'nuclei' prompt → Dice drops to 0.0 |
| G10: Translation | No clinical validation pipeline | 4 | Camyla: discovered architectures unvalidated |
| G11: Missing Modalities | Brittle to incomplete inputs | 4 | Cerebra: pipeline halts if MRI is absent |

---

# Priority Future Research Directions (Consolidated)

# Tier 1: Foundational (Must be solved for any clinical deployment)
1. **Clinical-grade calibrated uncertainty + abstention mechanisms** for medical agents
2. **Standardized prospective clinical benchmarks** with missing modalities, noisy data, and multi-turn interactions
3. **Native 3D medical reasoning architectures** that process volumetric data without slice sampling
4. **Error cascade prevention**: tool reliability estimation and cross-verification protocols

# Tier 2: Engineering (High practical impact)
5. **Efficient multi-agent routing**: adaptive agent pruning based on demonstrated contribution
6. **Sub-second edge inference** for clinical agents via distillation, quantization, and speculative execution
7. **Missing modality robustness**: graceful degradation rather than pipeline halts
8. **Medical ontology grounding**: reasoning over SNOMED-CT/RadLex hierarchies rather than token similarity

# Tier 3: Longer-horizon Research
9. **Annotation-free RL at scale** with stable training and leakage detection
10. **Autonomous research-to-validation pipelines** with built-in reproducibility verification
11. **Longitudinal patient context modeling**: agents that maintain patient history across visits, not just single-encounter reasoning
12. **Fairness and demographic-stratified evaluation** as mandatory benchmark criteria

---

# Outstanding Missing Notes (2 Papers)

Two papers (`2603.05860` — *Evolving Medical Imaging Agents via Experience-driven Learning* and `2603.05873` — *Shifting Adaptation from Weight Space to Memory Space*) were analyzed via a dedicated analyst but their dedicated `notes.md` files will be created separately. Their findings align with Gap 2 (3D limitations), Gap 6 (annotation bottleneck), and Gap 8 (human-in-loop dependency) as identified through the folder notes.

---

*Analysis produced: 2026-07-02 | Corpus: 27 papers, Category 1.1 Multi-Modal Clinical Agents*
*Cross-references: survey_taxonomy.md, categorized_paper_catalog.xlsx, progress_tracker.md*
