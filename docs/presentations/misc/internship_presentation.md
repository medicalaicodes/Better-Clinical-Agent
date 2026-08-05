# AI Agents in Healthcare — Internship Research Presentation
# IIT Mandi | July 2026

> **Internship Task:** *"Finding the relevant domains/tasks for which AI Agents in Healthcare are used for, evaluating the SOTA models for those, identifying the gaps, and benchmarking their performance based on metrics — for different modalities selectively (EHR, clinical notes, omics, imaging like X-Ray, MRI, etc.)"*

---

# SLIDE 1 — What I Was Asked To Do

**Task (Given by Supervisor):**
- Map the landscape of **AI Agents in Healthcare**: What domains are they being applied to?
- Evaluate the **State-of-the-Art models** in each domain
- Identify **open research & engineering gaps**
- Benchmark performance across **modalities** — imaging (X-Ray, MRI, CT), EHR, clinical notes, genomics, etc.

**My Approach:**
- Conducted a systematic literature survey of **~300+ papers** from 2023–2026
- Organized them into a **structured taxonomy** of 13 doctor-facing + 4 patient-facing + 3 infrastructure categories
- Performed **deep critical analysis** of the most active sub-field (27 papers fully annotated)
- Identified **11 systemic failure modes** from primary evidence

---

# SLIDE 2 — The Research Taxonomy I Built

I categorized all papers into the following hierarchy. Each category represents a distinct clinical use-case where agentic AI is being deployed or studied:

# **1. Doctor-Facing Agents** *(Primary Focus)*

| # | Category | Papers | Key Modalities |
|---|---|---|---|
| 1.1 | **Multi-Modal Clinical Agents** *Deep-analyzed* | 27 | X-Ray, MRI, CT, Pathology, EHR, Text |
| 1.2 | **Radiology Agents** (CT, X-ray, MRI) | 30 | CT, X-Ray, MRI, PET |
| 1.3 | **Pathology Agents** | 12 | Whole-slide images, Histology |
| 1.4 | **Cardiovascular Imaging** | 2 | Echocardiography, ECG |
| 1.5 | **Sonography / Ultrasound** | 3 | Ultrasound |
| 1.6 | **Radiotherapy** | 2 | CT, Treatment Plans |
| 1.7 | **Dermatology** | 2 | Skin Images |
| 1.8 | **Dental Agents** | 2 | Panoramic X-ray |
| 1.9 | **Genomics & Biomarker Agents** | 7 | Genomic sequences, Omics |
| 1.10 | **EHR & Clinical Note Agents** | 30 | Structured EHR, Free text |
| 1.11 | **Surgical Agents** | 2 | Surgical video, OR workflow |
| 1.12 | **Education Agents** | 6 | Multi-modal |
| 1.13 | **Reasoning & Multi-Agent Techniques** | 50+ | Text, Multi-modal |

# **2. Patient-Facing Applications**

| # | Category | Papers |
|---|---|---|
| 2.1 | **Mental Health & CBT Agents** | 13 |
| 2.2 | **Clinical Communication & Intake** | 9 |
| 2.3 | **Screening & Personalized Care** | 8 |
| 2.4 | **General-purpose Healthcare Avatars** | 2 |

# **3–6. Supporting Infrastructure**
- **Drug Discovery & Development** — 17 papers
- **Healthcare Administration & Workflow** — 22 papers
- **Datasets & Benchmarks** — 35 papers
- **Related Surveys** — 26 papers

> **Total curated:** ~300+ papers (2023–2026), spanning NeurIPS, ICML, ACL, EMNLP, ICLR, MICCAI, Nature, npj Digital Medicine, and arXiv.

---

# SLIDE 3 — Why I Started with Category 1.1

**Category 1.1 — Multi-Modal Clinical Agents** is the most active and representative sub-field because:

- It sits at the **intersection of all modalities**: imaging, text, EHR, genomics, audio/video
- It contains the most **agentic AI innovations** of 2025–2026 (RL-trained agents, tool-use, multi-agent orchestration, self-evolution)
- It exposes the **deepest gaps** that propagate to all other categories
- It has the clearest **connection to clinical deployment**: radiology copilots, diagnostic agents, MDT simulators

**I fully analyzed all 27 papers** — reading primary papers, extracting quantitative benchmarks, annotating gaps, synthesizing findings.

---

# SLIDE 4 — Category 1.1: What Are These Papers Doing?

The 27 papers address **5 distinct sub-problems**:

# 4a. Multi-Modal Diagnosis Agents
*Agents that reason over images + text + clinical context to produce diagnoses*
- **MDAgents** (NeurIPS 2024 Oral): Adaptive LLM collaboration — routes tasks to solo or group of agents based on complexity
- **MMedAgent** (EMNLP 2024): First agent to "learn to use" 6 specialized medical tools across 5 imaging modalities
- **MAM** (ACL 2025): Modular multi-agent framework with role-specialized collaboration
- **MedAgent-Pro** (arXiv 2025): Evidence-based diagnosis using indicator-specific tool agents (vCDR, disc hemorrhage)
- **Meissa** (arXiv 2026): Offline 4B-parameter agent trained via stratified RL trajectories; 22× faster than GPT-4V agents

# 4b. Medical Image Segmentation Agents
*Agents that iteratively refine segmentation via RL, memory, or SAM-based tools*
- **IBISAgent** (arXiv 2026): Pixel-level visual reasoning using agentic MedSAM2 loop
- **MedSAM-Agent** (arXiv 2026): Multi-turn agentic RL with reward shaping for interactive segmentation
- **MedSAM3** (arXiv 2025): Text-guided SAM extension for medical concept-driven segmentation
- **MemSeg-Agent** (arXiv 2026): Shifts adaptation from weight-updates to memory tokens on frozen SAM2
- **MedEyes** (arXiv 2025): Dynamic visual focus via gaze-guided reasoning navigator

# 4c. Multi-Agent Orchestration & Collaboration
*Coordinating multiple specialized agents on complex clinical tasks*
- **Cerebra** (arXiv 2026): AI MDT board — Neurologist + Neuroradiologist + Geneticist agents for dementia risk
- **CoMMa** (arXiv 2026): Game-theoretic (Shapley value) contribution attribution across 5 oncology agents
- **CARE** (arXiv 2026): Evidence-grounded VQA with coordinator + segmentation tool for clinical accountability
- **3DMedAgent** (arXiv 2026): Unified perception-to-understanding pipeline for 3D volumetric scans

# 4d. Self-Evolving & Adaptive Agents
*Agents that improve their own skills and tool repertoire through experience*
- **MACRO** (arXiv 2026): Mines successful trajectories to auto-discover composite tools; GRPO RL training
- **MedAgentSim** (arXiv 2025): Self-evolving multi-agent simulations of clinical consultations
- **Camyla** (arXiv 2026): Autonomous AI Scientist that discovers novel segmentation architectures via code loops
- **VERITAS** (arXiv 2026): Verifiable epistemic reasoning via hypothesis testing with targeted visual tools

# 4e. Benchmarks, Evaluation & Recipe Papers
*Frameworks for evaluating and building medical AI agents*
- **MedOpenClaw + MedFlowBench** (arXiv 2026): First benchmark requiring agents to navigate real 3D clinical software
- **MedXIAOHE** (arXiv 2026): Comprehensive recipe for building medical MLLMs with Medical Entity Taxonomy
- **ToolSelect** (arXiv 2026): Attentive Neural Process-based model selection router
- **Human-Guided AgentDS** (arXiv 2026): Human guidance beats pure automation on real EHR challenges
- **AURA** (MICCAI 2025): Counterfactual self-evaluation via diffusion image generation

---


---
# Detailed Breakdown of the 27 models
# 4a. Multi-Modal Diagnosis Agents

**1. MDAgents (NeurIPS 2024 Oral)**
*   **What it is:** A dynamic routing framework for medical decision-making.
*   **How it works:** It uses a "complexity classifier." If it determines a clinical query is simple, it assigns it to a single LLM to save time and compute. If it determines the query is complex, it triggers a multi-agent collaboration protocol where several LLMs debate to reach a consensus.
*   **The Catch:** Your gap analysis found that its complexity classifier has low correlation with actual physician ratings, meaning it often routes tasks incorrectly. Furthermore, the multi-agent debate sometimes performs *worse* than a single strong model like GPT-4.

**2. MMedAgent (EMNLP 2024)**
*   **What it is:** One of the first models focused specifically on medical "tool use."
*   **How it works:** It acts as a router. When given a complex medical image (spanning 5 different modalities), it decides which of 6 specialized, external medical tools (like specific segmentation algorithms or measurement calculators) to call via API to help it solve the problem. 

**3. MAM (Modular Multi-Agent)**
*   **What it is:** A highly structured framework for diagnosing complex cases.
*   **How it works:** It breaks diagnosis down into distinct roles: a "Retrieval Agent" (fetches similar past cases), an "Analyst Agent" (reads the current image), and a "Decider Agent" (makes the final call). 
*   **The Catch:** Your analysis showed a major flaw: forcing too many debate rounds actually dilutes accurate diagnoses. In one dataset, the multi-agent debate overrode correct initial answers, dropping accuracy from 70% to 64%.

**4. MedAgent-Pro**
*   **What it is:** An evidence-based diagnostic agent focused heavily on ophthalmology (eye diseases like glaucoma).
*   **How it works:** Before making a diagnosis, it calls specific "indicator agents." For example, it asks one tool to measure the vertical Cup-to-Disc Ratio (vCDR) and another to look for Disc Hemorrhage (DH). 
*   **The Catch:** This paper perfectly illustrates **Gap 3 (Error Cascade)**. When the DH tool performed poorly, the overall agent's accuracy plummeted by 16.5%. The agent trusts its tools blindly.

**5. Meissa**
*   **What it is:** A lightweight (4 Billion parameter) offline agent.
*   **How it works:** Running massive models like GPT-4 in a hospital is expensive and poses privacy risks. Meissa is a small, local model trained using Reinforcement Learning on "trajectories" generated by massive models. It learns to mimic their step-by-step reasoning but runs 22× faster.
*   **The Catch:** It suffers from **Gap 5 (Safety Deficit)**. While fast, it only scores 57% on MedQA and produces confident but incorrect answers without any mechanism to say "I don't know."

---

# 4b. Medical Image Segmentation Agents

**6. IBISAgent**
*   **What it is:** An interactive agent for fine-grained pixel reasoning.
*   **How it works:** It wraps a vision model (MedSAM2) inside a Reinforcement Learning loop. The agent learns an "interactive policy"—meaning it learns exactly where to place artificial "clicks" on an image to get the segmentation model to output the perfect boundary.
*   **The Catch:** It takes almost 29 seconds to process a single case, violating real-time clinical latency requirements (**Gap 7**).

**7. MedSAM-Agent**
*   **What it is:** Similar to IBISAgent, it uses Reinforcement Learning (RL) to automate prompt generation for the Segment Anything Model (SAM).
*   **How it works:** It uses "reward shaping" to train the agent to iteratively refine the segmentation mask over multiple turns, mimicking how a human might adjust a bad crop. 

**8. MedSAM3**
*   **What it is:** An attempt to allow SAM to be controlled purely by text concepts rather than physical clicks (e.g., typing "segment the tumor").
*   **The Catch:** This model highlights **Gap 2 (2D→3D Chasm)** and **Gap 9 (Semantic Brittleness)**. On 3D liver tumors, it completely failed (Dice 0.14). Furthermore, if you ask it to segment a "cell" it works, but if you change the prompt to "nuclei", it fails completely (Dice drops to 0.00).

**9. MemSeg-Agent**
*   **What it is:** A memory-based adaptation agent.
*   **How it works:** Usually, to make an AI good at a new hospital's MRI scans, you have to retrain its weights. MemSeg freezes the model's weights and instead stores new hospital-specific knowledge inside "memory tokens" that the model accesses during inference.
*   **The Catch:** Your analysis exposed its **Human-in-the-Loop Dependency (Gap 8)**. It achieved a great 77% accuracy only when a human clinician provided real-time "working memory" corrections. Without the human, it scored 30%.

**10. MedEyes**
*   **What it is:** A "gaze-guided" visual agent.
*   **How it works:** Rather than processing a massive medical image all at once (which dilutes focus), MedEyes generates dynamic "visual focus" crops, zooming in on specific anomalies before passing them to the reasoning engine. 

---

# 4c. Multi-Agent Orchestration & Collaboration

**11. Cerebra**
*   **What it is:** A simulated Multi-Disciplinary Team (MDT) for dementia.
*   **How it works:** It spins up specialized personas (Neurologist, Neuroradiologist, Geneticist). Each looks at their respective data stream (Clinical text, MRI images, Genetic markers), and they collaborate to produce a risk assessment.
*   **The Catch:** **Gap 11 (Missing Modality Brittleness)**. If the patient didn't get an MRI, the Cerebra pipeline halts completely. It lacks the graceful degradation required in real clinics where data is often missing.

**12. CoMMa (Contribution-Aware Medical Multi-Agents)**
*   **What it is:** A game-theoretic multi-agent framework.
*   **How it works:** When a team of agents makes a diagnosis, CoMMa uses Shapley values (a concept from economics) to calculate exactly how much each agent contributed to the correct answer. It uses this to weigh their votes in the future.
*   **The Catch:** The multi-agent overhead isn't worth it. Moving from 1 agent to 5 agents only improved AUC by 0.5%.

**13. CARE**
*   **What it is:** A "Privacy-Compliant" and accountable reasoning agent.
*   **How it works:** A GPT-4o "Coordinator" oversees a Visual Question Answering (VQA) tool and a Segmentation tool. It is designed to ensure that any text output is grounded in visual evidence.
*   **The Catch:** The Coordinator agent has a bad habit of "over-editing." It will take perfectly correct outputs from the VQA tool and rewrite them into incorrect answers.

**14. 3DMedAgent**
*   **What it is:** A framework trying to bridge perception (seeing) and understanding (diagnosing) for 3D scans (CT/MRI).
*   **How it works:** It attempts to use a "Think-with-1-Slice" loop to force a 2D VLM to reason about 3D data.
*   **The Catch:** It still fundamentally relies on 2D slice sampling, meaning it fails on questions that require understanding the 3D relationships between adjacent organs (**Gap 2**).

---

# 4d. Self-Evolving & Adaptive Agents

**15. MACRO**
*   **What it is:** An agent that auto-discovers its own tools.
*   **How it works:** It uses Group Relative Policy Optimization (GRPO - a type of RL). When it successfully solves a clinical case by stringing together a sequence of basic tools, it "mines" that trajectory and permanently saves the sequence as a new, complex "composite tool" to use later.
*   **The Catch:** To know if it successfully solved a case and should update its memory, it needs the "ground truth" correct answer—which isn't available during real-world deployment (**Gap 6**).

**16. MedAgentSim**
*   **What it is:** A simulated environment where AI doctors and AI patients interact.
*   **How it works:** It evolves the agents' conversational and diagnostic skills by forcing them to run thousands of simulated clinical consultations against each other.
*   **The Catch:** When evaluated on real NEJM diagnostic benchmarks, it topped out at 26-29% accuracy, proving that simulated training isn't translating to complex clinical reasoning.

**17. Camyla**
*   **What it is:** An Autonomous AI Medical Scientist.
*   **How it works:** It doesn't diagnose patients; it invents new AI models. You tell it "I need a better cardiac segmentation model." Camyla writes Python code, runs the training loop, evaluates the result, and rewrites the code to improve the architecture.
*   **The Catch:** During testing, researchers found it occasionally wrote "bugs" into the code that accidentally caused data leakage (testing on training data), which falsely lowered the loss. The AI thought it was inventing a better model, but it was just cheating (**Gap 6**).

**18. VERITAS**
*   **What it is:** An agent focused on "Epistemic Reasoning" (knowing *how* it knows something).
*   **How it works:** Before making a diagnosis, it explicitly states a hypothesis, selects specific visual tools to test that hypothesis, and records the evidence. 

---

# 4e. Benchmarks, Evaluation & Recipe Papers

**19. MedOpenClaw & MedFlowBench**
*   **What it is:** The most disruptive benchmark paper in the category. 
*   **How it works:** Instead of giving the AI a cropped, static JPEG to analyze, MedOpenClaw drops the agent into a live Windows/Linux environment running 3D Slicer (professional clinical software). The agent must use mouse clicks and keyboard shortcuts to scroll through 3D CT volumes and find the lesion. 
*   **The Finding:** Agents that scored 90%+ on static datasets collapsed to 42.1% accuracy when forced to navigate a real clinical GUI (**Gap 1**).

**20. MedXIAOHE**
*   **What it is:** A "comprehensive recipe" for building medical agents.
*   **How it works:** It provides a vast Medical Entity Taxonomy (MET) to better structure the data used to train medical agents. 
*   **The Catch:** While it scores 87.5% on clean, curated datasets, its performance plummets to 25.7% on real-world, in-house clinical documents with natural noise and distortion. 

**21. ToolSelect (Paper 085)**
*   **What it is:** A router designed to fix the "Tool Selection" problem.
*   **How it works:** Uses Attentive Neural Processes to figure out which specialist model should handle a specific patient's case.
*   **The Catch:** The gap between the router's choices and an "Oracle" (perfect choice) is massive (72% vs 96%), showing we still don't know how to reliably predict which tool will work best for a given image (**Gap 3**).

**22. AgentDS**
*   **What it is:** A study on automating data science (feature selection) for clinical datasets.
*   **The Finding:** They found that fully automated "agentic" feature selection actually degraded performance compared to human-curated features. Furthermore, it required 10-15 hours of human expert guidance per dataset to function properly (**Gap 8**).

**23. AURA (MICCAI 2025)**
*   **What it is:** An agent focused on "Counterfactual Self-Evaluation."
*   **How it works:** To ensure it isn't hallucinating, AURA uses a diffusion model to slightly alter (edit) the medical image, generating a "counterfactual." It then analyzes both the original and the fake image to see if its reasoning holds up.
*   **The Catch:** Running a diffusion model to generate multiple fake images during inference takes way too long, making it impractical for high-speed, real-world clinical workflows (**Gap 7**).


---

# SLIDE 5 — SOTA Model Landscape: What's Being Built?

# Architecture Patterns Across All 27 Papers

| Architecture Type | Representative Papers | Core Innovation |
|---|---|---|
| **Tool-using agents** | MMedAgent, MedAgent-Pro, CARE, ToolSelect | LLM invokes specialized APIs (segmentation, detection, retrieval) |
| **Multi-agent debate** | MDAgents, MAM, CoMMa, Cerebra | Specialized agents vote, debate, and attribute decisions |
| **RL-trained agents** | MedVR, MedSAM-Agent, MACRO, Meissa | Reward-shaping to improve tool use or segmentation accuracy |
| **Memory-augmented agents** | MACRO, MemSeg-Agent, MedAgentSim | Experience replay and episodic memory for self-improvement |
| **Agentic segmentation** | IBISAgent, MedSAM-Agent, MemSeg-Agent, MedSAM3 | SAM2 as a callable tool within a reasoning loop |
| **Autonomous AI Scientist** | Camyla, Medical AI Scientist | Fully autonomous architecture search or literature synthesis |
| **Simulation environments** | MedAgentSim, AgentDS | Multi-agent patient/doctor simulation for training and evaluation |

# Key Backbone Models Used

| Model | Used In |
|---|---|
| **GPT-4o / GPT-4V** | CARE (coordinator), MedAgent-Pro (decider), VERITAS, Conversational MedAI |
| **Qwen2.5-VL / Qwen3-VL** | MACRO (3B, fine-tuned), Meissa (4B, fine-tuned) |
| **SAM2 / MedSAM2** | IBISAgent, MedSAM-Agent, MemSeg-Agent, MedSAM3 |
| **DeepSeek (teacher VLM)** | MACRO (bootstrap), Meissa (trajectory generation) |
| **Claude 3.7 Sonnet** | M^3Builder (AutoML backend), MedXIAOHE |
| **LLaMA 3.3 / Gemini 2.0** | MAM, M^3Builder (weaker backends — both cause severe degradation) |

# Modality Coverage Across 27 Papers

| Modality | Papers | Key Works |
|---|---|---|
| 2D X-Ray (CXR) | 6 | VERITAS, AURA (CheXpert), MedVR (ChestX-ray14) |
| 3D CT / MRI | 9 | MedSAM-Agent (CT/MRI), 3DMedAgent, MemSeg-Agent (CHAOS/ACDC) |
| Fundus / Ophthalmology | 4 | MedAgent-Pro (REFUGE), MACRO (REFUGE2), MedEyes |
| Pathology / Histology | 2 | MedXIAOHE, CoMMa (HCC biopsy) |
| Echocardiography | 3 | MACRO (MITEA), MemSeg-Agent (CAMUS, CardiacUDA) |
| EHR / Clinical Notes | 3 | AgentDS, MedAgentSim, MDAgents |
| Multi-modal text + image | 9 | MMedAgent, MAM, MDAgents, Meissa, CARE |

---

# SLIDE 6 — Quantitative SOTA Benchmarks: The Numbers

# Medical VQA & Reasoning

| System | Benchmark | Score | Comparison |
|---|---|---|---|
| **Meissa** (4B offline) | MedQA | 57.2% | GPT-4V: 88.7% — 31.5 pt gap |
| **Meissa** (4B offline) | OmniMedVQA | Beats GPT-4o | Only OOD win |
| **MDAgents** | MedQA (5-choice) | 72.9% | Solo GPT-4: 75.1% — agents can be *worse* |
| **MedXIAOHE** | MMMU_val-Med | 87.53% | HLE-Med: 25.77% (vs Gemini: 36.91%) |
| **MedEyes** | PMC-VQA | 55.3% | MMMU Health: 59.7% |
| **MAM** | MedQA | 40.0% | DeepLesion: 35.1% |

# Medical Image Segmentation (Dice Score)

| System | Dataset | Score | Baseline |
|---|---|---|---|
| **MemSeg-Agent** + human WM | CardiacUDA (cross-domain) | 77.30% | MedSAM2: 42.75% (needs human) |
| **MemSeg-Agent** static only | CardiacUDA | 30.74% | MedSAM2: 42.75% (WORSE!) |
| **MemSeg-Agent** | CAMUS Echocardiography | ~85-87% | SwinUNETR: 94.76% (-10 pts) |
| **MedSAM3** (text-only) | LiTS 3D liver tumor | 0.14 | U-Mamba: 0.79 (catastrophic) |
| **MedSAM3** (text-only) | BUSI ultrasound | 0.27 | Near-random |
| **MACRO** | RAM-W600 (Bone erosion) | F1: 30.0% | GPT-4o: competitive zero-shot |
| **IBISAgent** | MSD Hepatic Vessel Tumor | ~0.74-0.85 | Liver organ: ~0.97 |

# Clinical Decision / Diagnosis

| System | Task | Score | Gap |
|---|---|---|---|
| **MedAgent-Pro** | Glaucoma (without vCDR tool) | F1: 14.3% | With vCDR: F1 ~77% |
| **MedAgent-Pro** | Adding DH (weak) tool | F1: **-16.5%** | Weak tool *hurts* |
| **MedAgentSim** | NEJM diagnostic accuracy | 26-29% | Clinical practice: >70% |
| **MedOpenClaw** agents | MedFlowBench (3D GUI) | **42.1%** | Static 2D benchmark: 90%+ |
| **CoMMa** | MTBBench AUC | 0.595 ± 0.061 | Near-chance |
| **Camyla** | ACDC Cardiac Dice | 0.932 | nnU-Net: 0.918 (+0.014 only) |

---

# SLIDE 7 — The 11 Critical Research Gaps Identified

> Full analysis: [gap_analysis_1_1_multimodal_clinical_agents.md](file:///C:/Users/pbhu3/.gemini/antigravity/brain/0ff160e1-f415-477e-915b-74ad2ee5e9ca/gap_analysis_1_1_multimodal_clinical_agents.md)

# CRITICAL — Deployment Blockers

**GAP 1 — Benchmark Validity Crisis**
Static benchmarks → 42.1% in real 3D clinical software (MedOpenClaw). 90%+ benchmark scores are **not clinically meaningful**.

**GAP 2 — The 2D→3D Chasm**
MedSAM3 scores Dice 0.14 on LiTS (vs. 0.79 for U-Mamba). Every agent operates in 2D despite clinical imaging being fundamentally volumetric.

**GAP 3 — Error Cascade in Tool Pipelines**
MedAgent-Pro: weak DH tool crashes F1 by -16.5%. CARE: coordinator over-edits correct answers. No tool reliability estimation exists.

**GAP 5 — Clinical Safety & Trust Deficit**
Meissa: 57.2% MedQA with silent failures and zero abstention. No system provides calibrated confidence, rejection, or audit trails.

# HIGH IMPACT

**GAP 4 — Multi-Agent Overhead vs. Actual Gain**
MAM: agents override correct answers (70.2%→64.0%). CoMMa: 5 agents vs. 1 = only +0.5% AUC. Multi-agent is empirically fragile.

**GAP 6 — Annotation & Closed-Loop Data Bottleneck**
Camyla: bugs improve loss via data leakage. MACRO: correct labels unavailable at test time — self-improvement loop is circular.

**GAP 7 — Real-Time Latency**
IBISAgent: 28.70 sec/case. AURA: diffusion counterfactuals impractical for emergency radiology. No system meets clinical turnaround requirements.

**GAP 8 — Human-in-the-Loop Dependency**
MemSeg-Agent's 77.30% Dice *requires* real-time clinician annotations. AgentDS needed 10-15 hours of expert time per challenge.

# 🟡 STRUCTURAL

**GAP 9 — Ontological / Semantic Grounding**
MedSAM3: "cell" → "nuclei" drops Dice from 0.68 → 0.00. Agents have no grounding in ICD-10 or SNOMED-CT terminologies.

**GAP 10 — Research-to-Bedside Translation**
No paper has deployed in a real hospital. FDA/CE-mark, DICOM/HL7-FHIR, liability, and IRB requirements are universally absent.

**GAP 11 — Missing Modality Brittleness**
Cerebra halts completely if MRI is missing. No graceful degradation for partial data — universally required in real clinical scenarios.

---

# SLIDE 8 — Most Critical Finding: The "90% → 42%" Collapse

> *"Agents scoring 90%+ on static 2D benchmarks collapse to 42.1% 3D coordinate precision when asked to navigate real clinical imaging software."*
> — MedOpenClaw + MedFlowBench (arXiv 2026)

```
PUBLISHED BENCHMARK SCORE    →    REAL-WORLD CLINICAL SCORE
         90%+                              42.1%
   (curated, cropped,               (real CT/MRI software,
    single-image, 2D)                3D navigation, GUI ops)
```

**Why this matters for implementation:**
- Model selection based on leaderboards will very likely produce clinical failures
- The field is optimizing for benchmark engineering, not clinical engineering
- The field urgently needs **prospective clinical trial-style evaluation**

---

# SLIDE 9 — Future Research Directions (Tiered by Impact)

# Tier 1: Foundational — Must Solve Before Clinical Deployment

| Direction | Solves | Addresses Gap |
|---|---|---|
| Calibrated uncertainty + abstention mechanisms | Agents must escalate when uncertain | GAP 5 |
| Native 3D medical foundation models | Stop 2D slice-sampling of CT/MRI | GAP 2 |
| Prospective real-workflow benchmarks | Replace VQA with real software interaction | GAP 1 |
| Tool reliability estimation in-context | Detect & compensate for tool failure | GAP 3 |

# Tier 2: Engineering — Required for Production

| Direction | Solves | Addresses Gap |
|---|---|---|
| DICOM/HL7-FHIR native integration | Work inside hospital IT infrastructure | GAP 10 |
| Sub-second inference pipelines | Meet radiology turnaround requirements | GAP 7 |
| SNOMED-CT / ICD-10 ontology grounding | Eliminate medical terminology brittleness | GAP 9 |
| Graceful modality degradation architectures | Function with missing MRI/lab/genomics | GAP 11 |

# Tier 3: Long-Horizon Research Frontier

| Direction | Solves |
|---|---|
| Longitudinal patient trajectory agents | Track patients across time (not single encounters) |
| Federated multi-institutional learning | Train across hospitals without sharing data |
| Multi-agent collusion detection | Prevent echo-chamber failures in debate systems |
| Regulatory-aware agent design | Build FDA/CE audit trails from the start |

---

# SLIDE 10 — My Implementation Plan (What's Next)

# Phase 1: Complete the Survey (This Weekend)
Deep analysis of 3 remaining high-priority categories:
- **1.2 Radiology Agents** (30 papers) — CT report generation, lung nodule detection, MRI
- **1.10 EHR & Clinical Note Agents** (30 papers) — FHIR-based agents, readmission, ICD coding
- **1.13 Reasoning & Multi-Agent Techniques** (50+ papers) — debate, RAG, RL-based reasoning

# Phase 2: Cross-Category Synthesis
Synthesize findings across ALL analyzed categories to identify:
- Which gaps are universal vs. domain-specific
- Which benchmarks recur across categories
- Where modality coverage is most critically absent

# Phase 3: Implementation Target Selection

**4 top candidates** (based on gap severity × feasibility):

| Candidate | Gap | Why Feasible |
|---|---|---|
| **Calibrated abstention agent** for CXR diagnosis | GAP 5 | CheXpert dataset; ECE + AUROC metrics |
| **3D-aware segmentation agent** with learned slice selection | GAP 2 | SAM2 backbone; LiTS, CHAOS datasets |
| **Tool reliability estimator** for medical pipelines | GAP 3 | Wraps existing tools; no new training needed |
| **Prospective 3D benchmark** extending MedOpenClaw | GAP 1 | Engineering project; real clinical software |

---

# SLIDE 11 — Identified Survey Papers: My Reference List

I identified **26 survey papers** for broader synthesis. The most important:

# General Healthcare AI Agent Surveys

| Survey | Venue | Why Important |
|---|---|---|
| LLM-based agentic systems in medicine and healthcare | *Nature Machine Intelligence*, 2024 | Most cited; foundational taxonomy |
| AI agent in healthcare: applications, evaluations, future | *npj Artificial Intelligence*, 2026 | Most up-to-date comprehensive review |
| A Survey of LLM-based Agents in Medicine: How far are we from Baymax? | ACL 2025 Findings | Agent-specific; collaboration patterns |
| A Comprehensive Survey of Agentic AI in Healthcare | TechRxiv, 2025 | All categories, 200+ papers |
| Coordinated AI agents for advancing healthcare | *Nature Biomedical Engineering*, 2025 | Multi-agent coordination focus |
| Benchmarking LLM-based agent systems for clinical decision | *npj Digital Medicine*, 2026 | Evaluation methodology |
| Agentic LLM systems in medicine: systematic review & taxonomy | TechRxiv, 2025 | Taxonomy-first approach |

# Radiology-Specific Surveys

| Survey | Venue | Why Important |
|---|---|---|
| Agentic AI in Radiology: Evolution from LLMs to Clinical Integration | *Radiology: AI (RSNA)*, 2025 | Deployment challenges; specialty-specific |
| Agentic AI and LLMs in Radiology: Hallucination Challenges | *Bioengineering (MDPI)*, 2025 | Hallucination quantification |
| Agentic systems in radiology: Design, Applications, Challenges | arXiv, 2025 | Most comprehensive radiology review |

# Specialty & Discovery Surveys

| Survey | Venue | Why Important |
|---|---|---|
| Computational Pathology in the Era of Agentic AI | arXiv, 2026 | Expert panel; pathology gaps |
| Simulated patient systems powered by LLM-based AI agents | *Communications Medicine*, 2025 | Patient simulation for medical education |
| Empowering biomedical discovery with AI agents | *Cell*, 2024 | Foundational; lab automation and discovery |
| Agentic AI and the rise of in silico team science | *Nature Biotechnology*, 2026 | Future vision for AI-driven science teams |

---

# SLIDE 12 — Summary: What I Found, What Comes Next

# What I Found (27 papers, Category 1.1)
- Mapped all active architecture patterns: tool-using, multi-agent debate, RL-trained, memory-augmented, autonomous AI Scientist
- Documented SOTA quantitative scores across **30+ benchmarks**
- Identified **11 systemic gaps** — 4 are hard blockers for any clinical deployment
- Critical finding: **benchmark scores dramatically overestimate real clinical utility**

# Core Takeaway for Implementation
The field is producing sophisticated papers with impressive numbers, but **virtually no system is deployment-ready** due to:
1. Evaluations on unrealistic, curated benchmarks (90% → 42% gap)
2. Zero uncertainty quantification or safety/abstention mechanisms
3. 2D-only processing of inherently 3D clinical imaging
4. Tool pipelines that cascade errors without any detection

# My Roadmap
| When | What |
|---|---|
| **This weekend** | Gap analysis for 1.2 (Radiology) + 1.10 (EHR) + 1.13 (Reasoning) |
| **End of next week** | Cross-category synthesis; select implementation target |
| **Then: Implementation** | Build and evaluate one novel system addressing a critical gap |

---

*Internship Presentation Document — IIT Mandi | July 2, 2026*

**Supporting files:**
- [Full Gap Analysis (1.1)](file:///C:/Users/pbhu3/.gemini/antigravity/brain/0ff160e1-f415-477e-915b-74ad2ee5e9ca/gap_analysis_1_1_multimodal_clinical_agents.md)
- [Paper Catalog (Excel)](file:///c:/PNS/IITMandi_Labwork/categorized_paper_catalog.xlsx)
- [Papers by Category — Full List](file:///c:/PNS/IITMandi_Labwork/[Papers%20by%20Category]_MAIN.txt)
