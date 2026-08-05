# The Paradigm Shift to Agentic AI in Healthcare: A Comprehensive Modality-Driven Survey

# Abstract
The integration of artificial intelligence in healthcare has reached an inflection point, transitioning from static, task-specific predictive models to dynamic, goal-oriented "AI Agents." These agents leverage Large Language Models (LLMs) and Vision-Language Models (VLMs) not merely as conversational endpoints, but as cognitive engines to orchestrate complex, multi-step workflows. This paper presents a highly comprehensive, academic-grade review of the State-of-the-Art (SOTA) in medical AI agents. Moving beyond specialty-biased surveys, we classify the landscape strictly by data modality: Clinical Text & EHR, Medical Vision & Imaging, and Omics & Systems Biology. We detail the exhaustive tasks tackled within these domains, profile cutting-edge single-agent and multi-agent frameworks, and analyze the profound shift in benchmarking theory—from static exam datasets to interactive, FHIR-compliant simulated environments. Finally, we address the critical "Translational Gap" and the ongoing challenges in quantifying clinical safety and ethical compliance.

---

# 1. Methodology & Definitional Taxonomy

# 1.1 Defining AI Agents vs. Static Models
Recent systematic reviews draw a sharp architectural boundary between "traditional" or "static" artificial intelligence and true "AI Agents" in medicine. 
*   **Static AI:** Defined by narrow prediction (e.g., CNNs for radiographic classification) or stateless text generation (querying a standard LLM). They lack memory, cannot navigate temporal clinical data autonomously, and do not wield external tools.
*   **Agentic AI:** Defined by autonomous, goal-directed reasoning. The foundational LLM serves as a cognitive core that orchestrates workflows. To be formally classified as an agent, systems must demonstrate:
    1.  **Iterative Reasoning:** Utilizing Chain-of-Thought (CoT) or Reason-and-Act (ReAct) paradigms to formulate clinical plans, observe outcomes, and adjust dynamically.
    2.  **Tool Integration:** Native ability to query FHIR-compliant Electronic Health Record (EHR) databases, call clinical calculators, or execute APIs.
    3.  **Multi-Agent Collaboration:** Distributing sub-tasks among specialized persona agents (e.g., radiologist, coder, auditor) to reach consensus.

# 1.2 The Seven-Dimensional Taxonomy Framework
Systematic evaluations now frequently employ the *Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-Based Agents* (Vatsal et al., 2026). This framework maps architectures across: Cognitive Capabilities, Knowledge Management (e.g., RAG), Interaction Patterns (HITL vs. Autonomous), Adaptation, Safety & Ethics, Framework Typology, and Task Specificity. Current analyses reveal that while "External Knowledge Integration" is widespread, advanced features like "Drift Detection" remain critically under-implemented.

---

# 2. Clinical Text, EHR, and Administrative Agents

# 2.1 Taxonomy of Specific Tasks
*   **Clinical Text & Ambient Scribes:** Moving beyond passive transcription, active agents perform multi-step charting, generate differential diagnoses, and autonomously draft orders for labs or imaging based on conversational context.
*   **Structured EHR Reasoning (Text-to-SQL):** Translating natural language queries into executable SQL over complex longitudinal databases. Agents must navigate temporal logic and clinical vocabulary mismatches.
*   **Medical Coding:** Mapping unstructured discharge summaries to ~70,000 ICD-10/CPT codes. Agents must extract exact sentence-level evidence to ensure rigorous billing compliance.
*   **Administrative/Operational (Computer-Use Agents):** Automating Revenue Cycle Management (RCM) and Prior Authorization. Visual agents navigate payer portals, handle Multi-Factor Authentication (MFA), and complete web forms autonomously.

# 2.2 State-of-the-Art Models & Frameworks
*   **EHRMaster:** A code-augmented agent framework that processes tabular EHR data via a deterministic 3-stage process (Solution Planning, Concept Alignment, Execution Strategy) [[1]](# references).
*   **MedCodER & Symphony:** Architectures utilizing a "planner-executor-verifier" structure. They leverage RAG and re-rankers to link every ICD code to specific clinical evidence [[2]](# references).
*   **Multi-Agent Orchestrators:** Frameworks that deploy an "Intake Agent," "EHR Query Agent," and "Verification Agent" coordinated by a central orchestrator, tightly integrated with FHIR APIs for HIPAA-compliant execution.

---

# 3. Medical Vision and Multimodal Agents

# 3.1 Exhaustive Task Taxonomy by Modality
*   **Radiology:** Visual Question Answering (VQA) on 3D scans, automated report generation grounded to visual bounding boxes, lesion measurement, and longitudinal tracking for disease progression triage.
*   **Digital Pathology (WSI):** Iterative gigapixel navigation (panning, zooming, extracting ROIs), cellular segmentation, phenotyping, and hierarchical diagnostic triage mimicking a human pathologist.
*   **Surgical Video Analysis:** Real-time scene understanding, instrument tracking to prevent iatrogenic injuries, surgical phase recognition, and automated post-operative documentation via audio/video fusion.

# 3.2 SOTA Models & Multi-Agent Frameworks
*   **CoMMa (Contribution-Aware Medical Multi-Agents):** A decentralized framework utilizing a game-theoretic objective to measure the marginal utility of specialized agents, yielding robust decision pathways [[3]](# references).
*   **SurgRAW:** Designed for robotic surgical video, utilizing a CoT framework where an orchestrator divides scene understanding into specialized reasoning streams to mitigate hallucinations [[4]](# references).
*   **PathFinder & SlideSeek:** Abandoning traditional Multiple Instance Learning (MIL), these employ role-based agents (Triage, Navigation, Diagnosis) to traverse pathology slides.

---

# 4. Omics, Genomics, and Systems Biology

# 4.1 Exhaustive Taxonomy of Agentic Tasks
*   **Autonomous Bioinformatics Pipelines:** Orchestrating WGS/WES variant calling and bulk RNA-seq differential expression. Agents retrieve data, select tools, write execution scripts, and compile reports.
*   **Single-Cell Omics:** Autonomous quality control, dimensionality reduction, evidence-driven cell-type annotation, and perturbation modeling (predicting cellular responses to knockouts).
*   **Systems Biology & Molecular Design:** Querying knowledge graphs for Drug-Target Interaction (DTI) and navigating chemical space while executing ADMET prediction tools.
*   **Closed-Loop Laboratory Automation:** "Self-driving labs" where agents autonomously interface with robotic liquid handlers to execute and iterate wet-lab experiments.

# 4.2 SOTA Models and Agent Architectures
*   **Single-Agent Planners (Biomni):** Monolithic agents integrated with hundreds of tools (CRISPR design, ClinVar, PDB) that translate natural language into Python/R scripts.
*   **Multi-Agent Systems (MAS):**
    *   *CellAgent:* Hierarchical structure (Planner, Executor, Evaluator) with self-reflective loops to refine single-cell clustering.
    *   *AutoBA:* A multi-omics agent emphasizing local deployment for data privacy.
    *   *DrugAgent:* Coordinates specialized agents for target identification and clinical trial simulation using transparent ReAct prompting.

---

# 5. Benchmarking Theory and Evaluation Metrics

# 5.1 The Shift to Interactive Environments
Static question-answering benchmarks (e.g., MedQA) suffer from a severe "knowledge-practice performance gap." Evaluation theory has shifted toward interactive, simulation-based environments:
*   **MedAgentBench & EHRStruct:** Agents are placed in FHIR-compliant virtual EHRs to autonomously navigate patient encounters, order medications, and interpret labs longitudinally [[1]](# references)[[5]](# references).
*   **EHRSQL:** Evaluates temporal queries and "abstention"—the critical ability to refuse unanswerable questions rather than hallucinating.
*   **BioXArena & BioAgentBench:** Sandboxed environments evaluating bioinformatics agents on executable code generation and pipeline artifact validity [[6]](# references).

# 5.2 Quantifying Safety, Ethics, and Trust
Safety evaluation has moved beyond standard accuracy to dynamic, adversarial frameworks.
*   **Adversarial Simulation:** Injecting noisy data, conflicting guidelines, or ambiguous presentations to quantify "hallucination under pressure."
*   **Clinical Nuance & Deferral Rates:** Scoring an agent's capability to recognize its own uncertainty and gracefully defer to a human clinician.
*   **Systems Efficiency:** Measuring latency, VRAM constraints (critical for surgical edge-deployment), and task completion cost.

---

# 6. Gap Analysis and Clinical Translation
Despite rapid advancements, significant hurdles remain before widespread clinical translation:
1.  **The Translational Gap:** Simulated benchmark performance rarely matches real-world utility. Agents excel in routine, structured scenarios but suffer massive performance degradation in rare, complex, or highly ambiguous clinical cases.
2.  **Operational Failures in CUAs:** Computer-Use Agents (navigating payer portals) exhibit high compounding error rates; a 99% step-accuracy degrades rapidly over a 50-click prior authorization workflow, currently limiting them to Human-in-the-Loop (HITL) deployments.
3.  **The "Black Box" Liability:** Multi-step autonomous reasoning can compound errors invisibly. Robust, transparent reasoning traces, continuous drift detection, and strict safety guardrails are imperative for regulatory approval and clinical trust.

# 7. Conclusion
The evolution of medical AI from static predictors to autonomous agents is fundamentally reshaping healthcare technology. By integrating complex reasoning, external tool use, and multi-agent collaboration, these systems offer unprecedented capabilities across clinical documentation, multimodal diagnostics, and autonomous bioinformatics. However, realizing their full potential requires resolving the severe translational gaps exposed by novel interactive benchmarks, ensuring that agentic autonomy is firmly grounded in clinical safety, privacy, and rigorous human oversight.

---

<a id="references"></a>
# 8. References

1. **EHRStruct & EHRMaster:** *EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks*. [arXiv:2511.08206](https://arxiv.org/abs/2511.08206)
2. **MedCodER:** *MedCodER: A Generative AI Assistant for Medical Coding*. [arXiv:2409.15368](https://arxiv.org/abs/2409.15368)
3. **CoMMa:** *Contribution-Aware Medical Multi-Agents*. [arXiv:2602.09159](https://arxiv.org/abs/2602.09159)
4. **SurgRAW:** *SurgRAW: Multi-Agent Workflow with Chain of Thought Reasoning for Robotic Surgical Video Analysis*. [arXiv:2503.10265](https://arxiv.org/abs/2503.10265)
5. **MedAgentBench:** *Evaluating Large Language Models as Agents in Medical Environments*. [GitHub Repository](https://github.com/stanfordmlgroup/MedAgentBench)
6. **BioXArena:** *BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks*. [arXiv:2605.15766](https://arxiv.org/abs/2605.15766)
