# Survey Taxonomy — Living Document

This document tracks the rapidly evolving landscape of AI agents in healthcare. As we have surpassed 100 cataloged papers, the taxonomy has naturally expanded from basic clinical QA to complex, embodied, and operational multi-agent systems.

---

# 1. Primary Domains and Categories
*As agentic AI expands beyond simple text retrieval, several distinct application domains have emerged:*

- **Clinical Decision Support (CDS) & Diagnostics:** The core domain, shifting from single-shot LLM answers to multi-agent debate (e.g., Dual-Agent systems, Proponent/Skeptic).
- **Administrative, EHR & Operations:** Highly operational tasks focusing on workflow automation (e.g., Computer-Use agents navigating GUI/Web, FHIR API interaction, scheduling).
- **Medical Vision & Multimodal:** Agents orchestrating computer vision tools for Radiology (3D CT/MRI, 2D X-Ray) and Dermatology. Edge-deployed vision agents are a growing subset.
- **Embodied Agents & Treatment Control:** AI agents moving into the physical world (e.g., drone logistics for medical supplies, robotic ultrasound control, continuous RL for anesthesia).
- **Research Agents:** Agents designed to automate scientific discovery, literature review, and real-world evidence generation (e.g., autonomous OMOP SQL execution, complex data extraction).
- **Benchmarking, Safety & Trustworthy AI:** A massive new category focusing on evaluating privacy, mitigating hallucinations (counterfactual reasoning), and ensuring secure deployments (Zero Trust Architecture).

# 2. Modalities Handled
*The range of data formats agents are interacting with:*
- **Text & Code:** Clinical Text (EHR, Guidelines, Dialogue), Scientific Literature, SQL, Code/Stats.
- **Vision:** 2D Images (Dermatology, X-Ray), 3D Imaging (CT, MRI, PET), Live Video (Ultrasound, Fetal, Pose), GUI/Web Interfaces.
- **Time-Series & Operations:** ECG, Hemodynamics, Operational/Logistics Data (Drone states, Hospital queues).
- **Genomics:** Genomic Data, ODE Simulators.

# 3. Recurring Architectures & Agent Frameworks
*The underlying cognitive and structural designs powering medical agents:*
- **Multi-Agent Debate & Consistency Verifiers:** E.g., Adversarial Debate, Proponent/Skeptic Agents, Critic Agents. Used to calibrate uncertainty and reduce hallucinations.
- **Computer-Use & Visual-Action:** E.g., OpenClaw, CarePilot, PhysicianBench. Agents that act by clicking and typing on legacy hospital software screens.
- **Tool-Augmented Orchestrators (ReAct/ToT):** Agents leveraging narrow APIs (e.g., a Cardiomegaly classifier, OMIM lookup) before synthesizing a final clinical output.
- **Self-Evolving & Dual-Memory Agents:** E.g., TheraAgent, Episodic Memory. Agents that remember past patient interactions and update their own heuristics over time.
- **Swarm & Generative Architectures:** E.g., OpenHospital, Hundred-Agent Hospital. Decentralized agents mimicking entire healthcare ecosystems (patients, nurses, doctors).
- **RL & Embodied Control:** E.g., Decentralized MARL (MAPPO), CEDA (CTDE DQN). Used for robotics and physical supply logistics.

# 4. Key Datasets & Benchmarking Paradigms
*How the community is measuring success:*
- **Static Clinical QA (Legacy):** MedQA, PubMedQA, MMLU-Med.
- **Long-Horizon & Interactive Benchmarks (Emerging):** CHI-Bench (workflow), MedMASLab Suite, HealthAdminBench.
- **Vision & Multimodal:** MIMIC-CXR, HAM10000, BraTS-Longitudinal, ACDC Cardiac MRI.
- **Safety & Privacy:** MedPriv-Bench, MedSafety-Eval, MedHalDetect.
- **Simulation Arenas:** OpenHospital Arena, Med-Drone-Sim, VitalDB / Simulated OR.

# 5. Identified Gaps and Emerging Problems
1. **The Interoperability Barrier:** While Computer-Use agents (OpenClaw) are promising for legacy EHRs, UI fragility causes catastrophic failure in long-horizon tasks. Standardized `MedGUI` benchmarks are urgently needed.
2. **The Privacy-Utility Trade-off:** As shown in `MedPriv-Bench`, aggressive PHI scrubbing destroys diagnostic accuracy. Novel "Safe-Context" memory architectures are required.
3. **Agentic Knowledge Maintenance:** Static knowledge graphs become outdated. We need continuous *Agentic Knowledge Engineering* where agents dynamically review literature to update hospital protocols.
4. **Physical Embodiment Constraints:** While agents excel in text, translating agentic logic to safe, real-time physical control (Robotic Ultrasound, Anesthesia) remains heavily bottlenecked by FDA regulatory frameworks (Contestability Protocol Frameworks).
