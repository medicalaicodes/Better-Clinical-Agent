# Evolving Medical Imaging Agents via Experience-driven Self-skill Discovery
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.05860

# 1. Problem Statement
Medical image interpretation relies on complex, multi-step, tool-centric workflows. Existing LLM-based medical agents depend heavily on static, pre-defined tool sets and invocation strategies. This rigidity makes them brittle under domain shifts (e.g., varying acquisition protocols, evolving diagnostic requirements) and prevents them from autonomously learning from clinical experience to adapt or improve over time.

# 2. Key Contributions
- Identifies the fundamental limitation of static tool orchestration in clinical AI.
- Proposes MACRO, a self-evolving medical agent that shifts from static composition to experience-driven tool discovery.
- Introduces an image-feature memory for context-grounded tool selection and a composite tool synthesis module that automatically discovers, verifies, and registers reusable multi-step sequences.

# 3. Methodology
- **Experience-grounded Memory**: Bootstraps trajectories using a teacher model. Stores successful interaction fragments and retrieves context via image-feature cosine similarity to provide visually grounded, in-context guidance.
- **Composite Tool Discovery**: Mines frequent, contiguous tool invocation subsequences from successful trajectories and registers them as single "composite tools," continuously expanding the agent’s action space.
- **Two-Stage Policy Optimization**: 
  - *Stage 1 (Supervised Cold Start)*: Mitigates exposure bias by rolling out the student's own execution context based on teacher demonstrations.
  - *Stage 2 (GRPO-based Reinforcement)*: Applies Group Relative Policy Optimization with a sparse reward (+1) to incentivize the use of registered composite tools alongside trajectory-level success verification.

# 4. Datasets & Benchmarks
- **REFUGE2**: Glaucoma diagnosis (Fundus imaging).
- **MITEA**: Heart disease diagnosis, e.g., dilated cardiomyopathy (3D Echocardiography).
- **RAM-W600**: Bone erosion (BE) classification using conventional radiography (CR).

# 5. Key Results
- MACRO consistently outperforms general VLMs (GPT-4o, LLaVA-Med, Qwen2.5-VL) and existing medical agentic frameworks (MedAgents, MMedAgent, MDAgents).
- On REFUGE2, MACRO achieves 92.7% BACC and 80.3% F1, beating the strong MedAgent-Pro baseline.
- On RAM-W600, MACRO surpasses task-specific vision models (ResNet, MobileNet, MedMamba), achieving 61.75% BACC and 30.00% F1.

# 6. Unsolved Gaps & Benchmark Failures (Hyper-Critical Analysis)
- **Benchmark Failures**:
  - **Severe Sensitivity Failure on RAM-W600 (Bone Erosion)**: Despite outperforming baselines, MACRO achieves a dismal **28.50% Sensitivity** and **30.00% F1 score** on RAM-W600, indicating it completely misses the vast majority of positive cases and remains highly susceptible to class imbalance in subtle radiographic findings.
  - **Inconsistent Dominance on MITEA**: MACRO achieves a BACC of 77.2% on the MITEA dataset, which is a **decrease of 0.6%** compared to the MedAgent-Pro baseline (77.8%), proving that its tool discovery does not uniformly improve all metrics across tasks.
- **Unsolved Gaps**:
  - **Fidelity Bottleneck**: Composite tool quality is fundamentally bounded by trajectory verification fidelity. Noisy or partially correct traces will introduce suboptimal or flawed diagnostic skills into the library.
  - **Brittle Semantic Transfer**: The framework is unproven on unseen imaging modalities with substantially different visual semantics, risking failure modes not captured by the current static benchmarks.
  - **Lack of Clinical Impact Evaluation**: The evaluation is restricted entirely to predictive metrics. The system lacks prospective human-in-the-loop validation assessing workflow efficiency, calibration, and actual downstream clinical utility.

# 7. Modalities & Key Models
- **Modalities**: Fundus imaging, 3D Echocardiography, Conventional Radiography (CR).
- **Key Models**: 
  - *Base Model*: Qwen2.5-VL-3B-Instruct (fine-tuned via LoRA).
  - *Teacher Model*: DeepSeek.
  - *Baselines*: GPT-4o, BioMedClip, LLaVA-Med, Janus-Pro-7B, InternVL2.5-8B, MedAgents, MedAgent-Pro, MobileViT, ResNet, MedMamba, ConvKAN.
