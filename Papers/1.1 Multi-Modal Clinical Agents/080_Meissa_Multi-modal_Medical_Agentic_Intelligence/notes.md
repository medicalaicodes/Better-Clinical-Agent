# Meissa: Multi-modal Medical Agentic Intelligence
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.09018

# 1. Problem Statement
Current medical agentic AI systems rely on large, closed-source frontier models (like GPT-4 and Gemini) and complex zero/few-shot tool prompting. This makes them expensive, slow, and unsuitable for offline or privacy-sensitive clinical environments. Smaller open-weight models, however, struggle with complex reasoning and multi-step tool coordination, preventing them from acting as capable autonomous agents.

# 2. Key Contributions
*   **Meissa:** A fully offline, 4B-parameter medical agent that matches or exceeds frontier models on complex clinical tasks.
*   **Stratified Trajectory Generation:** A 3-tier algorithm that categorizes training queries by difficulty to dynamically route queries to direct inference or agentic reasoning.
*   **Prospective-Retrospective Supervision:** A novel training method where the teacher model acts (prospective) and then synthesizes a condensed, reasoning-tightened trajectory (retrospective) to teach the student model *how* to act efficiently.
*   **Unified Agentic Formalism:** Integrates diverse agentic environments (tool calling, visual reasoning, multi-agent collaboration, and clinical simulation) into a single trajectory format.

# 3. Methodology
*   **Base Model:** Qwen3-VL-4B-Instruct fine-tuned using LoRA.
*   **Data Generation (Algorithm 1):** 
    *   *Tier 1 (Direct):* The student model can answer correctly.
    *   *Tier 2 (Enhanced):* The teacher model (Gemini-3-flash/GPT-4V) answers correctly without tools.
    *   *Tier 3 (Agentic):* Requires teacher multi-step tool use.
*   **Trajectory Validation:** Uses structural filtering (format correctness) and behavioral filtering (hallucination detection, repetition loops) to clean the data.

# 4. Datasets & Benchmarks
*   **Training Data:** ~40K filtered agentic trajectories synthesized using Gemini-3-flash and GPT-4V.
*   **Benchmarks:** 13 diverse benchmarks spanning VQA (MIMIC-CXR-VQA, PathVQA, SLAKE, VQA-RAD), Reference-Grounded (PubMedQA), Multi-Agent Collaboration (MedQA, ChestAgentBench, OmniMedVQA), and Multi-turn Clinical Simulation (NEJM, AgentClinic, MedQA Ext., MIMIC-IV). 

# 5. Key Results
*   **Frontier-Level Performance:** Outperforms base Qwen3-VL-4B by 17.3% on average and matches/exceeds GPT-4o on out-of-distribution (OOD) benchmarks like ChestAgentBench (62.8%).
*   **Efficiency:** Achieves a ~22x latency speedup compared to standard Gemini-3-flash agents (4.1s vs 87.2s) by effectively routing 72% of queries to direct inference and only using multi-step agentic paths for 28% of complex queries.

# 6. Unsolved Gaps & Benchmark Failures (Hyper-Critical)
*   **Benchmark Failures (Quantitative):** Meissa fails significantly on tasks requiring deep parametric knowledge and expert clinical reasoning. It scores only 57.2% on MedQA (compared to GPT-4V's 88.7%) and a dismal 36.0% on MedXpertQA (compared to Gemini-3-flash's 69.2%), falling behind even smaller baselines like Ophiuchus-7B. It also exhibits negative transfer (-3.3%) on MedQA Ext, revealing that forcing a lightweight model to learn agentic reasoning policies severely competes with and degrades its parametric knowledge retention.
*   **Unsolved Gaps (Clinical Deployment):** For real-world clinical use, the model entirely lacks calibrated uncertainty estimation and an abstention mechanism (principled deferral). When faced with out-of-distribution queries or queries beyond its limited 4B knowledge capacity, it cannot reliably abstain or hand off to human experts, presenting a critical safety risk. Furthermore, while the model filters out hallucinations in training data, its reliance on visual parsing tools means tool-noise can still cascade into erroneous diagnoses in live settings.

# 7. Modalities & Key Models
*   **Modalities:** Text, Vision (Clinical imagery, Chest X-Rays, Pathology slides).
*   **Key Models:** Meissa (Qwen3-VL-4B-Instruct, Student), Gemini-3-flash (Teacher), GPT-4V (Teacher), GPT-4o.
