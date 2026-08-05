# CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.01607

# 1. Unsolved Gaps (Failures in Real Clinical Deployment)
1. **Task Scope Limitations**: The framework is heavily optimized for diagnosis tasks requiring local detail/regions of interest. It is not well-suited for global-related tasks, holistic patient scans, or diagram analysis.
2. **Coordinator Hallucination & Over-editing**: The dynamic coordinator (which plans tool use and reviews answers) is prone to hallucinations. For instance, GPT-4o was found to frequently "over-edit" the correct answers provided by the expert EG-VQA agent, resulting in introduced errors.
3. **Dependency on Coordinator Scale**: Using weaker or smaller open-source models for coordination fails because they cannot make correct tool invocations or reliably choose the right evidence view, breaking the agentic workflow.
4. **Context Length Sensitivity**: Feeding the model multiple visual clues simultaneously (e.g., both zoom-in crops and masks) actually degrades performance because the input sequence becomes too long to process effectively.
5. **Latency Trade-offs**: The full CARE-Coord agent pipeline drastically increases inference time compared to the static CARE-Flow, which limits its viability in fast-paced, real-time clinical environments.

# 2. Benchmark Failures (Quantitative Shortfalls)
1. **Underperformance on VQA-RAD**: CARE-Coord-B (68.29% accuracy) and CARE-Flow-B (63.64%) fail to beat heavily fine-tuned specialist baselines. For example, FAVP achieves 71.90% open / 88.20% closed accuracy, and BioMed-VITAL-13B achieves 69.72% open / 84.86% closed.
2. **Underperformance on SLAKE**: Despite strong results, CARE-Coord-B's 87.34% open / 89.19% closed accuracy still trails behind BioMed-VITAL-13B (91.69% open / 90.70% closed).
3. **Out-of-Domain (OOD) Struggles**: On the VQA-Med-2019 benchmark, CARE-Coord-B (60.80%) is outperformed by massive generalist models like GPT-5 (62.20%) and InternVL3-78B (61.40%). Furthermore, CARE-Flow-B (56.60%) is beaten by Llama-3.2-11B-Vision (57.40%).
4. **Suboptimal Segmentation Integration**: When the authors swapped their custom entity segmentation model for a general-purpose one (BiomedParse), the downstream medical VQA performance dropped by a quantitative 3.4%.
