# MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic RL
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.03320

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Cumulative Latency**: A major unsolved gap for multi-turn iterative paradigms is the increased cumulative latency during inference. Since the agent continuously loops through thought-action-observation steps, it is too slow for real-time surgical or emergency workflows.
2. **Failure in Low-Contrast Modalities**: The agent struggles to achieve precise localization in challenging volumetric modalities like CT and MRI. The vast search spaces and low tissue contrast cause the agent to wander or fail to place effective clicks without heavy reward-shaping.
3. **Reward Function Dependency**: The system relies heavily on carefully crafted reward functions (e.g., $R_{imp}$ and $R_{over}$). If these are miscalibrated or removed, the agent falls into infinite loops, fails to terminate at optimal states, and loses mask fidelity.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Zero-shot / Base Model Failures**: When tested with out-of-the-box base models (like GPT-4o) without explicit fine-tuning, the Dice scores completely collapse on CT (Dice: 0.122) and MRI (Dice: 0.549), failing to match supervised benchmarks.
2. **Ambigous Boundary Breakdown**: In voluminous or ambiguously bounded cases (like diffuse tumors), the agent fails to effectively use bounding boxes as human-like anchors for global context, leading to suboptimal, overly conservative segmentation outputs compared to human experts.
