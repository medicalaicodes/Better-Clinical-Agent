# 3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.18064

# 1. Unsolved Gaps (Failures in Real Clinical Deployment)
*   **Dependency on Pretrained Visual Tools**: The agent’s early stages (Organ-Aware Memory Initialization and Coarse-to-Fine Lesion Targeting) are heavily bounded by the quality of underlying visual tools like VISTA3D (for segmentation) and CT-CLIP (for lesion localization). In real clinical deployments where domain shifts are common, errors from these models will propagate to downstream reasoning, causing suboptimal evidence initialization and incomplete lesion discovery.
*   **Fundamental 3D Spatial and Relational Blindspots**: Although the "Think-with-1-Slice Loop" (T1S-Loop) is effective for simple perceptual verification (e.g., checking lesion presence or size), it forces the 2D MLLM to reason from a limited set of 2D slices. This inherently prevents the model from achieving complex anatomical or relational reasoning that requires fine-grained, holistic comprehension of multiple medical structures in a 3D context.
*   **Rigid Zero-Shot Framework**: The agent currently operates strictly via zero-shot prompting. Its routing strategy, evidence aggregation, and tool selection are fixed and do not adapt or optimize through learning mechanisms (like RL or supervised fine-tuning), limiting its ability to handle complex or novel clinical scenarios iteratively.

# 2. Benchmark Failures (Quantitative Shortfalls)
*   **"Adjacent-Organ" Task Failure**: In the DeepTumorVQA benchmark, 3DMedAgent explicitly underperforms compared to strong general MLLM baselines on the `adjacent-organ` visual reasoning sub-task (e.g., "Which organ lies adjacent to the largest pancreatic lesion?"). 
*   **Reasoning for the Failure**: Because 3DMedAgent is forced to construct its answers from explicit visual evidence across a few 2D slices, it struggles to accurately interpret spatial relationships between adjacent organs. General MLLMs paradoxically score higher on this task by exploiting prior medical knowledge (heuristics about typical anatomical organ adjacency) rather than relying on genuine visual perception, exposing a fundamental trade-off where 3DMedAgent's evidence-grounded approach fails when visual cues are sparse or hard to observe.
