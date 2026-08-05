# IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2601.03054

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Computational Overhead:** The multi-step agentic loop induces significant latency (28.70s per case), unsuitable for time-sensitive clinical workflows.
2. **2D-Centric Limitation:** Lacks native 3D volumetric reasoning for CT/MRI diagnostics.
3. **Tool Dependency:** Success is bound by the frozen segmentation tool's capabilities (e.g., MedSAM2).

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Low-Contrast Boundaries:** Quantitative performance degradation observed in complex tumor and lesion segmentation tasks (e.g., MSD hepatic vessel tumor with Dice ~0.74-0.85, and siim-acr-pneumothorax) compared to routine organ segmentation (e.g., Liver, Dice ~0.97), indicating a failure to generalize to low-contrast, non-distinct lesion boundaries.
