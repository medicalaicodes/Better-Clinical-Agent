# MedSAM3: Delving into Segment Anything with Medical Concepts
**Authors:** N/A
**Year:** 2025 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2511.19046

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Semantic Misalignment & Terminology**: The model fails to distinguish fine-grained terms (e.g., failing on 'nuclei' but working on 'cell') and frequently misaligns regions, such as segmenting lungs when prompted for 'liver' or surrounding non-lesion skin when prompted for 'lesion'.
2. **Reliance on Geometric Cues**: Text-only prompting (MedSAM-3 T) remains highly unreliable for real clinical deployment. Bounding boxes are still practically required to achieve acceptable accuracy.
3. **Agent Bottleneck**: The MedSAM-3 Agent's iterative refinement effectiveness severely diminishes if the underlying base model is fundamentally misaligned semantically.
4. **Data Scarcity**: The model's adaptation ceiling is constrained by a lack of large-scale medical datasets with rich, high-quality clinical text descriptions.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Text-Only Crash**: MedSAM-3 T (pure text prompt) scores abysmally on 2D benchmarks: 0.2674 on BUSI, 0.0826 on RIM-ONE(Cup), and 0.1441 on Kvasir-SEG, falling far behind standard U-Net (~0.76-0.84).
2. **3D Volumetric Failure**: SAM 3 T completely breaks down on complex 3D datasets, achieving a Dice of 0.1374 on LiTS (vs U-Mamba's 0.7910) and 0.3033 on ISLES2024 (vs nn-Unet's 0.7718).
3. **ISIC 2018 Regression**: Even with text+box prompts (MedSAM-3 T+I), it scores 0.9058 on ISIC 2018, which fails to beat the older MedSAM baseline of 0.9177.
4. **Concept Fragility**: Shifting the prompt from 'cell' to 'nuclei' on the MoNuSeg dataset causes the Dice score to plummet from 0.6786 to exactly 0.0.
