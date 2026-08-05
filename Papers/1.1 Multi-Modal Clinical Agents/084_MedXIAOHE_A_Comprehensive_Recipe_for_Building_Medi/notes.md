# MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.12705

# 1. Unsolved Gaps (Clinical Deployment Risks)
- **Ontological Overfitting**: The paper's semantic coverage metric (AMCS) reveals a significant "Backward Gap" (0.68-0.79), indicating the MET captures many entities missing from public datasets, but also highlighting a lack of standardization. This proprietary taxonomy risks poor transferability to external, standardized healthcare IT systems.
- **Model-based Hallucination**: The mid-training and post-training pipelines rely heavily on LLM-synthesized data (e.g., multi-expert reject sampling, synthetic instruction-following). In high-stakes clinical environments, reasoning derived from synthetic loops rather than ground-truth clinical reality remains susceptible to hallucination.

# 2. Benchmark Failures (Quantitative Shortfalls)
- **Real-World Degradation**: While the model scores highly on sanitized public benchmarks (e.g., MMMU_val-Med at 87.53%), performance degrades significantly on uncurated, real-world data like "Inhouse OCR" (which intentionally includes real-world distortion, blur, and clutter) and "Inhouse VQA".
- **Complex Diagnostic Tasks**: The model underperforms significantly on extremely complex diagnostic reasoning benchmarks compared to SOTA proprietary models. For instance, on the HLE-Med benchmark, MedXIAOHE scores 25.77% against Gemini 3.0 Pro's 36.91%. It also experiences quantitative shortfalls in complex, multi-turn clinical diagnostic tasks (e.g., lower scores on 'HealthBench-hard' compared to 'HealthBench-normal').
