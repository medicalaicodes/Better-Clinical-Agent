# CoMMa: Contribution-Aware Medical Multi-Agents From A Game-Theoretic Perspective
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.09159

# 1. Unsolved Gaps (Real Clinical Deployment Hurdles)
1. **Lack of Open-Ended Reasoning:** The model strictly formulates tumor board decisions as discrete multiple-choice classifications with clear ground-truth labels. Real-world multidisciplinary clinical decision-making is inherently open-ended, ambiguous, and requires handling partially observed or uncertain outcomes.
2. **Absence of Narrative Explanations:** The authors explicitly note that CoMMa does not produce fine-grained narratives, providing only quantitative attribution (logits converted to percentiles). In actual clinical deployments, clinicians typically require textual rationales to trust and verify the AI's complex medical reasoning.
3. **Clustered Modality Entanglement:** The model struggles to achieve granular reasoning when processing highly clustered information across varying modalities, suggesting it cannot fully disentangle complex, interconnected patient evidence.
4. **Data Diversity Limitations:** The system's evaluation is heavily constrained to specific oncology cohorts (HCC Tumorboard and MTBBench) due to the scarcity of diverse real-world tumor board datasets, raising questions about its generalizability across different cancers or institutions.

# 2. Benchmark Failures (Quantitative Shortcomings)
1. **Poor Discriminative Ability on Progression:** On the MTBBench dataset for Tumor Progression Prediction, CoMMa achieves an AUC of only 0.595 ± 0.061. This indicates extremely weak discriminative capability, performing only slightly better than random chance.
2. **High Instability on Recurrence:** For Tumor Recurrence Prediction on MTBBench, while the raw accuracy is 0.818, the AUC is 0.656 with a massive variance of ± 0.204. This high standard deviation demonstrates severe instability and unreliability across different data subsets or patient cases.
3. **Marginal Accuracy Gains from Multi-Agent Scaling:** In the ablation studies on the HCC Tumorboard, transitioning from a single deterministic agent to the full 5-agent CoMMa system only marginally improves accuracy (from 0.826 to 0.831), calling into question the cost-benefit ratio of the multi-agent orchestration for accuracy alone.
