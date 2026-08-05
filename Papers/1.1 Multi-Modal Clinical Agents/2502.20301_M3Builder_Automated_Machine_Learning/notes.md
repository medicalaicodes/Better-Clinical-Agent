# M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging
**Authors:** N/A
**Year:** 2025 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2502.20301

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Lack of Native Visual Processing:** The agents operate by manipulating code and reading free-text descriptions of datasets, but they currently lack visual processing capabilities to natively "see" and interpret the scans like a clinician would.
2. **Dependency on Structured Workspaces:** The system heavily relies on pre-configured, highly curated "medical imaging ML workspaces" with well-crafted few-shot instructions. Real hospital environments have messy, unstructured, and fragmented data ecosystems.
3. **Automated Dataset Preparation:** The framework is not yet capable of fully automated raw clinical dataset preparation out-of-the-box. It requires human curation to map datasets to the structured format M³Builder expects.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Task-Specific Struggles:** In M³Bench evaluations (using Claude-3.7-Sonnet), organ segmentation proved to be the most challenging, recording the lowest success rate (86.67%) compared to tasks like anomaly detection (100%).
2. **Model Sensitivity:** The framework is highly sensitive to the underlying LLM. While Claude 3.7 and GPT-4o performed well, using other models like Llama 3.3 or Gemini 2.0 caused the success rate to plummet to an abysmal 4.29%.
3. **Architectural Fragility:** Ablation studies showed that without multi-agent collaboration, performance drops by 42.85%. Removing example instructions results in a 25.00% drop. Furthermore, achieving error-free code organization and execution within a strict 5-iteration limit remains a frequent failure modality. Curiously, self-reflection mechanisms were found to have minimal impact on system performance.
