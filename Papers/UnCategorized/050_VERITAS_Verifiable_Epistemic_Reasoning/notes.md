# VERITAS: Verifiable Epistemic Reasoning for Image-Derived Hypothesis Testing via Agentic Systems
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.12144

# Problem Statement
Current Medical VLMs output diagnostic probabilities but cannot explain their *epistemic uncertainty* (what they don't know). They cannot actively form hypotheses and test them against the visual evidence in a verifiable manner.

# Key Contributions
1. Introduces VERITAS, an agentic system that performs explicit hypothesis testing on medical images.
2. The agent states a hypothesis (e.g., "If this is pneumonia, I should see air bronchograms in the consolidation").
3. It then uses targeted visual cropping and zooming tools to explicitly look for that evidence, updating its belief state verifiably.

# Methodology / Architecture
The LLM acts as the epistemic reasoner. It proposes hypotheses based on the global image view. It then controls a Visual Attention Tool to take high-resolution crops of specific anatomical regions. An external Verification module checks if the agent's textual claim matches the visual features found in the crop.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Epistemic-Rad | Chest X-Rays w/ uncertainty labels| 5,000 images |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Standard VLM) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| Diagnostic Acc | AUROC | 0.88 | 0.83 | +0.05 |
| Hallucination Rate| False Visual Claims | 4.2% | 19.8% | -15.6% (More verifiable) |

# Limitations & Future Work
Requires the underlying VLM to perfectly understand spatial coordinates to drive the zooming/cropping tool accurately.

# Relevance to Our Survey
Relevant for **Medical Vision** and **Security/Safety**. Addresses the gap of unexplainable VLM hallucinations by enforcing an active, tool-using hypothesis testing framework. Modality: 2D X-Ray, Text.
