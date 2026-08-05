# SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.26122

# Problem Statement
Dermatological AI models are notorious for poor out-of-distribution performance (e.g., failing on different skin tones). They are static and do not adapt to new, rare skin conditions once deployed.

# Key Contributions
1. Proposes SkinGPT-X, a multi-agent system that updates its own retrieval database.
2. If the "Diagnostic Agent" encounters a lesion with low confidence, it triggers a "Human-in-the-Loop Agent" to ask a dermatologist. The result is synthesized and permanently added to the vector store.
3. Over time, the system self-evolves, significantly improving performance on rare diseases and diverse Fitzpatrick skin types.

# Methodology / Architecture
A VLM acts as the primary visual analyzer. A RAG system pulls past similar cases. A Collaboration Agent determines if the confidence is high enough to output a diagnosis. If not, it halts and queries a human. A Synthesis Agent updates the DB post-human feedback.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ISIC 2024 / Fitzpatrick-17k | Diverse skin lesions | 25k images |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (After 1mo deployment) | Baseline (Day 0) | Improvement |
|----------------|--------|-----------------------------------|------------------|-------------|
| Rare Disease | Top-3 Acc | 82.1% | 61.4% | +20.7% |
| Darker Skin Tones | AUC | 0.89 | 0.76 | +0.13 |

# Limitations & Future Work
Requires a dedicated team of dermatologists to answer the Human-in-the-Loop queries during the initial "cold start" phase of deployment.

# Relevance to Our Survey
Highly relevant for **Medical Vision** and **Continual Learning**. Proves that "Self-Evolving" architectures via human-in-the-loop agents solve the out-of-distribution and racial bias problems in visual AI. Modality: 2D Images (Dermatology).
