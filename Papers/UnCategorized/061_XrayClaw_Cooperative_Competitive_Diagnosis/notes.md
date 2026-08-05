# XrayClaw: Cooperative-Competitive Multi-Agent Alignment for Trustworthy Chest X-ray Diagnosis
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.02695

# Problem Statement
Current Vision-Language Models (VLMs) for X-rays suffer from "yes-man" behavior; if prompted with a leading question ("Is there pneumonia?"), the model will hallucinate features to satisfy the user. 

# Key Contributions
1. Uses a "Cooperative-Competitive" multi-agent framework to force alignment and calibrate confidence.
2. The "Cooperative Agent" tries to find evidence for the user's prompt. The "Competitive Agent" is explicitly rewarded for proving the Cooperative Agent wrong.
3. This friction results in highly trustworthy, un-biased diagnostic reports.

# Methodology / Architecture
Both agents analyze the same image embeddings. They exchange textual arguments for 3 rounds. A final "Judge Agent" determines the winner. The framework uses the OpenClaw alignment protocol to prevent the debate from looping indefinitely.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MIMIC-CXR | Chest X-Rays | 377k images |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single VLM) | Improvement |
|----------------|--------|------------|----------------------------|-------------|
| Diagnosis      | F1 (Macro)| 0.81 | 0.74 | +0.07 |
| Yes-Man Bias   | False Positive Rate on Leading Qs | 4.1% | 27.5% | Massive reduction |

# Limitations & Future Work
The competitive agent sometimes becomes hyper-skeptical, leading to false negatives for very subtle, early-stage pathologies.

# Relevance to Our Survey
Highly relevant for **Medical Vision** and **Clinical Decision Support**. Demonstrates that multi-agent adversarial alignment is necessary to cure the "sycophancy" bias in medical foundation models. Modality: 2D X-Ray.
