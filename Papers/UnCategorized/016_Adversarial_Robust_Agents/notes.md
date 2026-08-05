# Research on Security Enhancement Methods for Adversarial Robust Large Language Model Intelligent Agents for Medical Decision-Making Tasks
**Authors:** N/A
**Year:** 2026 | **Venue:** AINIT
**ArXiv/DOI:** https://arxiv.org/abs/2605.08257

# Problem Statement
Medical LLM agents are highly susceptible to adversarial prompt injection and data poisoning, which can lead to catastrophic medical errors. Evaluating robustness is a major gap.

# Key Contributions
1. Proposes a red-teaming benchmark for medical agents testing adversarial resilience.
2. Introduces an "Agentic Firewall" mechanism to sanitize clinical inputs before LLM processing.
3. Quantifies the degradation of SOTA agents under targeted adversarial noise.

# Methodology / Architecture
The framework employs a proxy agent designed to inject adversarial perturbations into simulated EHR data. The target agent's responses are evaluated for clinical safety violations.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Med-Adversarial | Perturbed EHR dataset | 1k cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Robustness | Safety Violation Rate | 4.2% | 18.5% | -14.3% (Fewer errors) |

# Limitations & Future Work
The Agentic Firewall adds inference latency, making it less suitable for real-time robotic surgery applications.

# Relevance to Our Survey
Addresses a major gap: **Security & Safety Benchmarking** in Agentic AI. Modality: Clinical Text.
