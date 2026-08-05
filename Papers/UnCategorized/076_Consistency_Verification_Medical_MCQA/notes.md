# Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.24481

# Problem Statement
Medical LLMs often suffer from severe overconfidence, providing incorrect diagnoses with 99% probability. Poor uncertainty calibration is a critical safety failure in clinical settings.

# Key Contributions
1. Proposes a Multi-Agent Consistency Verification framework.
2. Instead of single-pass generation, multiple agent instances independently solve a medical multiple-choice question. A Verification Agent then checks if the rationales are logically consistent across instances.
3. If the rationales diverge (even if the final answer matches), the system drastically lowers its confidence score, resulting in perfectly calibrated uncertainty.

# Methodology / Architecture
Three parallel Generator Agents provide an answer and a Chain-of-Thought (CoT). The Verifier Agent uses Natural Language Inference (NLI) to determine if the CoTs entail, contradict, or are neutral to each other. The final output probability is a function of both the answer voting and the NLI consistency score.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedQA & MedMCQA | Standard medical exams | 15k questions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single Agent) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| Calibration    | Expected Calibration Error (ECE) | 0.04 | 0.15 | 3.75x lower (better) |

# Limitations & Future Work
Computationally expensive, requiring 4x the inference cost per query compared to a standard LLM pass.

# Relevance to Our Survey
Relevant for **Clinical Decision Support**. Addresses a crucial safety aspect (Calibration) by using multi-agent variance as a proxy for epistemic uncertainty. Modality: Clinical Text.
