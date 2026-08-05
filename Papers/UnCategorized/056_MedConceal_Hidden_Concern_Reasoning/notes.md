# MedConceal: A Benchmark for Clinical Hidden-Concern Reasoning Under Partial Observability
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.08788

# Problem Statement
Standard diagnostic benchmarks present the LLM with a complete list of symptoms upfront. In reality, patients often conceal symptoms due to embarrassment, stigma, or cognitive decline (partial observability).

# Key Contributions
1. Introduces MedConceal, the first benchmark testing an agent's ability to deduce and actively probe for "hidden concerns" based on subtle linguistic cues and demographic risk factors.
2. Evaluates the "Tactfulness" of the agent's probing questions.
3. Reveals that standard LLMs aggressively diagnose based only on overt symptoms, missing critical underlying issues like substance abuse or domestic violence.

# Methodology / Architecture
The benchmark simulates a patient agent initialized with a "Hidden Concern". The diagnostic agent must interview the patient. The diagnostic agent is scored on its ability to trigger the patient agent into revealing the concern through empathetic, non-judgmental, and targeted questioning.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedConceal-Set | Simulated interviews w/ hidden states| 1,200 scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Standard Chatbot) | Improvement |
|----------------|--------|------------|----------------------------------|-------------|
| Hidden Discovery | Discovery Rate % | 64.2% | 22.1% | +42.1% |

# Limitations & Future Work
The hidden states are currently limited to behavioral health and stigmatized infectious diseases; needs expansion to vague autoimmune conditions.

# Relevance to Our Survey
Highly relevant for **Benchmarking Frameworks** and **Patient-Facing Copilots**. Addresses the gap of "partial observability" in real-world clinical interviews. Modality: Text (Dialogue).
