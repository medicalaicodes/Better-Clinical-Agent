# Statistics, Not Scale: Modular Medical Dialogue with Bayesian Belief Engine
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.20022

# Problem Statement
Massive monolithic LLMs (e.g., GPT-4 class) are computationally expensive and prone to uninterpretable hallucinations during medical dialogue. Scaling model size does not fundamentally solve probabilistic reasoning flaws in diagnostic interviews.

# Key Contributions
1. Proposes a modular architecture combining a small, fast LLM for natural language generation with a discrete "Bayesian Belief Engine" (BBE).
2. The BBE explicitly tracks the probability distribution of potential diseases based on symptom presence/absence during dialogue.
3. Outperforms 1T-parameter models on diagnostic accuracy using only a 7B-parameter LLM + BBE.

# Methodology / Architecture
The LLM acts as the conversational interface. It extracts symptoms from the user's text via structured parsing and passes them to the BBE. The BBE recalculates disease probabilities using a clinical prior matrix and outputs the "Next Best Question" to ask to maximize information gain. The LLM translates that back to natural dialogue.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| BayesDX-Dial      | Multi-turn diagnostic dialogue | 15k conversations |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (GPT-4) | Improvement |
|----------------|--------|------------|-----------------------|-------------|
| Diagnostic Acc | Top-3 Accuracy | 89.1% | 85.4% | +3.7% |
| Efficiency     | Inference Latency| 200ms | 1,500ms | 7.5x faster |

# Limitations & Future Work
The BBE relies on manually curated probability matrices (Symptom given Disease), which are incredibly difficult to maintain for rare diseases.

# Relevance to Our Survey
Highly relevant for **Patient-Facing Copilots** and **Taxonomy**. Highlights the architectural shift away from monolithic LLMs towards Neuro-Symbolic Agentic frameworks (LLM + Probabilistic Engines). Modality: Text, Structured Data.
