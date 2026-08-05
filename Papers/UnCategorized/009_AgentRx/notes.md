# AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks
**Authors:** M. Chen et al.
**Year:** 2026 | **Venue:** CHIL
**ArXiv/DOI:** https://arxiv.org/abs/2605.10286

# Problem Statement
Clinical prediction (e.g., mortality, readmission) relies heavily on either structured data (XGBoost) or unstructured notes (NLP). Agents offer a way to fuse multimodal data by orchestrating specific models, but there is no unified benchmark to test this multimodal orchestration.

# Key Contributions
1. Introduces AgentRx, benchmarking LLM agents on their ability to fuse tabular EHR, clinical notes, and time-series vitals.
2. Demonstrates that a "ReAct" style agent coordinating a tabular model and an NLP model outperforms monolithic multimodal LLMs.

# Methodology / Architecture
The AgentRx environment requires the LLM to act as a primary care physician. It is given a patient's ID and must decide whether to query the structured database (SQL tool) or the clinical notes (RAG tool) to predict 30-day readmission risk.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| AgentRx | Multimodal prediction benchmark derived from MIMIC-IV | 50k patients |

# Key Results & Metrics
| Task/Benchmark | Metric | Agentic Approach | Monolithic MLLM | Improvement |
|----------------|--------|------------|---------------|-------------|
| Readmission Pred| AUROC | 0.81 | 0.76 | +0.05 |

# Limitations & Future Work
The agent can get stuck in infinite loops querying the same database repeatedly. Needs better loop detection.

# Relevance to Our Survey
Highly relevant for **EHR & Operational Tasks**. Benchmarks multimodal data fusion via tool-calling. Modality: Multi-modal (EHR Tabular, Text).
