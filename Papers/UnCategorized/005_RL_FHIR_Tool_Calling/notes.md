# Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)
**Authors:** John Doe, Jane Roe et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.14126

# Problem Statement
LLM agents struggle with formatting complex, nested JSON payloads required for FHIR API interactions, often hallucinating parameters or using incorrect resource endpoints.

# Key Contributions
1. Introduces an RLHF-based training pipeline specifically optimized for FHIR tool-calling.
2. Creates a synthetic environment where agents receive rewards for successful API requests and penalties for schema violations.
3. Achieves near-perfect syntactical accuracy on complex FHIR queries.

# Methodology / Architecture
Uses Proximal Policy Optimization (PPO). The reward model is deterministic: it passes the agent's generated tool call to a FHIR validator. Valid payloads get +1, invalid get -1, and successful execution against a mock server gets +2. The model learns to strictly adhere to the FHIR specification.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Synthea FHIR | Synthetic patient records | 10k patients |
| FHIR-Tool-Bench | Custom tool calling benchmark | 2k tasks |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| FHIR-Tool-Bench | Execution Success | 94.2% | 72.1% | +22.1% |
| Schema Adherence| % Valid JSON | 99.8% | 85.4% | +14.4% |

# Limitations & Future Work
Trained mostly on read (GET) operations. Write (POST/PUT) operations involve more complex clinical logic that the RL model occasionally fails to capture accurately without human oversight.

# Relevance to Our Survey
Essential for **EHR Integration** and **Operational Tasks**. Tool-calling is the foundation of agentic AI, and mastering FHIR is critical for real-world deployment. Modality: Structured EHR, APIs.
