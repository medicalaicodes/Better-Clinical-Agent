# CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.01113

# Problem Statement
When agents encounter "Evidence Discordance" (e.g., the patient's EHR says they are not pregnant, but their recent lab results show high hCG levels), standard agents often leak PII (Personally Identifiable Information) to external search APIs when trying to resolve the conflict.

# Key Contributions
1. Introduces CARE, an agentic reasoning protocol that enforces strict PII scrubbing *before* any tool calls are made to external knowledge bases.
2. Resolves discordant EHR evidence by utilizing a local, privacy-safe "Hypothesis Generator" before querying external APIs with anonymized keywords.

# Methodology / Architecture
An Anonymizer Agent strips all identifiers. The Discordance-Resolution Agent identifies the conflict (e.g., hCG vs. EHR status) and formulates abstract, anonymized medical queries (e.g., "causes of elevated hCG in non-pregnant females") to search PubMed, ensuring zero PII leaves the local hospital server.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Discordant-EHR-500 | Synthetic conflicting records | 500 records |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Standard RAG) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| Privacy Leakage| % of API calls w/ PII | 0.0% | 14.2% | Perfect compliance |
| Conflict Res   | Accuracy | 88.1% | 85.0% | +3.1% |

# Limitations & Future Work
Heavy anonymization sometimes strips out critical temporal context (e.g., exact dates of a procedure) which is needed to resolve the discordance.

# Relevance to Our Survey
Crucial for **EHR & Operational** and **Safety/Ethics**. Solves the HIPAA/PII leakage problem that prevents most hospitals from deploying tool-using agents. Modality: Clinical Text.
