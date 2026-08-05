# BLUEmed: Retrieval-Augmented Multi-Agent Debate for Clinical Error Detection
**Authors:** N/A
**Year:** 2026 | **Venue:** IEEE ICHI
**ArXiv/DOI:** https://arxiv.org/abs/2604.10389

# Problem Statement
Medical error detection in clinical notes (e.g., wrong drug dosages, contraindicated treatment plans) requires fact-checking against external guidelines, which a single LLM struggles to do comprehensively.

# Key Contributions
1. Introduces BLUEmed, a multi-agent debate framework specifically augmented with a RAG vector database of FDA drug labels and clinical guidelines.
2. Agents explicitly debate whether a proposed treatment plan violates any retrieved constraints.
3. Significantly outperforms zero-shot GPT-4 in detecting iatrogenic errors in simulated EHRs.

# Methodology / Architecture
A "Reviewer Agent" scans the clinical note and queries the RAG database for relevant guidelines. A "Critic Agent" takes the retrieved guidelines and debates the Reviewer on whether a violation occurred. A "Senior Attending Agent" synthesizes the debate and flags the error.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedError-Detect | Synthetic notes with hidden errors| 4,000 notes |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single RAG)| Improvement |
|----------------|--------|------------|---------------------------|-------------|
| Error Detection| F1 Score | 82.4% | 67.1% | +15.3% |

# Limitations & Future Work
Highly dependent on the quality of the embedding model used for the initial RAG retrieval; if the correct FDA label isn't retrieved, the debate is useless.

# Relevance to Our Survey
Highly relevant for **EHR & Operational** and **Clinical Decision Support**. Combines RAG with Multi-Agent Debate to solve the clinical error detection gap. Modality: Clinical Text.
