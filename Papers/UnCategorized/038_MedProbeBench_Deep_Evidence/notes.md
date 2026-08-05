# MedProbeBench: Systematic Benchmarking at Deep Evidence Integration for Expert-level Medical Guideline
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.18418

# Problem Statement
Current medical benchmarks (like MedQA) test rote memorization. Medical agents need to be evaluated on their ability to retrieve, synthesize, and resolve conflicts across multiple lengthy, complex clinical guidelines (e.g., NCCN Oncology guidelines).

# Key Contributions
1. Introduces MedProbeBench, focusing on "Deep Evidence Integration" rather than simple QA.
2. Evaluates the multi-hop reasoning capabilities of agents querying 100+ page clinical guideline PDFs.
3. Reveals a massive capability gap: SOTA RAG agents drop critical contraindications when synthesizing from multiple documents.

# Methodology / Architecture
The benchmark requires the agent to answer complex patient scenario questions where the answer relies on chaining evidence from Guideline A (Primary Treatment) and Guideline B (Contraindications for Comorbidities). Evaluates Retrieval Precision, Synthesis Accuracy, and Conflict Resolution.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedProbeBench | Multi-document guideline QA | 1,500 expert-crafted queries |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Multi-hop Evidence | Synthesis F1 | 55.4% | N/A | Establishes new difficult baseline |

# Limitations & Future Work
The benchmark is currently limited to English guidelines and highly structured oncology/cardiology domains.

# Relevance to Our Survey
Crucial for **Benchmarking Frameworks** and **Knowledge Retrieval**. Uncovers a major gap in multi-hop RAG synthesis for expert-level clinical agents. Modality: Clinical Text (Guidelines).
