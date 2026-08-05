# SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning
**Authors:** Alice Chen, Bob Smith et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.17101

# Problem Statement
Medical reasoning often requires complex, multi-hop retrieval of external knowledge. Existing RAG pipelines rely on single-agent linear reasoning, which struggles with ambiguous medical queries and fails to self-correct when retrieving conflicting or irrelevant information from clinical guidelines.

# Key Contributions
1. Introduces SEMA-RAG, a multi-agent framework comprising Interpreter (I-Agent), Explorer (E-Agent), and Arbiter (A-Agent).
2. Proposes a self-evolving retrieval loop where the E-Agent iteratively refines search queries based on the Arbiter's feedback.
3. Achieves SOTA performance on MedQA and PubMedQA benchmarks compared to single-agent RAG.

# Methodology / Architecture
The framework uses three specialized agents:
- **I-Agent (Question Interpreter):** Parses complex medical queries into sub-questions.
- **E-Agent (Knowledge Explorer):** Interfaces with medical databases (PubMed, clinical guidelines). It evaluates retrieved documents and autonomously reformulates queries if evidence is insufficient.
- **A-Agent (Evidence Arbiter):** Adjudicates retrieved evidence, resolves conflicts, and constructs the final grounded response.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedQA-US | USMLE-style questions | ~12k QA pairs |
| PubMedQA | Biomedical research QA | ~1k Expert QA |
| MMLU-Med | Clinical knowledge subset | ~5k questions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| MedQA-US | Accuracy | 89.4% | 85.2% (Med-PaLM 2) | +4.2% |
| PubMedQA | Accuracy | 81.2% | 78.5% | +2.7% |

# Limitations & Future Work
The multi-agent debate increases inference latency and API costs significantly. Future work aims to distill the multi-agent reasoning paths into a single, smaller model.

# Relevance to Our Survey
Highly relevant for the **Clinical Decision Support** and **Knowledge Retrieval** categories. Showcases the shift from single-pass RAG to multi-agent iterative RAG in medicine. Modality: Clinical Text.
