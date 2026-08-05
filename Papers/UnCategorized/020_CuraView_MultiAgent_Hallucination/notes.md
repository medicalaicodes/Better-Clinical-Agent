# CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.03476

# Problem Statement
Hallucinations in medical LLMs are deadly. Traditional RAG relies on vector similarity, which often retrieves conceptually similar but factually contradictory medical guidelines.

# Key Contributions
1. Proposes CuraView, a multi-agent system utilizing GraphRAG (Knowledge Graphs) for rigorous fact-checking.
2. The "Critic Agent" cross-references the primary LLM's output against a massive medical knowledge graph.
3. Drastically reduces ungrounded clinical claims in generated summaries.

# Methodology / Architecture
Instead of flat vector retrieval, CuraView parses guidelines into a Knowledge Graph (nodes=diseases/drugs, edges=relationships). The generator agent creates a response, and the critic agent parses it, maps it to the Graph, and flags any unverified edges as hallucinations.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedHalDetect | Medical hallucination benchmark | 5k summaries |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Hallucination Det | F1 Score | 82.3% | 68.7% (Vector RAG) | +13.6% |

# Limitations & Future Work
Constructing and updating the Knowledge Graph is highly resource-intensive and requires constant manual expert curation.

# Relevance to Our Survey
Highly relevant for **Knowledge Retrieval & Hallucination Mitigation**. Highlights the shift from Vector RAG to GraphRAG in medicine. Modality: Clinical Text, Knowledge Graphs.
