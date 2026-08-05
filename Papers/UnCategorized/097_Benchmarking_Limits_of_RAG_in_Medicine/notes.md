# Benchmarking the Limits of RAG in Medicine: Why Agentic Retrieval is Necessary for Long-Horizon Cases
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.06451

# Problem Statement
Standard Retrieval-Augmented Generation (RAG) fails in medicine when the answer requires multi-hop reasoning across temporal events (e.g., "Did the patient's kidney failure start before or after starting lisinopril?").

# Key Contributions
1. Demonstrates the empirical failure of single-shot vector DB retrieval on temporal clinical questions.
2. Proposes "Agentic Retrieval", where an agent iteratively queries the DB, reads the returned chunks, and formulates new, more specific search queries until it resolves the timeline.
3. Introduces Long-Horizon-Med-QA, a benchmark specifically designed to break standard RAG.

# Methodology / Architecture
Evaluation/Methodology paper. Compares standard dense retrieval (Contriever/BGE) against an Iterative Search Agent (using ReAct prompting). 

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Long-Horizon-Med-QA | Multi-hop temporal clinical QA | 1,500 complex cases|

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agentic Retrieval) | Standard RAG | Improvement |
|----------------|--------|--------------------------------|--------------|-------------|
| Long-Horizon QA| Exact Match | 76.5% | 42.1% | Massive (+34.4%) |

# Limitations & Future Work
Agentic retrieval is slow, taking 30-40 seconds per question as the agent executes multiple search loops, limiting real-time point-of-care use.

# Relevance to Our Survey
Crucial for **Methodology & Taxonomy**. Proves why simple RAG is insufficient for medicine and why autonomous, multi-step agentic search is required. Modality: Clinical Text.
