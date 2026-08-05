# Agentic Knowledge Graphs: Dynamic Verification of Medical Claims through Automated Literature Review
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.04218

# Problem Statement
Medical knowledge graphs (like UMLS) are static and become outdated quickly as thousands of new papers are published daily.

# Key Contributions
1. Proposes "Agentic Knowledge Graphs" (AKG), where nodes in the graph are active agents rather than static data points.
2. When a new claim is introduced (e.g., "Drug X treats Disease Y"), a Verification Agent is dispatched to search PubMed, read recent papers, and vote on whether the claim should be added to the graph.
3. The graph dynamically updates itself and resolves conflicting evidence through agent debate.

# Methodology / Architecture
A Graph-Agent Architecture. Combining Knowledge Graphs with Multi-Agent Systems. Each edge in the graph has a "Confidence Score" that is continuously re-evaluated by specialized literature-review agents.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| PubMed Abstracts| Daily scientific literature | Streaming data |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agentic KG) | Static KG (UMLS) | Improvement |
|----------------|--------|-------------------------|------------------|-------------|
| KG Freshness   | Time to include new finding | 2.5 days | 8-12 months | Exponentially faster |

# Limitations & Future Work
Agents sometimes get tricked by low-quality, high-volume predatory journal publications, polluting the knowledge graph with false claims. Needs a strict journal-quality filter.

# Relevance to Our Survey
Relevant for **Research Agents** and **Methodology**. Shows how agents can maintain the underlying infrastructure (knowledge bases) that other medical AI systems rely on. Modality: Scientific Text.
