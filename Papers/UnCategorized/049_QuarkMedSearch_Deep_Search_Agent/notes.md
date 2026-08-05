# QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.12867

# Problem Statement
Standard search engines and basic RAG pipelines fail at answering highly complex, exploratory medical research queries (e.g., "What are the emerging molecular pathways linking gut microbiome dysbiosis to neuroinflammation in early-onset Alzheimer's?") because they require long-horizon, multi-step web navigation and synthesis.

# Key Contributions
1. Introduces QuarkMedSearch, a long-horizon agent designed for deep medical literature exploration.
2. The agent autonomously plans queries, navigates PubMed/bioRxiv via APIs, reads full texts, follows citation trails (snowballing), and synthesizes findings over a 20-30 minute execution window.
3. Radically outperforms standard single-query RAG on deep exploratory research questions.

# Methodology / Architecture
Utilizes a Tree-of-Thoughts (ToT) planning module. The agent explores multiple search branches simultaneously. A "Critic Agent" evaluates if a branch is yielding relevant papers; if not, it prunes the branch and redirects the search strategy. Synthesizes massive context windows via hierarchical summarization.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| DeepMed-QA | Complex exploratory questions | 500 expert queries |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Standard RAG)| Improvement |
|----------------|--------|------------|-----------------------------|-------------|
| Research Synthesis| Expert Eval Score (1-5)| 4.6 | 2.8 | +1.8 (Massive leap) |

# Limitations & Future Work
High token consumption and API costs per query. Susceptible to getting stuck in "rabbit holes" if the Critic Agent fails to prune irrelevant biological pathways.

# Relevance to Our Survey
Highly relevant for **Research & Systematic Review Agents**. Proves that long-horizon, autonomous search agents are required to close the gap between simple factual QA and deep medical intelligence synthesis. Modality: Scientific Literature.
