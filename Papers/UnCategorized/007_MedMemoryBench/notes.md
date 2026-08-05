# MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare
**Authors:** R. Gupta et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.11814

# Problem Statement
Medical agents often suffer from "amnesia" across long-term patient interactions. Current benchmarks evaluate single-turn accuracy, completely ignoring an agent's ability to retrieve and synthesize historical patient data over months or years.

# Key Contributions
1. Introduces MedMemoryBench, the first dataset specifically targeting long-term memory in medical agents.
2. Identifies the "Temporal Gap" where agents confuse recent symptoms with historical baseline conditions.
3. Proposes a Memory-RAG framework with time-decay weighting.

# Methodology / Architecture
The benchmark simulates multi-visit patient trajectories. Agents are evaluated on their ability to answer questions that require synthesizing information from visit 1, visit 4, and visit 10. The proposed baseline uses a hierarchical memory structure: Short-term (current visit) and Long-term (vector DB of past visits with temporal encoding).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedMemoryBench | Synthetic longitudinal EHR | 10k patient trajectories, 5-20 visits each |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Longitudinal QA | Accuracy | 78.4% | 52.1% (Standard RAG) | +26.3% |

# Limitations & Future Work
The benchmark is currently text-only. Future versions should incorporate longitudinal lab value trends and imaging.

# Relevance to Our Survey
Crucial for identifying the **Memory Gap**. Highlights how agents need temporal awareness for personalized healthcare. Modality: Clinical Text (Longitudinal).
