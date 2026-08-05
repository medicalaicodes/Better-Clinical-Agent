# MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.09909

# Problem Statement
Comparing different multi-agent architectures (Debate vs. Hierarchical vs. Swarm) is currently impossible because every paper uses a custom codebase and different prompting structures.

# Key Contributions
1. Releases MedMASLab, an open-source framework that standardizes the creation and benchmarking of medical MAS.
2. Allows researchers to swap out the "Topology" (e.g., switch from Star to Ring topology) with a single line of code while keeping the underlying LLMs constant.
3. Provides a unified dashboard tracking token usage, latency, and diagnostic accuracy across 10 different agent topologies.

# Methodology / Architecture
Benchmarking and Orchestration Framework. Includes wrappers for all major LLMs/VLMs and provides pre-built agent topologies (Sequential, Debate, Hierarchical, Decentralized Blackboard). Evaluates them on standard multimodal datasets (MedVQA, MIMIC).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedMASLab Suite | Aggregation of 5 medical datasets| N/A |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Framework Utility| N/A | Standardizes topology eval | N/A | First unified framework |

# Limitations & Future Work
The framework's memory footprint is very high when simulating 10+ agents simultaneously, requiring massive GPU clusters.

# Relevance to Our Survey
Crucial for **Benchmarking Frameworks**. This is the much-needed infrastructure that will allow the field to rigorously evaluate *which* multi-agent architecture is actually best. Modality: Multimodal.
