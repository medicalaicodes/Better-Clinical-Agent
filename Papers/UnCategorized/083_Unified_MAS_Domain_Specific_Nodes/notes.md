# Unified-MAS: Universally Generating Domain-Specific Nodes for Empowering Automatic Multi-Agent Systems
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.21475

# Problem Statement
Building multi-agent systems for niche medical domains currently requires manual prompt engineering to define the persona and rules for each specific agent (e.g., "You are an expert pediatric oncologist...").

# Key Contributions
1. Proposes a meta-agent framework that *automatically generates* the optimal multi-agent architecture and agent personas based on a simple task description.
2. If given a task like "Diagnose this pediatric rare disease", Unified-MAS autonomously creates a Geneticist, a Pediatrician, and a Pharmacologist agent, writes their system prompts, and orchestrates them.
3. Eliminates the need for manual prompt engineering when deploying MAS to new medical specialties.

# Methodology / Architecture
Uses a "Meta-Architect Agent" that searches PubMed/Wiki to understand a given disease domain. It then generates JSON configurations defining the optimal number of agents, their tools, and their interaction topology (e.g., debate vs. sequential).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedQA | Standard diagnostic cases | 12k cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Auto-MAS) | Manual MAS | Improvement |
|----------------|--------|-----------------------|------------|-------------|
| Agent Config   | Diagnostic Acc | 84.1% | 83.8% | Comparable, but 100% automated |

# Limitations & Future Work
Occasionally spawns too many agents (e.g., 15 agents for a simple cold), wasting massive amounts of tokens. Needs a cost-constraint module.

# Relevance to Our Survey
Relevant for **Methodology & Taxonomy**. Shows the future of agentic AI is *auto-generating* agents rather than manually hard-coding their personas. Modality: Clinical Text.
