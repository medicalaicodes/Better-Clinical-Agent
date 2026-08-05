# An Empirical Study of Agent Skills for Healthcare: Practice, Gaps, and Governance
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.02709

# Problem Statement
While many "skills" (tool-calling capabilities) are developed for healthcare agents, there is no comprehensive taxonomy or governance framework analyzing which skills are most effective and where the critical safety gaps lie.

# Key Contributions
1. Conducts an empirical analysis of over 500 open-source and proprietary medical agent skills.
2. Identifies a severe gap in "Write" skills (e.g., prescribing, modifying EHR) vs. "Read" skills (querying databases).
3. Proposes a governance framework for auditing and verifying medical agent tools.

# Methodology / Architecture
Qualitative and quantitative survey methodology. Categorizes skills into Read, Compute, Write, and Communicate. Evaluates failure modes of these tools in agentic pipelines.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| SkillAudit | Dataset of agent execution traces | 500+ skills analyzed |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Tool Failure Rate| % Failures | 34% (Write) vs 8% (Read) | N/A | Identifies major capability gap |

# Limitations & Future Work
Only analyzes skills available as of early 2026. Governance framework is theoretical and requires regulatory adoption.

# Relevance to Our Survey
Extremely relevant for **Taxonomy and Benchmarking Theory**. Identifies the exact "gaps" (specifically in Write operations) that our survey aims to uncover. Modality: N/A (Meta-analysis).
