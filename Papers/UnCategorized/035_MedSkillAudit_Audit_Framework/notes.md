# MedSkillAudit: A Domain-Specific Audit Framework for Medical Research Agent Skills
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.20441

# Problem Statement
Medical research agents increasingly rely on third-party "skills" (APIs, Python scripts, databases). There is no standardized framework to audit whether these skills are safe, up-to-date, or prone to hallucinating data.

# Key Contributions
1. Introduces MedSkillAudit, the first comprehensive taxonomy and automated testing framework for medical agent tools.
2. Identifies that 22% of open-source bioinformatics tools hallucinate metadata when queried by an LLM with edge-case inputs.
3. Provides a continuous integration (CI) test suite for evaluating agent tool-calling robustness.

# Methodology / Architecture
The framework uses a "Red Team Agent" that purposefully generates edge-case, malformed, or contradictory inputs into the target agent's tools. The target agent's ability to gracefully catch the error vs. hallucinate a response is audited.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Tool-Robust-Med | Edge-case API parameters | 5k tool calls |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Tool Robustness| Error Catch Rate | 94.2% (w/ Audit) | 61.5% (Baseline) | +32.7% |

# Limitations & Future Work
Currently focuses heavily on bioinformatics and genomic APIs; needs expansion to clinical EHR tools and imaging viewers.

# Relevance to Our Survey
Extremely relevant for **Taxonomy & Methodology**. Directly addresses the gap of auditing and benchmarking the *tools* an agent uses, rather than just the LLM's intrinsic knowledge. Modality: API / Tool-Calling.
