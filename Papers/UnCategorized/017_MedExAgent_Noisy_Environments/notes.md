# MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.07058

# Problem Statement
Most diagnostic agents are tested on clean, idealized datasets (like MedQA). In the real world, clinical environments are noisy, containing contradictory information, typos, and missing data.

# Key Contributions
1. Introduces MedExAgent, trained explicitly to handle incomplete and contradictory patient histories.
2. Proposes a multi-step "Examine" tool where the agent can actively query for missing lab results.
3. Develops the "NoisyClinic" benchmark.

# Methodology / Architecture
MedExAgent uses a ReAct formulation with specialized tools: Query_Lab, Ask_Patient, Consult_Guideline. During training, synthetic noise (e.g., swapping lab values, truncating histories) is injected.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| NoisyClinic | MIMIC-IV with synthetic noise | 15k trajectories |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| NoisyClinic | Diagnostic Acc | 71.4% | 55.2% | +16.2% |

# Limitations & Future Work
Training is computationally heavy. Needs validation on real, unstructured clinical notes rather than synthetically noised data.

# Relevance to Our Survey
Highlights the gap in **Real-world Robustness**. Modality: EHR Tabular, Text.
