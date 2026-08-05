# Neuro-Symbolic Resolution of Recommendation Conflicts in Multimorbidity Clinical Guidelines
**Authors:** N/A
**Year:** 2026 | **Venue:** AAAI Bridge
**ArXiv/DOI:** https://arxiv.org/abs/2604.17340

# Problem Statement
Standard RAG agents fail when summarizing guidelines for patients with multimorbidity, because Guideline A (for disease 1) often directly contradicts Guideline B (for disease 2) regarding medications or therapies. Pure LLMs cannot resolve these contradictions safely.

# Key Contributions
1. Proposes a neuro-symbolic agent architecture that extracts guideline recommendations into formal logic representations.
2. A symbolic solver (SMT solver) detects direct contraindications (e.g., prescribe Drug X AND do not prescribe Drug X).
3. The LLM then generates a unified, safe care plan that avoids the computationally proven conflict.

# Methodology / Architecture
The LLM acts as the parser, converting clinical guideline text into logical constraints (First-Order Logic). Z3 (an SMT solver) checks for satisfiability across all patient diseases. When unsatisfiable, the solver returns the conflicting axioms. The LLM then queries an "Exceptions" tool to find a secondary treatment pathway that satisfies all constraints.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MultiMorbid-Conflicts | Contradictory clinical guidelines | 1,200 multimorbid scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (GPT-4 RAG) | Improvement |
|----------------|--------|------------|---------------------------|-------------|
| Conflict Resolution| Safe Plan Generation % | 94.2% | 41.5% | +52.7% |

# Limitations & Future Work
Parsing noisy, unstructured clinical text into perfect First-Order Logic is prone to initial extraction errors.

# Relevance to Our Survey
Extremely relevant for **Clinical Decision Support**. Highlights the transition from pure RAG to neuro-symbolic constraint satisfaction for medical safety. Modality: Clinical Text.
