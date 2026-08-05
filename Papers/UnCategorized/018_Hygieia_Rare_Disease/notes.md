# Hygieia: A Versatile AI Agent for Rare Disease Diagnosis and Risk Gene Prioritization
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.06226

# Problem Statement
Diagnosing rare diseases takes an average of 5-7 years. Generalist medical LLMs lack deep genomic knowledge and struggle to connect disparate phenotypic symptoms to rare genotypes.

# Key Contributions
1. Hygieia: An agent that orchestrates genomics databases (OMIM, ClinVar) and phenotypic data (HPO).
2. Uses a dynamic tool-calling strategy to prioritize risk genes based on patient symptoms.
3. Achieves SOTA on rare disease diagnosis benchmarks.

# Methodology / Architecture
The agent receives patient symptoms, translates them to Human Phenotype Ontology (HPO) terms via a semantic tool, then queries genomic databases to retrieve candidate genes, finally ranking them using a reasoning module.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| RAMED | Rare disease benchmark | 3k cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Rare Disease Dx | Top-5 Accuracy | 68.9% | 52.3% | +16.6% |

# Limitations & Future Work
Heavily reliant on the completeness of OMIM and ClinVar. Struggles with entirely novel mutations.

# Relevance to Our Survey
Highly relevant for **Genomics & Omics Agents**. Modality: Genomic Data, Text.
