# DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.15456

# Problem Statement
Conducting systematic reviews and meta-analyses in medicine takes months. Current LLMs can summarize single papers but fail at executing the multi-step, rigorous PRISMA workflow required for evidence-based medicine.

# Key Contributions
1. Introduces DeepER-Med, an autonomous agentic pipeline that performs end-to-end systematic reviews.
2. Implements specific tools for PubMed querying, abstract screening, bias assessment (RoB 2), and data extraction.
3. Radically accelerates the generation of clinical evidence summaries.

# Methodology / Architecture
The framework operates as a state machine mimicking the PRISMA guidelines. The "Search Agent" builds boolean queries. The "Screening Agent" filters by inclusion/exclusion criteria. The "Extraction Agent" pulls tabular data (P-values, hazard ratios). Finally, the "Meta-Analysis Agent" runs statistical Python scripts to generate a forest plot and summary report.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| SysReview-Bench | Gold-standard systematic reviews | 150 full reviews |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Human) | Improvement |
|----------------|--------|------------|-----------------------|-------------|
| Paper Screening | Recall / Precision| 98.5% / 82.1% | 99% / 85% | Comparable |
| Time to Review | Total Duration | 3 Hours | 4 Months | Massive speedup |

# Limitations & Future Work
Struggles to extract data from highly complex or poorly formatted tables in older PDFs. Requires human-in-the-loop validation before clinical policy changes.

# Relevance to Our Survey
Crucial for **Research & Systematic Review Agents**. Addresses the massive operational gap in Evidence-Based Medicine (EBM) generation. Modality: Scientific Literature (PDFs), Code/Stats.
