# MedPriv-Bench: Benchmarking the Privacy-Utility Trade-off of Large Language Models in Medical Open-End Question Answering
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.14265

# Problem Statement
When agents retrieve data from patient EHRs to answer questions, they often leak Protected Health Information (PHI) in their reasoning trace or final output, violating HIPAA.

# Key Contributions
1. Introduces MedPriv-Bench, specifically measuring how much PHI agents leak when solving complex clinical cases.
2. Tests various anonymization strategies (Local Scrubbing Agents vs. Prompt-based rules).
3. Demonstrates the "Privacy-Utility Trade-off": Over-scrubbing patient data (e.g., removing exact age and race) significantly degrades the agent's diagnostic accuracy.

# Methodology / Architecture
Evaluation framework. An agent is given an unredacted EHR note and asked a clinical question. The output is parsed by a Named Entity Recognition (NER) model to detect leaked PHI. The diagnostic accuracy is also measured to calculate the trade-off.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedPriv-Bench | EHR notes with synthetic PHI | 4,000 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | Scrubbed Input | Unredacted Input | Trade-off |
|----------------|--------|----------------|------------------|-----------|
| Diagnostic Acc | Top-1 Accuracy | 71.2% | 85.5% | -14.3% penalty |
| Privacy Leakage| PHI Entities Leaked | 0.01/case | 3.5/case | Massive privacy gain |

# Limitations & Future Work
The synthetic PHI doesn't always mimic the complex, hidden ways real doctors embed identifying details in unstructured notes.

# Relevance to Our Survey
Relevant for **Benchmarking Frameworks** and **Safety/Ethics**. Quantifies the exact diagnostic penalty of enforcing privacy constraints on clinical agents. Modality: Clinical Text.
