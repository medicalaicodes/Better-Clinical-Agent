# Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.29709

# Problem Statement
Automated medical coding (assigning ICD-10/CPT codes to clinical notes) usually relies on black-box sequence-to-sequence models. When claims are denied by insurance, hospitals cannot explain *why* the AI chose a specific code.

# Key Contributions
1. Introduces Symphony, an agentic system that generates explainable audit trails for every code assigned.
2. Employs a "Code Proposer Agent" and a "Code Auditor Agent".
3. Maps every assigned ICD-10 code directly to exact sentence spans in the EHR using citation markers.

# Methodology / Architecture
The Proposer Agent reads the discharge summary and outputs a JSON of codes. The Auditor Agent takes these codes, searches the ICD-10 manual (via RAG), and verifies that the patient's documented symptoms meet the exact billing criteria. It outputs a human-readable justification report.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MIMIC-III-50 | Standard medical coding dataset | 50k summaries |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Seq2Seq) | Improvement |
|----------------|--------|------------|-------------------------|-------------|
| Coding Accuracy| Micro-F1 | 0.741 | 0.725 | +0.016 |
| Explainability | Human Audit Time | 1.2 mins/chart | 5.5 mins/chart | 4.5x faster audits |

# Limitations & Future Work
Struggles with highly fragmented, multi-encounter charts where symptoms for a code are spread across weeks of notes.

# Relevance to Our Survey
Crucial for **EHR & Operational Automation**. Solves the "Explainability Gap" in healthcare administration. Insurance billing requires verifiable proof, which standard LLMs cannot provide without agentic auditing. Modality: Clinical Text.
