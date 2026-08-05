# LungCURE: Benchmarking Multimodal Real-World Clinical Reasoning for Precision Lung Cancer Diagnosis and Treatment
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.06925

# Problem Statement
Existing oncology benchmarks are unimodal (text only) or test isolated tasks (e.g., only identifying a nodule on a CT). Real-world precision oncology requires synthesizing genomics (EGFR mutations), imaging (CT scans), and clinical history to recommend a targeted therapy.

# Key Contributions
1. Releases LungCURE, the first end-to-end multimodal benchmark for precision lung cancer.
2. Requires agents to ingest a CT scan, a genomic sequencing report, and EHR notes simultaneously to output a final TNM stage and specific targeted therapy recommendation.
3. Exposes that while agents excel at the text-genomics portion, they fail to integrate visual tumor size correctly with the genomic data.

# Methodology / Architecture
The benchmark provides a unified prompt containing the patient's multimodal data. It evaluates agents across 3 axes: Diagnostic (Staging), Prognostic (Survival estimation), and Therapeutic (Drug selection based on mutations and contraindications).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| LungCURE | Multimodal NSCLC patient cases | 850 complete patient profiles |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (GPT-4V Baseline)| Expert Oncologist | Improvement |
|----------------|--------|-----------------------------|-------------------|-------------|
| Therapeutic Choice| Treatment F1 | 58.2% | 94.5% | Agent fails at multimodal synthesis|

# Limitations & Future Work
Currently limited to Non-Small Cell Lung Cancer (NSCLC).

# Relevance to Our Survey
Crucial for **Benchmarking Frameworks**. Highlights the massive gap in *multimodal* evidence integration for precision medicine, setting the standard for future agent evaluation. Modality: 3D CT, Genomic Text, EHR Text.
