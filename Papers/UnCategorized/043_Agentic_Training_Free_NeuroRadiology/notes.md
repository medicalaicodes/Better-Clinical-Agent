# Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.16729

# Problem Statement
Training domain-specific Vision-Language Models (VLMs) for every highly specialized subset of radiology (e.g., neuro-radiology vs. MSK) is computationally prohibitive and limited by data scarcity.

# Key Contributions
1. Proposes a "Training-Free" agentic framework that bypasses the need for specialized VLMs.
2. The central LLM agent orchestrates pre-existing, open-source narrow AI tools (e.g., Brain Tumor Segmentation models, stroke detection APIs).
3. Achieves expert-level neuro-radiological reports without any fine-tuning of the core LLM.

# Methodology / Architecture
Given a neuro-MRI, the agent dynamically decides which off-the-shelf CNNs/tools to invoke based on the clinical indication. If it sees a mass, it calls a segmentation tool. It then synthesizes the numerical outputs of these narrow tools into a cohesive, structured radiology report.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Neuro-RadBench | Neuro-MRI analysis tasks | 3,000 scans |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Specialized VLM)| Improvement |
|----------------|--------|------------|--------------------------------|-------------|
| Report Generation | Clinical F1 | 79.8% | 76.5% | +3.3% |

# Limitations & Future Work
Heavily dependent on the availability and accuracy of the narrow, underlying CNN tools. Cannot detect findings for which no tool exists.

# Relevance to Our Survey
Crucial for **Medical Vision**. Identifies a paradigm shift: using agents as orchestrators of existing narrow AI tools rather than trying to build monolithic medical VLMs. Modality: 3D MRI, Text.
