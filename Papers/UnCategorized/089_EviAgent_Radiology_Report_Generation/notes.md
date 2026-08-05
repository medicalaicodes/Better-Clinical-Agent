# EviAgent: Evidence-Driven Agent for Radiology Report Generation
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.13956

# Problem Statement
Vision-Language Models generate fluent radiology reports, but often hallucinate findings that aren't actually in the image. 

# Key Contributions
1. Proposes EviAgent, which separates the process into "Evidence Gathering" and "Report Writing".
2. The agent uses a suite of narrow, highly accurate visual tools (e.g., a Cardiomegaly Classifier, a Nodule Detector) to extract hard evidence from the X-ray first.
3. The LLM then writes the report *strictly* constrained by the symbolic evidence output by the tools, drastically reducing hallucinations.

# Methodology / Architecture
Tool-Augmented VLM. Step 1: LLM looks at the image and calls specialized APIs (e.g., `detect_effusion(image_id)`). Step 2: The APIs return boolean/bounding box data. Step 3: The LLM writes the report conditioned on this exact data.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MIMIC-CXR | Chest X-Rays and Reports | 100k pairs |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (End-to-End VLM)| Improvement |
|----------------|--------|------------|-------------------------------|-------------|
| Report Generation| F1 (CheXbert) | 0.68 | 0.59 | +0.09 |
| Hallucination  | False Positive Rate| 4.2% | 15.8% | -11.6% (Huge reduction) |

# Limitations & Future Work
The agent is limited by its tool library. If a rare disease appears for which it has no specific detection tool, it tends to ignore it.

# Relevance to Our Survey
Relevant for **Medical Vision**. Validates that Tool-Use (calling narrow expert models) is superior to monolithic End-to-End VLMs for factual report generation. Modality: 2D Imaging, Clinical Text.
