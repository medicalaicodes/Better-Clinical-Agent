# RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.15231

# Problem Statement
End-to-end Vision-Language Models (VLMs) for Chest CTs often struggle with fine-grained anatomical localization and quantitative measurements (e.g., measuring the exact diameter of a nodule), leading to vague radiology reports.

# Key Contributions
1. Proposes RadAgent, an LLM orchestrator that interacts with a suite of specialized diagnostic tools (e.g., 3D lung segmenter, nodule detector, measuring calipers).
2. Introduces a "stepwise reasoning" framework (Observe -> Measure -> Synthesize) rather than single-pass generation.
3. Radically improves the quantitative accuracy and anatomical grounding of generated radiology reports.

# Methodology / Architecture
RadAgent receives the raw CT scan. It first calls a `LungSegmentation` tool. Then it queries a `NoduleDetection` API. It takes the bounding boxes from that API and calls a `MeasureDiameter` tool. Finally, the LLM takes this structured data (e.g., "7mm nodule in right upper lobe") and drafts the formal text report.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| LUNA16 & Custom CT| Nodule detection and reporting | 888 scans |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (End-to-end VLM)| Improvement |
|----------------|--------|------------|--------------------------------|-------------|
| Quantitative Acc | Diameter Error Margin| ±0.5mm | ±2.1mm | 4x more precise |
| Report Quality | RadGraph F1 | 75.2% | 66.8% | +8.4% |

# Limitations & Future Work
The latency of running multiple cascaded tools sequentially makes the system slower than a single-pass VLM (approx 45 seconds per scan).

# Relevance to Our Survey
Highly relevant for **Medical Vision**. Demonstrates that tool-using agents (combining LLMs with deterministic measuring/segmenting APIs) outperform monolithic VLMs on tasks requiring high spatial precision. Modality: 3D CT, APIs.
