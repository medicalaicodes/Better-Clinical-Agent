# ABRA: Agent Benchmark for Radiology Applications
**Authors:** T. Johnson et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.11224

# Problem Statement
Radiology AI is mostly evaluated on single-image classification. Real-world radiology requires an agent to review prior scans, read the clinical indication, measure lesions, and synthesize a formal reporta complex multi-step workflow.

# Key Contributions
1. Develops ABRA, a comprehensive benchmark for evaluating autonomous radiology agents.
2. Includes tasks for visual grounding, prior-comparison, and tool use (e.g., calling a ruler API).
3. Reveals that SOTA VLMs fail at basic spatial reasoning when used without agentic tool-calling.

# Methodology / Architecture
ABRA evaluates agents in a simulated PACS environment. Agents can zoom, pan, and call segmentation tools. They must output a structured radiology report.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ABRA Dataset | Curated chest X-rays and MRIs with multi-step annotations | 2k cases |
| MIMIC-CXR | Chest X-rays | subset |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent) | VLM Baseline | Improvement |
|----------------|--------|------------|---------------|-------------|
| Report Generation| RadGraph F1 | 64.2% | 48.7% | +15.5% |
| Lesion Tracking | IoU | 0.72 | 0.51 | +0.21 |

# Limitations & Future Work
Evaluation is computationally expensive. Future work aims to optimize the environment for faster benchmarking.

# Relevance to Our Survey
Highly relevant for **Medical Vision**. Identifies the gap between static image classification and interactive radiological workflows. Modality: Imaging, Clinical Text.
