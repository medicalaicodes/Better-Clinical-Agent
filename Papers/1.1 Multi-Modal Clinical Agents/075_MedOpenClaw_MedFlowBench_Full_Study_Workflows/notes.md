# MedOpenClaw and MedFlowBench: Auditing Medical Agents in Full-Study Workflows
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.24649

# Problem Statement
Most medical vision benchmarks test agents on pre-cropped 2D images. Real radiologists must navigate full 3D studies (hundreds of slices), operate complex viewing software (e.g., 3D Slicer), and produce auditable visual evidence for their findings.

# Key Contributions
1. Releases MedOpenClaw, a runtime environment where Computer-Use Agents can natively operate clinical imaging software (3D Slicer, QuPath).
2. Introduces MedFlowBench, evaluating agents on full-study workflows. Agents must scroll through slices, adjust contrast windows, and drop 3D coordinates on lesions.
3. Proves that VLM agents that score 90%+ on static 2D benchmarks completely fail when forced to navigate a full 3D study GUI.

# Methodology / Architecture
The Agent receives a full 3D CT scan. It uses mouse/keyboard tool APIs to interact with the 3D Slicer GUI (changing slices, adjusting Hounsfield units). It must eventually output the exact 3D (x,y,z) coordinate of the tumor. The benchmark checks this coordinate against ground truth masks.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedFlowBench | Full 3D CT/MRI studies | 1,000 complete studies |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent in MedOpenClaw) | Static 2D VLM | Improvement |
|----------------|--------|-----------------------------------|---------------|-------------|
| Lesion Localization| 3D Coordinate Precision | 42.1% (Low) | N/A (Fails entirely)| Establishes baseline |

# Limitations & Future Work
The agent frequently gets "lost" in the software interface (e.g., clicking the wrong menu and failing to recover), limiting its medical evaluation score.

# Relevance to Our Survey
Absolutely critical for **Medical Vision** and **Benchmarking Frameworks**. This bridges the gap between static image analysis and real-world Computer-Use Agents in radiology. Modality: 3D Imaging, GUI / Visual-Action.
