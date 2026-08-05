# Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.21494

# Problem Statement
Tracking brain tumor progression across dozens of longitudinal MRI scans using the BT-RADS scoring system is highly subjective and time-consuming for radiologists.

# Key Contributions
1. First fully agentic pipeline to automate BT-RADS scoring for glioma follow-up.
2. The agent automatically retrieves the historical baseline MRI, co-registers it with the current MRI, segments the tumor, and computes volumetric changes.
3. Generates a drafted radiology report with the formal BT-RADS score and evidence links.

# Methodology / Architecture
A "Pipeline Agent" orchestrates several sub-modules: Image Registration Tool -> V-Net Segmentation Tool -> Volumetric Comparison Tool. A final "Reporting Agent" synthesizes the tool outputs into the standard BT-RADS reporting template.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| BraTS-Longitudinal| Simulated longitudinal MRI | 500 patient timelines|

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Expert Radiologists | Improvement |
|----------------|--------|------------|---------------------|-------------|
| BT-RADS Accuracy| Kappa Agreement| 0.89 | 0.85 (Inter-rater) | Exceeds human inter-rater agreement |

# Limitations & Future Work
Fails if the patient underwent surgical resection between scans, as the sudden absence of tumor tissue confuses the volumetric comparison tool.

# Relevance to Our Survey
Relevant for **Medical Vision**. A perfect example of an agent using external software tools (registration/segmentation) to automate a highly standardized clinical workflow. Modality: 3D MRI.
