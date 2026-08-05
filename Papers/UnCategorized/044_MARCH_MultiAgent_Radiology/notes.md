# MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation
**Authors:** N/A
**Year:** 2026 | **Venue:** ACL
**ArXiv/DOI:** https://arxiv.org/abs/2604.16175

# Problem Statement
Automated CT report generation using single models often misses incidental findings or hallucinates anatomical descriptions due to the massive visual complexity of full-body CT scans.

# Key Contributions
1. Introduces MARCH, a hierarchical multi-agent framework mirroring a real radiology department.
2. Divides the CT scan among specialized "Junior Agents" (e.g., Pulmonary Agent, Hepatic Agent).
3. A "Senior Attending Agent" synthesizes the localized findings into a comprehensive, conflict-free report.

# Methodology / Architecture
The framework segments the 3D CT volume into distinct anatomical regions. Each region is processed by an anatomical-specific VLM (Junior Agent) which drafts a "findings" section. The Senior Agent reads these drafts, checks for global consistency (e.g., metastasis spreading), and writes the final "Impression".

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| CT-ReportGen | Chest/Abdomen CT scans | 15k paired scan/reports |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single VLM) | Improvement |
|----------------|--------|------------|----------------------------|-------------|
| Report Accuracy | RadGraph F1 | 68.4% | 59.2% | +9.2% |
| Incidental Recall| Sensitivity | 85.1% | 61.4% | +23.7% |

# Limitations & Future Work
The communication overhead between Junior and Senior agents significantly increases latency, making it unsuitable for rapid trauma bay triage.

# Relevance to Our Survey
Highly relevant for **Medical Vision**. Validates that hierarchical, multi-agent architectures (mirroring human hospital hierarchies) solve the gap of comprehensive incidental finding detection. Modality: 3D CT, Text.
