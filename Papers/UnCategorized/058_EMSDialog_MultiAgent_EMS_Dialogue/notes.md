# EMSDialog: Synthetic Multi-person Emergency Medical Service Dialogue Generation from Electronic Patient Care Reports via Multi-LLM Agents
**Authors:** N/A
**Year:** 2026 | **Venue:** ACL Findings
**ArXiv/DOI:** https://arxiv.org/abs/2604.07549

# Problem Statement
Training ambient listening copilots for Emergency Medical Services (EMS) is hindered by the extreme lack of publicly available, privacy-compliant, multi-speaker chaotic trauma dialogue data.

# Key Contributions
1. Introduces EMSDialog, a multi-agent framework designed solely to *generate* ultra-realistic synthetic training data.
2. Agents take on personas: Paramedic, Distressed Patient, Panicked Bystander, and Medical Control Dispatch.
3. The framework generates chaotic, overlapping, and jargon-heavy dialogue grounded in real, de-identified electronic Patient Care Reports (ePCRs).

# Methodology / Architecture
Given an ePCR summary, the "Director Agent" assigns roles to multiple LLM sub-agents. The agents interact in a simulated environment, incorporating interruptions and background noise tags. A "Clinical Reviewer Agent" ensures the synthetic dialogue medically aligns with the original ePCR outcome.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Synthetic-EMS | Generated transcripts | 10k multi-speaker dialogues |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Human Docs) | Improvement |
|----------------|--------|------------|----------------------------|-------------|
| Data Generation| Clinical Accuracy| 96.5% | N/A | High fidelity |
| Copilot Training| ASR/NLU downstream| 88.2% F1 | 74.1% (trained on general data)| +14.1% |

# Limitations & Future Work
Struggles to perfectly mimic regional colloquialisms or extreme panic speech patterns, sometimes sounding too clinically formal.

# Relevance to Our Survey
Relevant for **EHR & Operational Automation** (Ambient Scribes) and **Data Generation**. Proves multi-agent simulation is the SOTA method for bypassing the HIPAA/privacy data bottleneck in chaotic medical settings. Modality: Text (Transcript).
