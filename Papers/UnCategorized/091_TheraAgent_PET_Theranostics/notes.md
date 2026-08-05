# TheraAgent: Multi-Agent Framework with Self-Evolving Memory and Evidence-Calibrated Reasoning for PET Theranostics
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.13676

# Problem Statement
Theranostics (using radioactive drugs to both diagnose and treat cancer) requires highly personalized dosing. Standard VLMs struggle to integrate 3D PET scan uptake values with complex longitudinal patient histories.

# Key Contributions
1. Introduces TheraAgent, a multi-agent framework specifically for precision oncology and radioligand therapy.
2. Uses "Self-Evolving Memory" where the agent tracks its past dosing recommendations and their actual clinical outcomes, continuously updating its internal heuristic weights.
3. Successfully predicts the optimal radiotracer dose based on multimodal evidence (PET scans + EHR).

# Methodology / Architecture
Includes an Imaging Agent (processes SUVmax from PET), a Clinical Agent (processes liver/kidney function from EHR), and a Dosimetry Agent. They vote on the final radioactive dose. A post-treatment feedback loop updates the Agent's Episodic Memory.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| PSMA-Thera-Sim | Simulated prostate cancer therapy | 1,200 simulated cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Baseline (Standard Formula) | Improvement |
|----------------|--------|------------|-----------------------------|-------------|
| Dose Accuracy  | MAPE (Mean Abs Pct Error) | 8.5% | 15.2% | -6.7% error reduction |

# Limitations & Future Work
Currently relies on simulated post-treatment feedback. Real-world validation is pending FDA approval due to the high risks of radiation dosing.

# Relevance to Our Survey
Relevant for **Clinical Decision Support** and **Medical Vision**. Highlights the importance of "Self-Evolving Memory" in treatments where trial-and-error feedback is available over time. Modality: 3D PET, Clinical Text.
