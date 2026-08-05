# Cerebra: A Multidisciplinary AI Board for Multimodal Dementia Characterization and Risk Assessment
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.21597

# Problem Statement
Diagnosing dementia requires synthesizing cognitive tests, PET/MRI scans, and genomic data (APOE status), which is currently done via slow, costly multidisciplinary tumor/medical boards.

# Key Contributions
1. Introduces "Cerebra", an AI-simulated Multidisciplinary Board (MDT).
2. Specialized agents (a Neurologist Agent reading cognitive tests, a Neuroradiologist Agent reading MRIs, a Geneticist Agent reading APOE alleles) review the multimodal data independently and then debate.
3. Provides a 5-year risk assessment for progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD).

# Methodology / Architecture
Multimodal Multi-Agent Architecture. Uses specialized models for each node (e.g., Med3D for the MRI agent, clinical LLMs for the others). The orchestrator summarizes the debate into a final risk probability score.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ADNI | Multimodal Alzheimer's Data | 3,000 patients |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single Multimodal Model) | Improvement |
|----------------|--------|------------|-----------------------------------------|-------------|
| 5-Year AD Risk | AUROC | 0.92 | 0.85 | +0.07 |

# Limitations & Future Work
The system is rigid; if a patient is missing the MRI modality, the Neuroradiologist agent halts the entire pipeline rather than adapting to missing data.

# Relevance to Our Survey
Relevant for **Clinical Decision Support** and **Medical Vision**. Validates the "Agentic MDT" (Multidisciplinary Team) concept for complex, multimodal chronic diseases. Modality: Text, 3D MRI, Genomics.
