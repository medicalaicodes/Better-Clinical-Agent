# Thinking Like a Clinician: A Cognitive AI Agent for Clinical Diagnosis via Panoramic Profiling and Adversarial Debate
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.23605

# Problem Statement
Single-agent diagnostic models often suffer from confirmation bias, latching onto early symptoms and ignoring contradictory later evidence. 

# Key Contributions
1. Introduces a Multi-Agent Adversarial Debate framework (Proponent vs. Skeptic agents).
2. The "Skeptic" agent is actively rewarded for finding evidence in the EHR that refutes the Proponent's leading differential diagnosis.
3. Significantly improves diagnostic accuracy on complex, multi-morbid patient cases.

# Methodology / Architecture
The framework uses a "Panoramic Profiling" module to extract all symptoms, medications, and labs. The Proponent agent proposes a diagnosis. The Skeptic agent actively searches the profile for contradictory evidence (e.g., "The patient's elevated Troponin contradicts your diagnosis of costochondritis"). A "Chief Resident" agent synthesizes the debate.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MultiMorbid-DX | Complex multi-morbid clinical cases | 2,500 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single Agent) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| Diagnostic Acc | Top-1 Accuracy | 81.4% | 68.2% | +13.2% |

# Limitations & Future Work
The adversarial debate process requires 3x the API calls (and inference time) compared to a standard ReAct agent, making it expensive for low-acuity triage.

# Relevance to Our Survey
Highly relevant for **Clinical Decision Support**. Addresses the gap of single-agent confirmation bias and introduces multi-agent adversarial debate as the SOTA methodology for complex diagnostics. Modality: Clinical Text.
