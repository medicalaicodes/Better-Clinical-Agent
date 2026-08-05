# Clinically Interpretable Sepsis Early Warning via LLM-Guided Simulation of Temporal Physiological Dynamics
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.20924

# Problem Statement
Traditional Deep Learning models (LSTMs) for sepsis prediction are highly accurate but act as black boxes. Clinicians ignore their alerts because the models cannot explain *why* the patient is deteriorating.

# Key Contributions
1. Uses an LLM agent to orchestrate and explain a physiological simulation model (ODE-based) of the patient's cardiovascular system.
2. The agent generates a natural language, clinically interpretable rationale for the impending sepsis alert based on the simulation's projected temporal dynamics.
3. Increases clinician trust and adoption of the early warning system.

# Methodology / Architecture
EHR time-series data (vitals, labs) is fed into a mathematical ODE simulation of physiological decay. The LLM agent queries this simulation via a tool, projecting the patient's state 6 hours into the future, and writes a synthesized "Warning Note" for the clinician.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MIMIC-IV Sepsis | ICU patient time-series | 30k patient trajectories |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Sepsis Prediction| AUROC | 0.89 | 0.90 (Blackbox LSTM)| -0.01 (Comparable) |
| Clinical Trust | Clinician Alert Acceptance| 84% | 41% (Blackbox) | +43% (Massive adoption increase) |

# Limitations & Future Work
The ODE simulation tools require heavy calibration per patient and struggle with patients having extreme baseline anomalies (e.g., end-stage renal disease).

# Relevance to Our Survey
Relevant for **Clinical Decision Support (ICU)**. Addresses the interpretability gap, proving that agents can bridge the gap between black-box predictive models and clinician trust via natural language explanations. Modality: Time-Series Tabular, Text.
