# SymptomAI: Toward a Conversational AI Agent for Everyday Symptom Assessment
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.04012

# Problem Statement
Current symptom checkers (e.g., Ada, WebMD) use rigid decision trees. LLM alternatives hallucinate or jump to conclusions too quickly without asking enough clarifying questions.

# Key Contributions
1. SymptomAI, a conversational agent fine-tuned for empathetic and thorough symptom gathering.
2. Introduces a "Triage-First" reward mechanism via RLHF to prioritize urgent conditions (e.g., heart attack vs. indigestion).

# Methodology / Architecture
The agent is trained using DPO (Direct Preference Optimization) on a dataset of doctor-patient triage conversations. The preference model heavily penalizes missed "red flag" symptoms.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| TriageQA | Simulated symptom assessments | 10k interactions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Triage Accuracy | Triage F1 | 88.5% | 81.2% | +7.3% |
| Red Flag Detection | Sensitivity | 99.1% | 94.5% | +4.6% |

# Limitations & Future Work
Currently text-only. Integration with voice assistants (Alexa, Siri) is the next step.

# Relevance to Our Survey
Relevant for **Patient-Facing Copilots (Triage)**. Addresses the gap in conversational symptom gathering. Modality: Text.
