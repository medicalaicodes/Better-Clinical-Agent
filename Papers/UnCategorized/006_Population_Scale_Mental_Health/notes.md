# An Agentic LLM-Based Framework for Population-Scale Mental Health Screening
**Authors:** S. Kim et al.
**Year:** 2025 | **Venue:** IEEE BigData
**ArXiv/DOI:** https://arxiv.org/abs/2605.13046

# Problem Statement
Traditional mental health screening relies on static questionnaires (PHQ-9, GAD-7) which have low completion rates and fail to capture conversational nuance. Scaling psychiatric evaluation to a population level requires agents capable of empathetic, dynamic interviewing.

# Key Contributions
1. Presents an agentic framework for conducting dynamic mental health assessments.
2. Integrates an "Empathy Modulator" and a "Clinical Evaluator" agent working in tandem.
3. Achieves high concordance with human psychiatrist evaluations on depression severity.

# Methodology / Architecture
The framework uses a dual-agent system. The Interviewer Agent interacts with the patient, dynamically adapting questions based on sentiment. The Evaluator Agent monitors the conversation passively, scoring symptoms against DSM-5 criteria. If self-harm risk is detected, a deterministic override triggers emergency protocols.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Multi-WOZ Mental | Mental health dialogue corpus | 5k dialogues |
| PsyQA | Chinese psychiatric QA dataset | 22k pairs |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| PHQ-9 Concordance| Pearson r | 0.82 | 0.71 | +0.11 |

# Limitations & Future Work
Risk of hallucinating empathy in inappropriate contexts. Needs multi-modal integration (voice tone, facial expressions).

# Relevance to Our Survey
Highly relevant for **Clinical Decision Support (Mental Health)**. Demonstrates how agentic workflows replace static forms with dynamic, safe interaction. Modality: Clinical Text/Dialogue.
