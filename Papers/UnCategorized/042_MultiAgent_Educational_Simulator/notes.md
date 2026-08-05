# Persona-Based Requirements Engineering for Explainable Multi-Agent Educational Systems: A Scenario Simulator for Clinical Reasoning Training
**Authors:** N/A
**Year:** 2026 | **Venue:** CSTE
**ArXiv/DOI:** https://arxiv.org/abs/2604.17186

# Problem Statement
Medical students lack scalable, realistic practice environments to hone their clinical reasoning. Static case studies do not react to the student's probing questions, and standard LLM chatbots act as "answer engines" rather than authentic simulated patients.

# Key Contributions
1. Introduces a Multi-Agent Educational Simulator tailored for clinical reasoning.
2. Utilizes persona-based engineering: one agent acts as the uncooperative/anxious patient, another as the senior attending physician evaluating the student.
3. Provides explainable feedback on the student's diagnostic hypotheses.

# Methodology / Architecture
The system employs three interacting agents. The "Patient Agent" has a hidden state (true diagnosis and vitals) and communicates via dialogue. The "Student" (human) asks questions. The "Evaluator Agent" silently tracks the student's dialogue against clinical guidelines and issues a debriefing report after the interaction, grading their differential diagnosis.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ClinEdu-Sim | Clinical simulation transcripts | 500 simulated scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Static cases) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| Educational Efficacy| Diagnostic Accuracy (Post-test)| 82.5% | 71.0% | +11.5% |

# Limitations & Future Work
The Evaluator Agent occasionally hallucinates rigid clinical rules that penalize creative but valid student diagnostic approaches.

# Relevance to Our Survey
Highly relevant for **Medical Education & Simulation Copilots**. Highlights the use of specialized personas in multi-agent environments. Modality: Text (Dialogue).
