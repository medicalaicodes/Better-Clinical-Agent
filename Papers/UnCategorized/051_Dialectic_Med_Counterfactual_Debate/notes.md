# Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate
**Authors:** N/A
**Year:** 2026 | **Venue:** ACL
**ArXiv/DOI:** https://arxiv.org/abs/2604.11258

# Problem Statement
Standard single-agent diagnostic LLMs suffer from severe hallucinations when presented with noisy or incomplete EHR data, often confidently inventing symptoms to fit their internal differential diagnosis.

# Key Contributions
1. Introduces Dialectic-Med, a framework that forces two agents to debate a diagnosis.
2. The "Counterfactual Agent" explicitly generates "what-if" scenarios (e.g., "What if the patient's fever is just a viral artifact and not related to the rash?") to attack the "Proponent Agent".
3. Significantly reduces diagnostic hallucinations.

# Methodology / Architecture
The architecture consists of a Proponent Agent (generating hypotheses), a Counterfactual Attacker Agent (generating opposing scenarios), and a Judge Agent (synthesizing the final diagnostic report after a set number of debate rounds).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedHallucinate-Eval| EHR cases with forced noise | 3,000 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single Agent)| Improvement |
|----------------|--------|------------|-----------------------------|-------------|
| Diagnostic Safety| Hallucination Rate | 2.1% | 14.3% | Massive drop |

# Limitations & Future Work
The counterfactual generation occasionally distracts the Judge Agent with highly improbable zebra diagnoses (e.g., suggesting Malaria for a patient who has never left Alaska).

# Relevance to Our Survey
Highly relevant for **Clinical Decision Support**. Proves that counterfactual adversarial debate is a SOTA methodology for mitigating hallucinations. Modality: Clinical Text.
