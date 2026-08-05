# Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.25821

# Problem Statement
Current evaluations (e.g., MedQA) only test isolated, single-turn question answering. Real clinical care is an end-to-end, multi-turn, multi-day process involving triage, testing, diagnosis, and follow-up.

# Key Contributions
1. Releases Doctorina MedBench, a sprawling end-to-end evaluation environment.
2. Agents must interact with simulated patients, order tests (which cost virtual money and take virtual time), interpret the returned test results, and manage the patient's condition over several simulated days.
3. Proves that agents with high MedQA scores often fail horribly at longitudinal patient management (ordering redundant tests or forgetting to follow up).

# Methodology / Architecture
The benchmark is a Gym-like environment. The Agent takes actions (Talk, Order Test, Prescribe, Wait). The Environment responds with new symptoms or test results based on a Markov Decision Process (MDP) representing disease progression. 

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Doctorina MedBench| Longitudinal simulated cases | 5,000 patient journeys |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single-turn LLMs) | Improvement |
|----------------|--------|------------|----------------------------------|-------------|
| Patient Management| Survival & Cost Score | 78.4 | 42.1 (GPT-4 Base) | +36.3 |

# Limitations & Future Work
The underlying MDP disease models are simplified and do not perfectly capture complex multi-organ failure dynamics.

# Relevance to Our Survey
Highly relevant for **Benchmarking Frameworks**. Defines the standard for evaluating long-horizon, multi-turn clinical management rather than static Q&A. Modality: Clinical Text (Longitudinal).
