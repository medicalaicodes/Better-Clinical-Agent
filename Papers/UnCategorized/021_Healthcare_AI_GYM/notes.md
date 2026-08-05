# Healthcare AI GYM for Medical Agents
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.02943

# Problem Statement
Developing robust medical agents requires safe, reproducible simulation environments. There is a lack of standardized reinforcement learning environments (like OpenAI Gym) specifically tailored for healthcare workflows.

# Key Contributions
1. Introduces Healthcare AI GYM, a suite of simulated environments for training and evaluating medical agents.
2. Includes modular environments for EHR navigation, patient triage, and diagnostic multi-turn dialogue.
3. Provides standardized reward functions balancing clinical accuracy and diagnostic efficiency.

# Methodology / Architecture
Built on standard RL APIs (e.g., Gymnasium). State space includes patient vitals, history, and current symptoms. Action space includes ordering tests, asking questions, and prescribing. The environment simulates patient state transitions using epidemiological and pharmacokinetic models.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Healthcare AI GYM | Simulation suite | 10+ distinct clinical scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| RL Agent Training| Reward Convergence | Stable after 100k steps | N/A | First standardized environment |

# Limitations & Future Work
Simulated patients lack the unpredictable psychological behaviors of real patients (e.g., non-compliance, lying).

# Relevance to Our Survey
Foundational for **Benchmarking Theory & Frameworks**. Standardized simulated environments are necessary for evaluating RL-based agents. Modality: EHR, Text.
