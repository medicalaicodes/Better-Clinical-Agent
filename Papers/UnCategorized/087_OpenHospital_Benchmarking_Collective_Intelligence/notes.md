# OpenHospital: A Thing-in-itself Arena for Evolving and Benchmarking LLM-based Collective Intelligence
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.14771

# Problem Statement
We lack virtual environments where hundreds of agents can interact simultaneously to simulate hospital-wide dynamics (triage queues, bed management, inter-departmental consults).

# Key Contributions
1. Releases OpenHospital, a massive multiplayer virtual environment mimicking a real hospital.
2. Agents take on roles (Doctor, Nurse, Patient, Admin) and must coordinate to treat incoming "patient agents" while managing limited virtual resources.
3. Used to benchmark "Collective Intelligence"—how well agents cooperate rather than just how well they diagnose in isolation.

# Methodology / Architecture
Simulation Environment. Built on an LLM-driven game engine. Agents communicate via natural language. The environment tracks global metrics like average patient wait time, mortality rate, and budget usage.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| OpenHospital Arena| Multi-agent simulation | 500 simultaneous agents|

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent Staffed) | Baseline (Random Queue) | Improvement |
|----------------|--------|----------------------------|-------------------------|-------------|
| Resource Management| Patient Survival Rate| 82.1% | 41.5% | +40.6% |

# Limitations & Future Work
The environment does not simulate the physical geometry of a hospital, only the logical flow of resources and communication.

# Relevance to Our Survey
Relevant for **Benchmarking Frameworks** and **EHR & Administrative**. A highly novel approach to testing Agent swarms at the hospital operational level. Modality: Text (Simulation).
