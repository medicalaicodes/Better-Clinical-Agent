# MediHive: A Decentralized Agent Collective for Medical Reasoning
**Authors:** N/A
**Year:** 2026 | **Venue:** IEEE ICHI
**ArXiv/DOI:** https://arxiv.org/abs/2603.27150

# Problem Statement
Most multi-agent systems use a rigid, hierarchical structure (e.g., one Senior agent managing three Junior agents). If the Senior agent hallucinates, the entire system fails.

# Key Contributions
1. Introduces MediHive, a *decentralized* swarm of specialized agents (Cardiologist, Pulmonologist, Pharmacist).
2. Agents communicate via a shared "blackboard" memory space rather than a strict chain-of-command.
3. The collective reaches a diagnosis via a weighted voting mechanism, preventing single-point-of-failure hallucinations.

# Methodology / Architecture
Uses Blackboard Architecture. When a complex ICU case is presented, the relevant specialist agents activate and write their findings to the shared blackboard. Other agents read the blackboard and append their own critiques. The final diagnosis is determined by aggregating the confidence scores of all active agents.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MIMIC-IV-ICU | Intensive care case summaries | 1,500 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Hierarchical) | Improvement |
|----------------|--------|------------|------------------------------|-------------|
| ICU Diagnosis  | F1 Score | 83.9% | 79.4% | +4.5% |

# Limitations & Future Work
The lack of a central controller means agents occasionally talk past each other, getting stuck in loops arguing over minor physiological details.

# Relevance to Our Survey
Relevant for **Clinical Decision Support**. Provides an alternative (Decentralized Swarm) to the standard Hierarchical multi-agent frameworks. Modality: Clinical Text.
