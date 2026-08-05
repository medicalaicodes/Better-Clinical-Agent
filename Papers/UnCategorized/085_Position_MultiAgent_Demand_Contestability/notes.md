# Position: Multi-Agent Algorithmic Care Systems Demand Contestability for Trustworthy AI
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.20595

# Problem Statement
As multi-agent systems become more complex and autonomous, they often produce decisions that human clinicians cannot easily challenge or override (the "Black Box" problem on a system level).

# Key Contributions
1. Argues that Agentic Healthcare systems must be designed for "Contestability"—meaning a clinician must be able to argue with, interrogate, and easily override the agent's logic.
2. If a multi-agent board recommends surgery, there must be a UI mechanism for the doctor to inject a counter-premise (e.g., "Patient is frail") and force the agents to re-run their debate.
3. Proposes an evaluation framework for measuring how well an agent accepts user correction.

# Methodology / Architecture
A theoretical position paper. Outlines a "Contestability Protocol" where agents are required to expose their internal reasoning trace (the blackboard/scratchpad) to the human user, and accept natural language "override commands" that forcefully alter the agent's context window.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| N/A | Position/Policy Paper | N/A |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| N/A | N/A | N/A | N/A | N/A |

# Limitations & Future Work
Implementing contestability may drastically slow down clinical workflows if physicians feel compelled to argue with the agent on every case.

# Relevance to Our Survey
Crucial for **Methodology & Taxonomy** and **Safety/Ethics**. Defines the regulatory and safety requirements for deploying autonomous agents in the real world. Modality: N/A.
