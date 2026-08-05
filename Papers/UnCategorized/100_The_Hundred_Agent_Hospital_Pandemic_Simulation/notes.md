# The Hundred-Agent Hospital: A Scalable Simulation for Pandemics and Resource Allocation
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.03195

# Problem Statement
Predicting how a hospital will collapse during a pandemic surge (e.g., running out of ventilators or nursing staff) is difficult with standard mathematical models, as human behavior and panic are non-linear.

# Key Contributions
1. Simulates a 100-agent hospital during an infectious disease outbreak (e.g., COVID-19).
2. Agents have customized psychological profiles (e.g., a "fatigued nurse agent", a "panicked patient agent").
3. Reveals emergent bottlenecks (e.g., panic in the waiting room disrupting triage faster than the actual bed shortage).

# Methodology / Architecture
Generative Agents (similar to the "Smallville" paper) adapted for healthcare. Uses a vector database for agent memories. Each agent operates independently, interacting with physical (virtual) hospital resources and other agents.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Custom Simulation | Pandemics scenario generator| 100 Agents |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Simulation) | Traditional SIR Model| Improvement |
|----------------|--------|-------------------------|----------------------|-------------|
| Bottleneck Pred. | Triage Collapse Time Acc | High Correlation | Low Correlation | Captures behavioral collapse |

# Limitations & Future Work
Behavioral simulation relies heavily on the LLM's stereotyping of human panic and fatigue, which may not accurately reflect trained medical professionals' actual crisis response.

# Relevance to Our Survey
Highly relevant for **EHR & Administrative / Operations**. A milestone paper (100 agents) showing the application of Generative Agents for large-scale health systems engineering. Modality: Text (Simulation).
