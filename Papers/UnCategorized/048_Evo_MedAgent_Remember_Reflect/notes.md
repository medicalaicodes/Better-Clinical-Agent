# Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.14475

# Problem Statement
Most clinical LLMs are amnesic; they do not learn from their own past mistakes on similar patients. If they misdiagnose a rare presentation of Lupus today, they will make the exact same mistake tomorrow.

# Key Contributions
1. Introduces Evo-MedAgent, a system featuring a long-term episodic memory bank and a "Reflection Agent".
2. When the agent is provided with the ground-truth diagnosis (post-hoc), it reflects on its incorrect reasoning path and saves a "clinical pearl" to a vector database.
3. On future cases, it retrieves these pearls to avoid repeating similar cognitive errors, demonstrating continual learning without weight updates.

# Methodology / Architecture
Uses a dual-loop architecture. The Inner Loop is standard ReAct-based diagnosis. The Outer Loop is the Reflection Engine. After a case concludes, if the diagnosis was wrong, the Reflection Agent compares its transcript to the true answer, extracts the missed clue (e.g., "I ignored the malar rash"), and stores it in the Memory Vector Store for future retrieval via RAG during the Inner Loop.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Continual-MedQA | Sequential diagnostic cases | 2,000 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Static LLM) | Improvement |
|----------------|--------|------------|----------------------------|-------------|
| Continual Learning| Accuracy (Last 500 cases)| 88.4% | 69.1% | +19.3% (Learns from history) |

# Limitations & Future Work
The vector database eventually becomes saturated with contradictory "pearls" if the reflection process is noisy, leading to a degradation in performance over very long horizons.

# Relevance to Our Survey
Crucial for **Clinical Decision Support**. Addresses the gap of continual learning and memory in autonomous medical agents, transitioning from stateless LLMs to stateful, self-improving agents. Modality: Clinical Text.
