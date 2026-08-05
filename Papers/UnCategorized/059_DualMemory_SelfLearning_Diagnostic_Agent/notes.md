# Joint Optimization of Reasoning and Dual-Memory for Self-Learning Diagnostic Agent
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.07269

# Problem Statement
Agents with a single vector database memory struggle to differentiate between "Semantic Memory" (general medical textbook knowledge) and "Episodic Memory" (specific patient case histories and past mistakes), leading to retrieved context that is cluttered and irrelevant.

# Key Contributions
1. Proposes a Dual-Memory architecture mirroring human cognitive psychology.
2. Semantic memory stores static clinical guidelines; Episodic memory stores the agent's past reasoning trajectories and post-hoc reflections.
3. An explicit "Memory Routing Agent" jointly optimizes which memory bank to query based on the current diagnostic impasse.

# Methodology / Architecture
When faced with a diagnostic query, the Router Agent decides if it needs factual rules (routes to Semantic DB) or experiential heuristics (routes to Episodic DB). The retrieved contexts are fused. If the agent makes an error, the Reflection module only updates the Episodic DB, keeping the Semantic DB pristine and hallucination-free.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Dual-Diag-Bench | Complex diagnostic cases | 3,000 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single Memory) | Improvement |
|----------------|--------|------------|-------------------------------|-------------|
| Diagnostic Acc | Top-1 Accuracy | 84.7% | 76.2% | +8.5% |
| Retrieval Noise| Irrelevant Context | 12% | 41% | -29% (Cleaner context) |

# Limitations & Future Work
The memory routing agent adds a layer of latency and API cost. Managing the eviction policy for the Episodic memory as it grows remains an open challenge.

# Relevance to Our Survey
Highly relevant for **Clinical Decision Support**. Solves the "context clutter" gap in continual learning agents by introducing explicitly segregated dual-memory banks. Modality: Clinical Text.
