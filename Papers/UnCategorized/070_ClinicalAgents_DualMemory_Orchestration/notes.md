# ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.26182

# Problem Statement
Deploying agents in real-time clinical settings requires balancing deep reasoning (which is slow) with fast heuristic matching (which is prone to errors).

# Key Contributions
1. Introduces ClinicalAgents, an orchestrator that uses Dual-Memory (Semantic and Episodic) to speed up multi-agent decision making.
2. If a new case perfectly matches an episodic memory (a past solved case), it bypasses the slow multi-agent debate entirely (Fast Thinking).
3. If the case is novel, it triggers full multi-agent deliberation (Slow Thinking).

# Methodology / Architecture
Inspired by Kahneman's System 1 and System 2 thinking. The Routing Agent checks the Episodic Vector DB. If cosine similarity > 0.95, it uses the cached diagnosis (System 1). Otherwise, it spawns a specialized debate team (System 2) and saves the result to the Episodic DB.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedQA | Standard diagnostic cases | 12k cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Always Debate) | Improvement |
|----------------|--------|------------|-------------------------------|-------------|
| Latency | Avg Time per Query | 3.2s | 14.5s | 4.5x faster |
| Accuracy | Top-1 Accuracy | 85.1% | 85.4% | Maintained accuracy |

# Limitations & Future Work
If the initial System 2 diagnosis was subtly wrong and cached, the agent will confidently repeat that error instantly in the future (Memory Poisoning).

# Relevance to Our Survey
Crucial for **Clinical Decision Support**. Solves the latency problem of Multi-Agent Systems by introducing System 1/System 2 routing via Episodic Memory. Modality: Clinical Text.
