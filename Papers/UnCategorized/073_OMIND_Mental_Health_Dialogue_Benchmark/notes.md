# OMIND: Framework for Knowledge Grounded Finetuning and Multi-Turn Dialogue Benchmark for Mental Health LLMs
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.25105

# Problem Statement
Mental health chatbots often provide generic, platitude-heavy responses or offer dangerous clinical advice due to a lack of grounding in established psychological therapies (e.g., CBT, DBT).

# Key Contributions
1. Introduces OMIND, a framework that fine-tunes agents on psychological knowledge graphs.
2. Releases a rigorous multi-turn dialogue benchmark specifically testing empathetic engagement and adherence to Cognitive Behavioral Therapy (CBT) protocols.
3. The resulting agent acts as a supportive mediator, guiding users through structured reflection without hallucinating medical advice.

# Methodology / Architecture
Uses a "Knowledge-Grounded Agent" that queries a CBT-specific RAG database before generating a response. An "Empathy Auditor Agent" evaluates the proposed response and rejects it if it sounds too robotic or dismissive of the user's distress.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| OMIND-Bench | Multi-turn mental health dialogues| 8,000 sessions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Standard Chatbot) | Improvement |
|----------------|--------|------------|----------------------------------|-------------|
| Empathy Score  | Human-rated (1-5) | 4.3 | 3.1 | +1.2 |
| CBT Adherence  | Protocol Fidelity % | 89% | 45% | +44% |

# Limitations & Future Work
Cannot handle acute crisis scenarios (e.g., active self-harm); the agent hard-codes a pivot to emergency hotlines, which disrupts the conversational flow.

# Relevance to Our Survey
Crucial for **Patient-Facing Copilots** and **Benchmarking Frameworks**. Validates the need for specialized, knowledge-grounded agents in behavioral health. Modality: Text (Dialogue).
