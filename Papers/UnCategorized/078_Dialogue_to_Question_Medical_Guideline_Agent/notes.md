# Dialogue to Question Generation for Evidence-based Medical Guideline Agent Development
**Authors:** N/A
**Year:** 2025 | **Venue:** ML4H
**ArXiv/DOI:** https://arxiv.org/abs/2603.23937

# Problem Statement
Building RAG agents grounded in clinical guidelines (e.g., NCCN for cancer) is hard because standard RAG retrieval struggles with complex, multi-turn conversational queries from doctors.

# Key Contributions
1. Proposes a synthetic data generation pipeline to convert dense, static medical guidelines into conversational Question-Answer pairs.
2. Uses an LLM to simulate a "Doctor-Agent" dialogue, where the doctor asks clarifying questions about a guideline, and the agent answers based *strictly* on the text.
3. This synthetic dataset is used to fine-tune a specialized embedding model, drastically improving retrieval performance for Guideline Agents.

# Methodology / Architecture
Data augmentation framework. It takes PDF guidelines, parses them into chunks, and prompts GPT-4 to generate complex, multi-turn clinical scenarios (the "Dialogue") that would logically require that specific chunk (the "Evidence") to answer.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Guide-Dial-Syn | Synthetic clinical dialogue | 50,000 pairs |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (BM25 + DPR) | Improvement |
|----------------|--------|------------|----------------------------|-------------|
| Evidence Retrieval| Recall@5 | 88.4% | 72.1% | +16.3% |

# Limitations & Future Work
The synthetic dialogues occasionally lack the idiosyncratic phrasing real doctors use in practice.

# Relevance to Our Survey
Relevant for **Patient/Physician-Facing Copilots**. Focuses on the data engineering required to make Guideline-Grounded Agents actually work in conversational settings. Modality: Clinical Text.
