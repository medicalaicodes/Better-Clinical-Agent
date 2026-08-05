# GAZE: Grounded Agentic Zero-shot Evaluation with Viewer-Level Tools and Literature Retrieval on Rare Brain MRI
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.00876

# Problem Statement
Evaluating VLMs on rare neurological diseases is difficult because training data is scarce. Current benchmarks don't test a model's ability to zero-shot diagnose by actively retrieving literature and using viewer-level tools (contrast adjustment, slice scrolling) on 3D MRIs.

# Key Contributions
1. GAZE Framework: Evaluates agents' zero-shot reasoning on rare brain MRIs.
2. Equips the agent with a Literature Retrieval tool (PubMed) and MRI Viewer tools (Window/Level, Scroll).
3. Demonstrates that tool-augmented zero-shot diagnosis outperforms fine-tuned monolithic models on rare diseases.

# Methodology / Architecture
The agent is placed in an interactive loop. It views a 2D slice, decides to scroll up/down, adjusts contrast to see white matter lesions better, retrieves similar cases from PubMed, and synthesizes a final diagnosis.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| NeuroRare-MRI | Rare neurological MRI dataset | 1,200 cases, 50 diseases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (GAZE) | Fine-tuned VLM | Improvement |
|----------------|--------|------------|---------------|-------------|
| Rare Disease Dx | Accuracy | 58.4% | 45.1% | +13.3% |

# Limitations & Future Work
Retrieval tool sometimes pulls irrelevant papers if the initial visual description is slightly off, leading to cascading diagnostic errors.

# Relevance to Our Survey
Highly relevant for **Medical Vision** and **Knowledge Retrieval**. Shows tool-use (RAG + Vision Tools) is essential for zero-shot rare disease diagnosis. Modality: 3D MRI, Text.
