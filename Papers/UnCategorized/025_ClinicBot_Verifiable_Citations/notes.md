# ClinicBot: A Guideline-Grounded Clinical Chatbot with Prioritized Evidence RAG and Verifiable Citations
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.00846

# Problem Statement
Medical chatbots generate plausible but unverified advice. Clinicians cannot trust the output without spending significant time verifying the claims against source guidelines, negating the efficiency benefits of the AI.

# Key Contributions
1. ClinicBot: A chatbot framework enforcing strict verifiable citations for every clinical claim.
2. Prioritized Evidence RAG: A hierarchical retrieval system that prioritizes Grade A clinical guidelines over general medical literature or web search.
3. Achieves near 100% citation precision, reducing verification time for clinicians by 60%.

# Methodology / Architecture
The generation pipeline forces the LLM to output inline citations `[DocID, ParagraphID]`. A post-generation Verification Agent checks if the cited text logically entails the generated claim. If entailment fails, the response is regenerated or flagged.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Guideline-QA | QA pairs derived from clinical guidelines | 8k questions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Citation Precision | % Valid Citations | 98.5% | 76.2% (Standard RAG)| +22.3% |
| Verification Time | Clinician Time/Query| 45 seconds | 115 seconds | -60% time |

# Limitations & Future Work
Strict citation enforcement sometimes causes the model to refuse answering complex questions that require synthesizing multiple low-grade evidence sources.

# Relevance to Our Survey
Relevant for **Clinical Decision Support**. Focuses on the gap in "Trust and Verifiability" for generative models. Modality: Clinical Text.
