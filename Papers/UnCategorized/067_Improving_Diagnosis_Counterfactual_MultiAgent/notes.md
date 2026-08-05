# Improving Clinical Diagnosis with Counterfactual Multi-Agent Reasoning
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.27820

# Problem Statement
Standard diagnostic LLMs anchor heavily on the first few symptoms presented (Anchoring Bias).

# Key Contributions
1. Introduces a Counterfactual Reasoning framework specifically designed to break anchoring bias.
2. An agent explicitly generates scenarios like "If this patient did *not* have a fever, how would the differential diagnosis change?"
3. Outperforms standard ReAct prompting on complex MedQA diagnostic cases.

# Methodology / Architecture
The architecture involves a Primary Diagnostician, a Counterfactual Generator, and a Consensus Module. The Counterfactual generator forcibly alters one critical symptom in the context window and asks the Primary Diagnostician to re-evaluate, ensuring the diagnosis is robust to minor symptom variations.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedQA-Complex | Hard MedQA subset | 2,500 questions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (ReAct) | Improvement |
|----------------|--------|------------|-----------------------|-------------|
| Diagnostic Acc | Top-1 Accuracy | 86.3% | 80.1% | +6.2% |

# Limitations & Future Work
Requires significantly more token usage due to branching counterfactual generation.

# Relevance to Our Survey
Relevant for **Clinical Decision Support**. Demonstrates that forced counterfactual reasoning is a necessary architectural component for breaking cognitive biases in LLMs. Modality: Clinical Text.
