# CareGuardAI: Context-Aware Multi-Agent Guardrails for Clinical Safety & Hallucination Mitigation in Patient-Facing LLMs
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.26959

# Problem Statement
Standard LLM guardrails rely on simple keyword blocking or generalized toxicity filters, which fail to catch subtle, context-specific medical hallucinations (e.g., prescribing aspirin to a patient on blood thinners).

# Key Contributions
1. CareGuardAI: A multi-agent framework dedicated entirely to real-time output moderation.
2. Uses distinct agents for Pharmacological checking, Diagnostic safety, and Tone/Empathy evaluation.
3. Significantly reduces unsafe outputs in patient-facing chat environments.

# Methodology / Architecture
The generated response is passed through a parallel ensemble of specialized Guardrail Agents. If any agent flags a high-severity clinical risk, the response is blocked and a fallback response is generated.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedSafety-Eval | Curated unsafe medical scenarios | 3k prompts |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Safety Violations | % Unsafe Outputs | 0.8% | 8.5% (Llama Guard) | -7.7% |

# Limitations & Future Work
Increases inference time significantly due to the parallel LLM calls required for the guardrails.

# Relevance to Our Survey
Addresses the core gap in **Security & Safety Benchmarking** for generative AI in medicine. Modality: Clinical Text.
