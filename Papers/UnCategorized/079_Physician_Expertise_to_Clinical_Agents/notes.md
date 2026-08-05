# From Physician Expertise to Clinical Agents: Preserving, Standardizing, and Scaling Physicians' Medical Expertise with Lightweight LLM
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.23520

# Problem Statement
Senior physicians have tacit knowledge (heuristics, edge-case intuition) that isn't written in textbooks. Standard LLMs lack this intuition.

# Key Contributions
1. Introduces a knowledge-extraction framework where an agent "interviews" senior physicians about complex cases, standardizes their reasoning into a formal logical graph, and injects it into a lightweight (7B) local LLM.
2. Allows hospitals to "clone" the specific expertise of their top specialists.
3. The resulting lightweight agent outperforms GPT-4 on the hospital's specific patient demographic.

# Methodology / Architecture
The "Interviewer Agent" uses active learning to ask physicians "Why did you choose X over Y here?" The responses are mapped into a Clinical Knowledge Graph (CKG). A 7B Llama-3 model is then LoRA fine-tuned on this specific CKG.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Local-Hospital-Eval| Internal hospital cases | 500 cases |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (7B Agent) | GPT-4 | Improvement |
|----------------|--------|-----------------------|-------|-------------|
| Local Diagnostic Acc| F1 Score | 89.2% | 83.5% | +5.7% |

# Limitations & Future Work
The extraction process still requires dozens of hours of valuable time from senior physicians to answer the agent's interview questions.

# Relevance to Our Survey
Relevant for **Clinical Decision Support**. Demonstrates "Agentic Knowledge Engineering"—using agents to extract tacit human knowledge to build better local agents. Modality: Clinical Text.
