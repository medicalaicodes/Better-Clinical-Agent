# AI-Care: A Conversational Agentic System for Task Coordination in Alzheimer's Disease Care
**Authors:** Preyash Yadav, Michelle Cohn, et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** N/A

# Problem Statement
Managing daily tasks imposes a high cognitive load on individuals living with Alzheimer's disease (AD) and related dementias.

# Key Contributions
1. Develops an agentic AI layer built on a remote caregiving platform, co-designed with AD/ADRD patients.
2. Uses a stateful LangGraph-based orchestration for complex task management (to-do lists, calendar reminders).
3. Implements strict safety-constrained design, grounding critical info in caregiver-verified records.

# Methodology / Architecture
The system uses a LangGraph approach: sanitization/intent classification -> context/safety checks -> deterministic slot collection/tool execution -> response composition. It uses voice-first interaction (ElevenLabs) and controlled multi-turn clarification to avoid "silent failures" on ambiguous requests.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Pilot Study | AD/ADRD participants | 4 individuals |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Task Coordination| User Success/Trust | High trust & competence | N/A | Successful pilot completion |

# Limitations & Future Work
Small sample size in the pilot study. Requires larger clinical trials to validate long-term cognitive load reduction.

# Relevance to Our Survey
Highly relevant for **Patient-Facing Copilots** and **Operational Tasks**. Showcases safe, stateful tool orchestration (LangGraph) for vulnerable populations. Modality: Voice, Text.
