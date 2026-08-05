# Design and Evaluation of a Culturally Adapted Multimodal Virtual Agent for PTSD Screening
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.17871

# Problem Statement
Standard text-based PTSD screening questionnaires suffer from high drop-out rates and lack cultural nuance, especially among marginalized populations who may mistrust standard clinical settings.

# Key Contributions
1. Develops a multimodal, avatar-based virtual agent that conducts verbal PTSD screenings (CAPS-5).
2. The agent adapts its conversational tone, dialect, and non-verbal cues (avatar facial expressions) based on the user's cultural demographic profile.
3. Significantly increases patient engagement and screening completion rates.

# Methodology / Architecture
The architecture combines an LLM dialogue manager with a 3D avatar rendering engine and a Voice Synthesizer. The agent's Persona (system prompt) and Visuals are dynamically parameterized at the start of the session to match the patient's cultural background, optimizing for empathy and trust.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Cultur-PTSD-Set | Multimodal screening sessions | 500 patient interactions |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Text Bot) | Improvement |
|----------------|--------|------------|--------------------------|-------------|
| Patient Engagement | Screening Completion Rate | 88.5% | 61.2% | +27.3% |
| Diagnostic Agreement| Concordance with Clinician | 91.0% | 85.5% | +5.5% |

# Limitations & Future Work
Requires high bandwidth for streaming the 3D avatar, making it difficult to deploy in rural areas with poor internet connectivity.

# Relevance to Our Survey
Relevant for **Patient-Facing Copilots** and **Multimodal Agents**. Addresses the gap of patient engagement and cultural competency in autonomous screening tools. Modality: Audio, Visual (Avatar), Text.
