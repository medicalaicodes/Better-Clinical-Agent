# Towards Conversational Medical AI with Eyes, Ears and a Voice
**Authors:** Google DeepMind
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** N/A

# Problem Statement
Traditional medical AI is text-based, missing crucial clinical nuances present in audio and visual data (e.g., patient gait, respiratory patterns, skin changes).

# Key Contributions
1. Introduces the "AI co-clinician", a multimodal conversational AI that uses video and audio processing.
2. Employs a dual-agent architecture (Talker Agent and Planner Agent) to balance natural conversation with clinical safety.
3. Frames the future of medical AI as "triadic care" (clinician, patient, AI).

# Methodology / Architecture
The system uses a two-agent structure. The **Talker Agent** handles fluid, natural conversation. The **Planner Agent** runs in the background, monitoring the interaction to ensure clinical reasoning remains within safety constraints.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Simulation Study | Interface-blinded encounters | 120 encounters |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Differential Dx | Clinical Accuracy | Approaches PCP level | GPT-Realtime | Significant outperformance |

# Limitations & Future Work
Requires expert clinical supervision. Future work focuses on refining real-time processing of subtle visual cues.

# Relevance to Our Survey
Crucial for **Multimodal & Conversational Agents**. Demonstrates the shift from text-only chatbots to fully multimodal "co-clinicians" with dual-agent safety architectures. Modality: Audio, Video, Text.
