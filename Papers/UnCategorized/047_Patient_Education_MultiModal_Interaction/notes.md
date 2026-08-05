# Rethinking Patient Education as Multi-turn Multi-modal Interaction
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.14656

# Problem Statement
Standard patient education materials (discharge summaries, pamphlets) are static, filled with jargon, and poorly comprehended by patients with low health literacy.

# Key Contributions
1. Reframes patient education from "document generation" to an interactive, multi-agent conversational experience.
2. The agent generates on-the-fly, personalized diagrams and analogies based on the patient's real-time questions (e.g., generating an image of a clogged pipe to explain atherosclerosis).
3. Significantly improves patient comprehension and retention of discharge instructions.

# Methodology / Architecture
The architecture involves a "Health Literacy Agent" that gauges the user's reading level from their prompts. A "Didactic Agent" synthesizes the medical records. If the patient asks for visual clarification, a "Visual Generation Agent" (using a diffusion model) creates a tailored, safe medical diagram/analogy to accompany the text response.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Edu-Health-Lit | Patient comprehension QA | 400 patient scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Static Text) | Improvement |
|----------------|--------|------------|-----------------------------|-------------|
| Comprehension  | Teach-Back Score (1-10)| 8.7 | 5.2 | +3.5 points |

# Limitations & Future Work
The image generation model occasionally produces anatomically confusing analogies if not strictly prompted, requiring a safety filter.

# Relevance to Our Survey
Relevant for **Patient-Facing Copilots**. Highlights the use of Multi-Agent Systems and Multimodality (Text-to-Image) to solve the gap of patient health literacy. Modality: Text, Synthetic Images.
