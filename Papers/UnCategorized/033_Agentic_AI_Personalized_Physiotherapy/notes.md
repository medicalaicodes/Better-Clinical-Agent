# Agentic AI for Personalized Physiotherapy: A Multi-Agent Framework for Generative Video Training and Real-Time Pose Correction
**Authors:** N/A
**Year:** 2026 | **Venue:** ICDH IEEE
**ArXiv/DOI:** https://arxiv.org/abs/2604.21154

# Problem Statement
Traditional digital physiotherapy relies on static videos. Patients often perform exercises incorrectly at home without real-time feedback, leading to injury or lack of progress.

# Key Contributions
1. Proposes a real-time, closed-loop multi-agent system for home physiotherapy.
2. The "Vision Agent" tracks 3D skeletal poses in real-time via smartphone cameras.
3. The "Therapist Agent" issues dynamic verbal and visual corrections (e.g., "straighten your left knee").

# Methodology / Architecture
The architecture streams camera frames to a lightweight edge-VLM (Vision Agent) that emits skeletal coordinates. A cloud-based LLM (Therapist Agent) compares these coordinates to the prescribed exercise mechanics. A Speech Agent converts the feedback to natural audio.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| PhysioPose3D | Annotated physiotherapy movements | 10k video clips |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Pose Correction| Correction Delay | 400ms | 1,200ms (Cloud VLM) | -800ms (Faster) |
| Clinical Outcome| Adherence/Form | 92% proper form| 65% (Static Video)| +27% |

# Limitations & Future Work
Requires good lighting and a clear camera angle. Fails when the patient is partially occluded by furniture.

# Relevance to Our Survey
Relevant for **Medical Vision** and **Patient-Facing Copilots**. Highlights the use of edge-to-cloud multi-agent architectures to solve latency gaps in real-time visual healthcare tasks. Modality: Video/Skeletal Pose, Audio.
