# Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation in Full-Procedure Colonoscopy Videos
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.25645

# Problem Statement
Annotating dense lesions (polyps) in full, 30-minute colonoscopy videos is brutally tedious for gastroenterologists, hindering the development of computer-aided detection (CAD) systems.

# Key Contributions
1. Proposes Colon-Bench, where an Agentic Workflow automates the dense annotation of full-procedure videos.
2. The agent uses tools (frame samplers, bounding box generators, temporal smoothing) to automatically tag and track polyps across thousands of frames.
3. Generates a massive, high-quality dataset of annotated colonoscopy videos at 1/100th the cost of human annotation.

# Methodology / Architecture
A "Keyframe Agent" rapidly scrubs the video to find anomalous frames. A "Dense Annotation Agent" (using a VLM) draws bounding boxes. A "Temporal Consistency Agent" ensures the boxes track smoothly across adjacent frames. Human reviewers only audit the final output.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Colon-Bench | Annotated colonoscopy videos | 2,000 full procedures |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent Annotated)| Expert Human Annotated | Improvement |
|----------------|--------|-----------------------------|------------------------|-------------|
| Annotation Quality| IoU Agreement | 0.88 | 0.91 (Gold Standard) | Comparable, 100x cheaper |

# Limitations & Future Work
The agent sometimes loses tracking on fast camera pans or when the polyp is partially obscured by fluid.

# Relevance to Our Survey
Relevant for **Medical Vision** and **Research Agents**. Demonstrates how agentic workflows can be used to autonomously build the very datasets needed to train future medical AI. Modality: Medical Video.
