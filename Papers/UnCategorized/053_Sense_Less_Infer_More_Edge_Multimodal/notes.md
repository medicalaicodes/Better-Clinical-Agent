# Sense Less, Infer More: Agentic Multimodal Transformers for Edge Medical Intelligence
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.10404

# Problem Statement
Deploying massive multimodal agentic systems on edge devices (like a wearable ECG or a rural point-of-care ultrasound) is impossible due to memory and compute constraints.

# Key Contributions
1. Introduces a lightweight, agentic multimodal transformer designed for edge deployment.
2. The agent uses an active "sensing" tool: it only requests high-resolution image crops or full ECG waveforms when it is uncertain, heavily reducing compute.
3. Achieves cloud-level diagnostic accuracy on edge-level hardware (Raspberry Pi 5).

# Methodology / Architecture
The edge agent acts as a controller. It evaluates a low-resolution stream (e.g., 64x64 video). When the internal uncertainty metric crosses a threshold, the agent calls a `RequestHighRes` tool to fetch a 1024x1024 crop of that specific frame, conserving battery and bandwidth.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| POCUS-Edge | Point of Care Ultrasound | 5,000 clips |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Cloud VLM) | Improvement |
|----------------|--------|------------|---------------------------|-------------|
| Inference Energy | mJoules per frame | 45mJ | 850mJ (Cloud API) | 95% reduction |
| Diagnostic Acc | AUROC | 0.86 | 0.87 | Comparable |

# Limitations & Future Work
If the agent fails to trigger the high-res tool during a critical transient event (like a 1-second arrhythmia), the diagnosis is missed entirely.

# Relevance to Our Survey
Relevant for **Medical Vision** and **Edge Agents**. Addresses the compute/latency gap by using agentic tool-calling (requesting high-res conditionally) to optimize hardware constraints. Modality: Video, Time-Series.
