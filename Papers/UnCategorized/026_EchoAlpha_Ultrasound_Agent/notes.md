# Echo-α: Large Agentic Multimodal Reasoning Model for Ultrasound Interpretation
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.28011

# Problem Statement
Ultrasound interpretation requires continuous spatiotemporal reasoning over video streams, unlike static X-rays. Current VLMs struggle with the high noise and dynamic nature of ultrasound.

# Key Contributions
1. Echo-α: A multimodal agent that can autonomously rewind, pause, and measure specific frames in an ultrasound video loop.
2. Integrates a temporal reasoning module specifically trained on fetal and cardiac echocardiograms.
3. Achieves SOTA on dynamic measurement tasks (e.g., Ejection Fraction calculation).

# Methodology / Architecture
The agent uses a "Video Navigator" tool that acts on the video buffer, allowing the LLM to selectively sample high-quality frames from a noisy continuous feed. It then applies specialized segmentation tools to those frames.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| EchoNet-Dynamic | Cardiac ultrasound | 10k videos |
| FetalUS-Video | Fetal ultrasound videos | 5k videos |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Ejection Fraction | MAE | 4.1% | 6.2% | -2.1% (Lower Error) |

# Limitations & Future Work
Highly computationally expensive; cannot run in real-time on edge devices inside portable ultrasound probes yet.

# Relevance to Our Survey
Highly relevant for **Medical Vision**. Highlights the gap between static image processing and spatiotemporal (video) agentic reasoning. Modality: Ultrasound Video.
