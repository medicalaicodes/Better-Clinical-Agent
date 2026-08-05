# MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning
**Authors:** N/A
**Year:** 2026 | **Venue:** ICLR
**ArXiv/DOI:** https://arxiv.org/abs/2604.08203

# Problem Statement
Training Medical VLMs requires massive datasets of bounding boxes and pixel-level annotations created by expert radiologists, which is extremely expensive and unscalable.

# Key Contributions
1. Introduces MedVR, which uses Agentic Reinforcement Learning to achieve visual reasoning *without* bounding box annotations.
2. The agent is given a global image-level text label (e.g., "Pneumothorax"). It learns to use a "Cropping Tool" to find the pathology through trial and error, guided by an RL reward signal.
3. Achieves weakly-supervised localization comparable to models trained on expensive dense annotations.

# Methodology / Architecture
The agent starts with a wide view of an X-Ray. It outputs (x,y, zoom) coordinates to a cropping tool. If the resulting crop increases the confidence of a frozen image classifier for the known disease label, the agent receives a positive reward. Over time, the agent learns the spatial priors of diseases autonomously.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ChestX-ray14      | Global labels only | 112k images |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (MedVR) | Previous SOTA (Fully Supervised) | Improvement |
|----------------|--------|--------------------|----------------------------------|-------------|
| Localization   | IoU > 0.5 | 68.4% | 71.2% | -2.8% (but 0 annotations needed)|

# Limitations & Future Work
Training the RL agent is highly unstable and computationally expensive compared to standard supervised learning.

# Relevance to Our Survey
Relevant for **Medical Vision**. Identifies Agentic RL and tool-use as a solution to the massive data annotation bottleneck in medical imaging. Modality: 2D X-Ray, RL Actions.
