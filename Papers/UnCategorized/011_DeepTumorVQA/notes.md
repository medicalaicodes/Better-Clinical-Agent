# DeepTumorVQA: A Hierarchical 3D CT Benchmark for Stage-Wise Evaluation of Medical VLMs and Tool-Augmented Agents
**Authors:** N/A (Open Source)
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** N/A

# Problem Statement
Traditional benchmarks for medical Vision-Language Models (VLMs) provide a single aggregate accuracy score, which fails to capture where the reasoning process breaks down in 3D CT scan analysis.

# Key Contributions
1. Introduces DeepTumorVQA, a hierarchical benchmark that decomposes diagnostic reasoning into four stages: Recognition, Measurement, Visual Reasoning, and Medical Reasoning.
2. Provides a tool-interaction environment for agents to use segmentation models and measurement tools.
3. Reveals that quantitative measurement is a significant bottleneck for current models, hindering later reasoning stages.

# Methodology / Architecture
The benchmark evaluates both direct VLM inference and tool-using agent configurations. Agents are given ground-truth tool-use traces for supervision. The dataset includes 476,000 questions covering 42 clinical subtypes across 9,262 3D CT volumes.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| DeepTumorVQA | 3D CT volumes with hierarchical QA | 476k questions, 9.2k volumes |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| 3D CT VQA | Stage-wise Accuracy | N/A | N/A | Establishes new baseline for stage-wise evaluation |

# Limitations & Future Work
Quantitative measurement remains a bottleneck. Future work involves improving the agents' ability to utilize external measurement tools effectively.

# Relevance to Our Survey
Highly relevant for **Medical Vision** and **Tool-Augmented Agents**. Highlights the need for hierarchical, stage-wise evaluation rather than aggregate metrics. Modality: 3D CT Imaging.
