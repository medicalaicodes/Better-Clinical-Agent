# Camyla: Scaling Autonomous Research in Medical Image Segmentation
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.10696

# Problem Statement
Discovering novel neural network architectures for medical image segmentation (like U-Net variants) traditionally takes months of human researcher trial-and-error. 

# Key Contributions
1. Introduces Camyla, an autonomous "AI Scientist" agent tailored specifically for medical imaging research.
2. The agent autonomously writes PyTorch code for novel segmentation architectures, runs the experiments on GPUs, analyzes the dice scores, and iterates.
3. Automatically discovered a novel segmentation architecture that outperforms the standard nnU-Net on 3D cardiac MRI.

# Methodology / Architecture
The framework uses an "Idea Generation Agent" that reads arXiv papers to propose architectural tweaks. A "Coding Agent" implements the PyTorch model. An "Execution Agent" runs the training loop on a cluster. A "Reviewer Agent" analyzes the loss curves and suggests the next iteration.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| ACDC Cardiac MRI  | Cardiac segmentation | 150 patients |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent-Discovered)| Previous SOTA (nnU-Net) | Improvement |
|----------------|--------|------------------------------|-------------------------|-------------|
| Heart Seg | Dice Score | 0.932 | 0.918 | +0.014 |

# Limitations & Future Work
Burns massive amounts of GPU compute running failed experiments. The agent sometimes introduces "bugs" that accidentally improve the loss through data leakage.

# Relevance to Our Survey
Crucial for **Research Agents** and **Medical Vision**. Validates that autonomous AI scientists can now generate and execute code to discover novel medical vision architectures. Modality: PyTorch Code, 3D MRI.
