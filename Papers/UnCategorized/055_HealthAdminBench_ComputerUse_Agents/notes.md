# HealthAdminBench: Evaluating Computer-Use Agents on Healthcare Administration Tasks
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.09937

# Problem Statement
While agents excel at text generation, there is no benchmark to evaluate "Computer-Use Agents" (agents that control a mouse and keyboard) on grueling healthcare administrative tasks like submitting prior authorizations or navigating clunky EHR GUIs (like Epic or Cerner).

# Key Contributions
1. Releases HealthAdminBench, the first evaluation suite for visual Computer-Use Agents in simulated healthcare administrative environments.
2. Evaluates agents on GUI navigation, data entry, insurance verification, and billing coding.
3. Reveals that SOTA generalist computer-use agents fail on medical GUIs due to complex drop-down menus and nested tabs.

# Methodology / Architecture
The benchmark provides agents with a virtual machine running a simulated, open-source EHR GUI (OpenEMR). The agent receives a natural language instruction (e.g., "Submit a prior authorization for an MRI for John Doe"). The agent must autonomously move the mouse, click tabs, type patient details, and submit the form.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| HealthAdminBench | GUI interaction trajectories | 500 administrative tasks |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Baseline) | Previous SOTA | Improvement |
|----------------|--------|-----------------------|---------------|-------------|
| Task Completion| Success Rate % | 34.2% | N/A | Establishes baseline |

# Limitations & Future Work
The open-source OpenEMR simulator does not perfectly replicate the proprietary complexities of major commercial EHRs (Epic/Cerner).

# Relevance to Our Survey
Crucial for **EHR & Operational Automation** and **Benchmarking Frameworks**. Identifies a massive gap: "Computer-Use Agents" are the missing link to truly automating hospital administration, but current SOTA agents are only at 34% success rates. Modality: GUI / Visual-Action.
