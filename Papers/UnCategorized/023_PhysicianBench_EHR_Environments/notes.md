# PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.02240

# Problem Statement
LLMs perform well on multiple-choice medical questions but fail when dropped into a complex, messy Electronic Health Record (EHR) system to retrieve information and reason about a patient's care.

# Key Contributions
1. PhysicianBench: A fully interactive benchmark built on top of a simulated EPIC/Cerner-style EHR interface.
2. Evaluates agents on their ability to navigate tabs, synthesize longitudinal notes, and place mock orders.
3. Shows that ReAct-based agents suffer from severe "navigation looping" when searching for buried lab results.

# Methodology / Architecture
The benchmark uses WebVoyager-style computer-use agents that interact with a web-based EHR clone. The evaluation metrics focus on Task Completion Rate and Trajectory Efficiency (number of clicks/queries to find the answer).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| PhysicianBench | Simulated EHR scenarios | 500 clinical tasks |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| EHR Navigation | Task Completion | 42.5% | N/A | Establishes baseline |

# Limitations & Future Work
Does not evaluate natural language interactions with patients, only the UI navigation aspect of the physician's job.

# Relevance to Our Survey
Crucial for **Administrative & Operational (Computer-Use)** agents. Highlights UI navigation gaps in EHR systems. Modality: GUI/Web, Text.
