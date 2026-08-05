# When OpenClaw Meets Hospital: Toward an Agentic Operating System for Dynamic Clinical Workflows
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.11721

# Problem Statement
Hospitals use dozens of disconnected software systems (Epic, PACS, LIMS, paging systems). An agent operating in just one silo cannot manage a real clinical workflow.

# Key Contributions
1. Proposes an "Agentic Operating System" where an overarching Orchestrator Agent routes sub-tasks to specialized Computer-Use Agents (using the OpenClaw framework).
2. The Orchestrator can seamlessly move data between silos: e.g., reading a critical lab value in LIMS, logging it in Epic, and sending a pager alert to the attending physician.
3. Introduces a standardized API schema for hospital agent interoperability.

# Methodology / Architecture
Orchestration Framework. The OS acts as a routing layer. When a trigger occurs (e.g., "Patient X arrived"), the OS spawns an Intake Agent, a Triage Agent, and an Order-Entry Agent, managing the data flow between their respective software tools.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Hosp-Workflow-Eval| End-to-end simulated IT tasks | 500 multi-system tasks |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Single-system Agent)| Improvement |
|----------------|--------|------------|------------------------------------|-------------|
| Workflow Comp. | Cross-Silo Task Success | 78% | 12% | Massive (+66%) |

# Limitations & Future Work
Heavy reliance on screen-scraping (Computer-Use) makes the OS fragile to UI updates across the dozens of different hospital software systems.

# Relevance to Our Survey
Crucial for **EHR & Administrative** tasks. Defines the architecture for hospital-wide agent deployment rather than isolated, single-task bots. Modality: GUI / Visual-Action across multiple systems.
