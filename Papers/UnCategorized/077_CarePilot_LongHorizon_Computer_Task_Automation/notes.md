# CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare
**Authors:** N/A
**Year:** 2026 | **Venue:** CVPR Findings
**ArXiv/DOI:** https://arxiv.org/abs/2603.24157

# Problem Statement
Nurses and administrative staff spend up to 40% of their shifts navigating clunky EHR interfaces (Epic, Cerner) to process patient intakes and prior authorizations.

# Key Contributions
1. Introduces CarePilot, a pure Computer-Use Agent designed to operate standard Windows/Web EHR GUI interfaces.
2. The agent breaks down high-level intents ("Process intake for John Doe") into low-level mouse clicks and keyboard strokes.
3. Features a specialized "GUI Recovery Agent" that handles pop-ups, lag, or unexpected UI state changes.

# Methodology / Architecture
Uses a Hierarchical Multi-Agent system. The "Planner Agent" reads the clinical intent and outlines the required steps. The "Execution Agent" (a VLM) looks at the screen screenshot and outputs (X, Y) mouse clicks. The "Recovery Agent" triggers if the screen doesn't match the expected state (e.g., an Epic warning pop-up).

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| HealthGUI-1K | Simulated EHR UI tasks | 1,000 tasks |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (WebArena agents) | Improvement |
|----------------|--------|------------|---------------------------------|-------------|
| EHR Navigation | Task Success Rate | 71.2% | 38.5% | +32.7% |

# Limitations & Future Work
The agent is highly sensitive to UI updates. If the hospital updates the EHR software, the agent's spatial understanding degrades until re-trained.

# Relevance to Our Survey
Highly relevant for **EHR & Administrative**. A major step forward in autonomous Computer-Use Agents for reducing clinical burnout. Modality: GUI / Visual-Action.
