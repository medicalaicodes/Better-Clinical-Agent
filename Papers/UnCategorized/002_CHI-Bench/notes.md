# CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?
**Authors:** David Lee, Sarah Johnson et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.16679

# Problem Statement
While AI agents excel at isolated tasks like medical coding, they fail at long-horizon operational workflows such as Prior Authorization, which require navigating complex UI/policies, handling MFA, and multi-step reasoning over prolonged periods (Computer-Use Agents).

# Key Contributions
1. Presents CHI-Bench, the first interactive benchmark for long-horizon healthcare administrative workflows.
2. Identifies the "compounding error" problem in current multimodal LLMs when executing 50+ step workflows.
3. Evaluates 10 SOTA VLMs/Agents on prior authorization, claim denials, and patient scheduling.

# Methodology / Architecture
CHI-Bench provides a sandboxed, fully interactive simulated environment of an EHR and insurance payer portal. Agents are tasked with completing workflows end-to-end. The evaluation engine measures step-wise accuracy, trajectory efficiency, and catastrophic failure rates.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| CHI-Bench | Simulated Administrative Workflows | 500 scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Prior Auth | End-to-End Success | 12.4% (GPT-4o) | N/A | Baseline established |
| Claim Denial | Trajectory Acc | 45.1% | N/A | Baseline established |

# Limitations & Future Work
The simulated environment does not capture the full chaotic nature of real-world payer portals (e.g., CAPTCHAs, unannounced UI updates). Future work needs to integrate robust visual error recovery.

# Relevance to Our Survey
Critical for the **Administrative & Operational** category. Highlights the operational failures and the "Complexity Gap" for visual computer-use agents in healthcare. Modality: GUI/Web, Clinical Text.
