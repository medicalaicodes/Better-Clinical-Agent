# MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions
**Authors:** N/A
**Year:** 2025 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2503.22678

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Clinical Safety & Governance:** Existing setups do not robustly assess clinical safety behaviors during information gathering. Unsupervised "self-evolving" mechanisms risk propagating biases, factual hallucinations, and errors into the memory buffers.
2. **Reality & Social Gaps:** The simulation struggles to replicate the personality, social context, and complexity of real-world patient interactions. Prompt-driven models often fail to incorporate longitudinal follow-up data or multi-party clinical context.
3. **Long-Term Dialogue Management:** The framework has difficulty managing long-term, unstructured consultation flows over extended interactions, missing the division of labor present in actual healthcare environments (e.g., primary care vs. specialist handoffs).

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Low Overall Accuracy:** Despite improving over the baselines (which typically cap around 20%), MedAgentSim's diagnostic accuracy on rigorous benchmarks (like NEJM-based datasets) remains constrained between 26% and 29%. This is vastly below the precision required for clinical practice.
2. **Turn-Limit Failures:** In evaluations like 3MDBench, dialogues are often strictly capped (e.g., at 28 turns). MedAgentSim frequently fails to reach a diagnosis within these efficiency limits, leading to interactions marked as "unresolved" or "incomplete".
