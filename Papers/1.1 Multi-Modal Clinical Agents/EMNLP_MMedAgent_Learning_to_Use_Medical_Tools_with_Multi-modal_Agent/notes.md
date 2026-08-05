# MMedAgent: Learning to Use Medical Tools with Multi-modal Agent
**Authors:** N/A
**Year:** 2024 | **Venue:** EMNLP Findings
**ArXiv/DOI:** https://aclanthology.org/2024.findings-emnlp.510/

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Workflow Brittleness:** While the agent can select and use individual tools, multi-step workflows that require composing several dependent software outputs (e.g., interactive segmentation followed by registration) remain brittle.
2. **Clinical Reliability and Safety:** These agentic systems still lack the predictable reliability required for real-world clinical environments.
3. **Evidence-Grounding:** Ensuring the agent's reasoning is grounded in actual medical evidence during complex, multi-step analysis remains a persistent difficulty. 
4. **Static Pipelines:** Current implementations largely rely on static or pre-defined pipelines, lacking the dynamic adaptability necessary for unpredictable real-world clinical reasoning.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Instruction Adherence:** In benchmarking environments (like MedAgentBench), models frequently fail to follow instructions exactly, outputting invalid actions or adding conversational text when strict JSON/API formatting is required.
2. **Static Benchmark Limitations:** Existing benchmarks feature straightforward queries that fail to capture the multi-turn failure modes prevalent in real-world workflows, potentially inflating perceived performance.
3. **Data Sensitivity:** Performance drops off significantly when evaluating out-of-distribution or noisy data due to the complexity of diverse medical formats (e.g., varying EHR systems).
