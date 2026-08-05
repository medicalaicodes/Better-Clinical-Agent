# MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow
**Authors:** N/A
**Year:** 2025 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2503.18968

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Conservative LLM Behavior**: The LLM-based decider agent struggles with uncertainty and missing data. If critical clinical markers (like vCDR in glaucoma) are unavailable, the LLM fails to make a definitive diagnosis. It also hesitates when multiple indicators provide conflicting evidence (e.g., one normal indicator despite several abnormal ones).
2. **Cascading Errors from Inaccurate Tool Agents**: The final diagnosis is highly sensitive to the individual accuracy of the specialized tool agents. For indicators that lack robust detection models (such as Disc Hemorrhages), the system relies on general VQA and classification tools. These tools lack fine-grained perception, introducing high false-positive rates that degrade the reliability of the overall diagnosis. 

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Catastrophic F1 Drop on Missing Indicators**: In ablation studies, when the critical vCDR indicator is removed, the LLM decider's F1 score plummets to 14.3%.
2. **Negative Impact of Weak Indicators**: Adding the poorly performing Disc Hemorrhage (DH) indicator to the multiple-indicator suite actively harms performance. It causes a 5.1% drop in F1 score for the MOE decider and a massive 16.5% drop in F1 score for the LLM decider.
3. **Weak Single Indicator Performance**: Evaluated in isolation, certain tool-derived indicators perform poorly (e.g., DH only achieves 66.8% mACC and 29.6% F1).
4. **Baseline LLM Refusals**: Pure foundation models like GPT-4o entirely refused to provide clear diagnoses on these multi-modal inputs, resulting in missing metrics for baselines.
