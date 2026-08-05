# MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making
**Authors:** N/A
**Year:** 2024 | **Venue:** NeurIPS (Oral)
**ArXiv/DOI:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Static vs. Dynamic Environments**: The evaluation relies on standardized, single-turn educational benchmarks (like MedQA and PubMedQA). Real clinical deployment involves high-stakes, multi-modal, dynamic coordination (EHR integrations, time-series data) which these static proxies do not capture.
2. **Subjectivity in Complexity**: The core framework adjusts agent collaboration based on perceived task complexity. However, creating a "ground truth" for complexity is challenging; human experts themselves show only moderate inter-rater reliability. The paper notes LLMs lack the clinical nuance to assess complexity reliably.
3. **Transparency and Safety**: The multi-agent debate and collaboration processes act as a "black box," masking the decisiveness-safety tradeoffs required in medicine. The absence of clear interpretability and autonomous verification presents a massive barrier for safe clinical deployment.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Failed to Sweep Benchmarks**: The framework achieved the best performance in 7 out of 10 evaluated medical benchmarks, meaning it fell short and was outperformed by single-agent or alternative baseline methods in the remaining 3 benchmarks.
2. **Low Correlation with Physicians**: When quantitatively evaluating the LLM's classification of medical complexity against ground-truth ratings provided by human physicians, the model exhibited poor correlation.
