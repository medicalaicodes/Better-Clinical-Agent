# First, Do No Harm (With LLMs): Mitigating Racial Bias via Agentic Workflows
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.18038

# Problem Statement
Foundation models reflect the systemic biases present in historical clinical training data (e.g., underestimating pain in Black patients, biased eGFR calculations). Standard prompt engineering fails to robustly de-bias output.

# Key Contributions
1. Proposes an Agentic Workflow designed specifically to audit and mitigate demographic bias in clinical decision support tools.
2. Introduces a "Demographic Auditor Agent" that runs counterfactual patient profiles (swapping race/ethnicity) to test for output variance.
3. Automatically corrects treatment recommendations if unwarranted variance is detected.

# Methodology / Architecture
When the primary diagnostic agent generates a treatment plan, the Auditor Agent clones the prompt, changes the patient's demographic descriptors, and forces regeneration. If the new treatment plan differs significantly (e.g., prescribing a cheaper, less effective drug for a minority patient), an "Alignment Agent" halts the output and forces a guideline-grounded reconciliation.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedBias-Eval | Counterfactual patient vignettes | 4,000 paired vignettes |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Bias Mitigation | Output Variance (lower=better) | 1.2% | 18.5% (Base GPT-4) | -17.3% (Massive reduction in bias) |

# Limitations & Future Work
Triples the API cost per query due to the counterfactual generation step.

# Relevance to Our Survey
Extremely relevant for **Security & Safety Benchmarking** and **Ethics**. Identifies the gap of systemic bias in autonomous agents and provides a multi-agent architectural solution. Modality: Clinical Text.
