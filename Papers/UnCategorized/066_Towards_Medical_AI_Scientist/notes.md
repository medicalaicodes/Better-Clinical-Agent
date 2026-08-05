# Towards a Medical AI Scientist
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.28589

# Problem Statement
While agents can assist in clinical tasks, the "holy grail" is an agent that can autonomously conduct novel medical research, formulate biological hypotheses, and write review papers without hallucinating fake citations.

# Key Contributions
1. Outlines a comprehensive framework for a "Medical AI Scientist" using hierarchical Tree-of-Thoughts (ToT).
2. The agent autonomously navigates PubMed, PMC, and ClinicalTrials.gov to find contradictory literature, synthesizes the conflict, and generates a novel research hypothesis.
3. Successfully authored 10 mini-reviews that expert physicians rated as "publishable quality".

# Methodology / Architecture
Employs a "Hypothesis Generator", a "Literature Reviewer" (with PubMed API tools), an "Experimental Designer" (proposing clinical trial protocols), and an "Academic Writer". The ToT framework allows the agent to backtrack if a literature search yields zero results.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| MedScientist-Eval | Expert evaluation of generated hypotheses | 50 hypotheses |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Hypothesis Quality | Expert Approval % | 78% | 34% (Vanilla GPT-4) | +44% |

# Limitations & Future Work
The agent currently cannot physically run wet-lab experiments, meaning its hypotheses are purely theoretical.

# Relevance to Our Survey
Highly relevant for **Research Agents**. Represents the pinnacle of autonomous medical research workflows. Modality: Scientific Text.
