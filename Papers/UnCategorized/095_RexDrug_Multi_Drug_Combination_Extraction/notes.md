# RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.08166

# Problem Statement
Extracting complex multi-drug combinations (e.g., "Drug A + Drug B is safe, but adding Drug C causes toxicity") from unstructured biomedical literature is incredibly difficult for standard NLP models.

# Key Contributions
1. Introduces RexDrug, a reasoning-enhanced agent designed specifically for extracting drug-drug interaction (DDI) tuples from full-text journal articles.
2. The agent uses a "Think-then-Extract" pipeline, where it first writes a summary of the biochemical mechanism before outputting the formal JSON knowledge graph.
3. Creates a highly reliable database of off-label drug combination safety profiles.

# Methodology / Architecture
Information Extraction Agent. Uses a two-stage process: 1) Mechanism Reasoning (CoT prompting to understand the pharmacokinetic interaction) and 2) Structured Extraction (outputting precise Subject-Predicate-Object JSON). Includes a Self-Correction agent that verifies the JSON against the original text.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| RexDrug-Corpus | Annotated full-text papers | 5,000 papers |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (BioBERT) | Improvement |
|----------------|--------|------------|-------------------------|-------------|
| DDI Extraction | F1 Score | 86.4% | 75.2% | +11.2% |

# Limitations & Future Work
Struggles with extracting dosages; it accurately identifies *that* two drugs interact, but often fails to extract the specific mg/kg threshold where the interaction occurs.

# Relevance to Our Survey
Relevant for **Research Agents** and **Clinical Text**. A prime example of using agentic CoT to automate the construction of massive medical knowledge graphs from literature. Modality: Scientific Text.
