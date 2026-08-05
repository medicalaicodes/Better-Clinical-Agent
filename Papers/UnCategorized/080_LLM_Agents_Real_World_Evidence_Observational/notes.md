# Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.22767

# Problem Statement
Conducting observational studies on real-world EHR databases (like MIMIC or OMOP) requires specialized SQL/Python skills, taking biostatisticians months to write the cohorts and run propensity score matching.

# Key Contributions
1. Evaluates if Autonomous Agents can conduct complete, end-to-end observational studies directly on OMOP databases.
2. The agent translates a clinical hypothesis (e.g., "Does drug A cause less bleeding than drug B?") into SQL cohort definitions, executes the queries, runs Python statistical matching (PSM), and outputs a survival curve (Kaplan-Meier).
3. Achieves 70% concordance with human-authored published studies.

# Methodology / Architecture
A Data Science Multi-Agent Framework. The "Epidemiologist Agent" defines the inclusion/exclusion criteria. The "SQL Agent" translates this to OMOP-CDM SQL. The "Statistician Agent" writes and executes Python code for Propensity Score Matching and Cox Proportional Hazards modeling.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| RWE-Agent-Bench | Published observational studies | 50 replicated studies|

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA (Text-to-SQL only) | Improvement |
|----------------|--------|------------|----------------------------------|-------------|
| Study Replication | Concordance Rate | 72% | 15% | +57% |

# Limitations & Future Work
The SQL agent frequently hallucinates column names if the OMOP database has custom, non-standard local extensions.

# Relevance to Our Survey
Highly relevant for **Research Agents** and **EHR & Administrative**. Proves agents can automate the generation of Real-World Evidence (RWE), significantly accelerating epidemiological research. Modality: SQL, Code, EHR Structured Data.
