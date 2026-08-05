# Caging the Agents: A Zero Trust Security Architecture for Autonomous AI in Healthcare
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.17419

# Problem Statement
Autonomous agents connected to hospital networks via API tools pose a massive cybersecurity and patient safety risk (e.g., an agent hallucinates a SQL `DROP TABLE` command on the EHR database).

# Key Contributions
1. Proposes a "Zero Trust Agentic Architecture".
2. Agents are placed in isolated containers ("Cages") with read-only access to clinical data by default.
3. Any write-action (ordering a drug, altering a record) is intercepted by a deterministic, rule-based "Warden System" that requires human cryptographic sign-off before executing.

# Methodology / Architecture
Security/Architecture paper. The Agent operates in a simulated environment. When it decides on an action, it issues an Intent. The Intent is passed to the Warden, which checks it against static hospital safety rules (e.g., "Is dose > max safe dose?"). If it passes, it is queued for human authorization.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| N/A | Security Architecture | N/A |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| N/A | N/A | N/A | N/A | N/A |

# Limitations & Future Work
The "Warden" rules are static and hard to maintain across thousands of possible medical edge cases.

# Relevance to Our Survey
Highly relevant for **EHR & Administrative** and **Safety/Ethics**. Defines how to safely deploy computer-use and SQL agents in live hospital IT systems. Modality: N/A.
