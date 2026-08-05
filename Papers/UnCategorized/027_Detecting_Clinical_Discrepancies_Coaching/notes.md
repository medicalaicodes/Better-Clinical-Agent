# Detecting Clinical Discrepancies in Health Coaching Agents: A Dual-Stream Memory and Reconciliation Architecture
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2604.27045

# Problem Statement
Long-term health coaching agents often contradict past advice when the patient's context changes (e.g., advising a high-protein diet to a diabetic patient who recently developed chronic kidney disease).

# Key Contributions
1. Proposes a Dual-Stream Memory architecture separating static medical facts from dynamic patient state.
2. Introduces a "Reconciliation Agent" that detects contradictions before generating the final coaching response.
3. Defines a new metric: Contextual Contradiction Rate (CCR).

# Methodology / Architecture
Stream A retrieves general clinical guidelines (e.g., ADA guidelines). Stream B retrieves the patient's longitudinal EHR. The Reconciliation Agent acts as a judge, comparing the generated advice against both streams to flag contraindications.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Coach-EHR-Bench | Simulated longitudinal coaching | 2k patient trajectories |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Contradiction | CCR | 2.1% | 15.4% | -13.3% (Safer) |

# Limitations & Future Work
The Reconciliation Agent sometimes overly restricts the primary agent, resulting in highly generic, unhelpful advice ("Please consult your doctor").

# Relevance to Our Survey
Crucial for **Patient-Facing Copilots** and **Memory/Reasoning Gaps**. Modality: Clinical Text (Longitudinal).
