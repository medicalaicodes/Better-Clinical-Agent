# HEAL-KGGen: A hierarchical multi-agent LLM framework with knowledge graph enhancement for genetic biomarker-based medical diagnosis

**Publication:** *BioRxiv* (2025).
**Authors:** Kaiwen Zuo, Z Zhong, P Huang, S Tang, Y Chen, Y Jiang.

# Key Features and Architecture
*   **Hierarchical Multi-Agent System**: Employs a structured team of agents:
    *   *General Practitioner (GP) Agent*: Initial triage of biomarkers.
    *   *Specialist Agents*: Genomics, transcriptomics, proteomics, and clinical interpretation.
*   **Dynamic Knowledge Graph Pipeline**: Dynamically builds and utilizes a medical knowledge graph.
    *   *Entity Extraction*: Identifies entities based on patterns and semantics.
    *   *Ontology-Aligned Normalization*: Maps data to UMLS, MeSH, and SNOMED CT.
    *   *Multi-Source Integration*: Constructs relationships from biomedical databases and literature.
*   **Explainable Reasoning**: Retrieves relevant subgraphs, transforms them into contextual prompts, and guides the LLM to provide explainable pathways.
*   **Performance**: Enhances question-answering accuracy significantly for genetic/molecular diagnostics (e.g., 43.75% increase with Claude 3.5 Sonnet).
