# Phase 2: Textual and Structured Data Ingestion

# Core Concept
Ingesting language, structured records, and historical text to form a baseline patient profile. Before an agent can analyze complex images or dynamic signals, it must understand the patient's story.

# 1. Natural Language & Conversations
*   **The Challenge**: Managing multi-turn dialogues, maintaining contextual continuity, and interpreting patient-reported symptoms. Humans speak in fragmented sentences, not data tables.
*   **Contextual Continuity**: Connecting facts across turns (e.g., Turn 1: "My head hurts", Turn 15: "Light makes it worse" -> Recognizing a migraine pattern).
*   **Interpreting Symptoms**: Translating casual phrasing ("heavy weight on chest") into clinical concepts (angina).
*   **SOTA Examples**: 
    *   `Doctoragent-rl`: Multi-agent collaborative reinforcement learning for multi-turn clinical dialogues. Conducts structured questioning to refine hypotheses.
    *   `Medagents`: LLMs as collaborators simulating multidisciplinary discussion.
    *   `Cactus`: Conversational agents for cognitive behavioral therapy.
*   **Open Challenges**: Dialogue systems trained on static datasets struggle with real-world unpredictability. Sustaining long-term coherence requires domain-specific reasoning.

# 2. Electronic Health Records (EHRs)
*   **The Challenge**: Navigating structured data environments by translating high-level clinical intent into executable queries (SQL, Python, FHIR mappings) to query patient databases.
*   **Translating Intent into Code**: A doctor's request ("kidney issues") is mapped to ICD-10 codes, and an SQL query is generated to retrieve relevant lab results.
*   **SOTA Examples**:
    *   `EHRFlow`: Uses multiple agents to decompose complex physician requests into smaller analytical tasks.
    *   `TrustEHRAgent`: Acts as a verifier estimating confidence in its outputs and abstaining from uncertain predictions to maintain clinical safety.
*   **Open Challenges**: Errors in code generation can lead to unsafe conclusions or incorrect cohort selection. Privacy requires secure local computation.

# 3. Clinical Notes
*   **The Challenge**: Extracting temporal data to reconstruct longitudinal, coherent clinical timelines from years of fragmented, unstructured, free-text entries filled with jargon, copy-pasted text, and contradictions.
*   **SOTA Architectures**:
    *   **Infherno (Agentic FHIR Resource Synthesis)**: Automates translating free-form notes into HL7 FHIR resources. Uses code execution loops (Python `fhir.resources` module) instead of raw JSON generation to avoid errors. Employs RAG to query terminology servers (validating against SNOMED CT ontology).
    *   **CARE-AD (Collaborative Analysis and Risk Evaluation for Alzheimer's Disease)**: Multi-agent framework predicting Alzheimer's risk using longitudinal EHR notes. Deploys specialized LLM sub-agents simulating a multidisciplinary diagnostic board. Capable of predicting risk up to 10 years before official diagnosis.
*   **Open Challenges**: Longitudinal reasoning is difficult as disease indicators are sparse and distributed across years. Adhering to output schemas (FHIR) is tough. Interpreting implicit knowledge and contextual shorthand requires deep understanding.
