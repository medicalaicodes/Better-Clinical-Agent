# Literature Surveys & Research Progression

This directory contains the systematic literature surveys, hierarchical taxonomy definitions, gap analysis documents, and deep-read evaluations produced throughout the internship.

Files are organized in **chronological research order**, with descriptive names detailing their specific focus and significance.

---

# Chronological Research Progression

| File Name | Significance & Research Focus |
|---|---|
| **`01_landscape_survey_ai_agents_in_healthcare.md`** | **Broad Landscape Survey**: Establishes the paradigm shift toward modality-driven agentic AI in clinical environments. |
| **`02_clinical_agent_taxonomy_20_categories.md`** | **The 20-Category Taxonomy**: Defines the strict operational hierarchy framing all 300+ surveyed papers (13 Doctor-Facing, 4 Patient-Facing, 3 Infrastructure). |
| **`03_multimodal_clinical_agents_gap_analysis.md`** | **Category 1.1 Deep Analysis**: Critical evaluation of 27 primary multimodal agent papers, identifying **11 systemic failure modes**. |
| **`04_evidence_reliable_coordination_survey.md`** (`.pdf`) | **SOTA Baseline Survey**: Evaluates AgentRx, MedAgentBoard, MDAgents, CARE, CoMMa, and Cerebra across uncertainty representation and missingness. |
| **`05_erc_framework_strengths_and_flaws_assessment.md`** (`.pdf`) | **ERC Critical Stress-Test**: Analyzes theoretical strengths, vulnerabilities, and system failure risks of evidence-reliable coordination. |
| **`06_erc_vulnerability_fix_notes.txt`** | **Architectural Fix Notes**: Notes detailing specific design mechanisms to resolve ERC system vulnerabilities. |
| **`07_sota_clinical_mas_comparison_matrix.md`** | **Comparative Matrix**: Benchmark feature matrix comparing clinical multi-agent systems across key capability dimensions. |
| **`08_medagent_pro_lineage_and_verification_survey.md`** | **MedAgent-Pro Analysis**: Traces the evolution of hierarchical workflow planning and step-level verification. |
| **`09_deep_read_pass_medvqa_deeprare_confagents.md`** | **Deep-Read Evaluation**: Analyzes Med-VQA, DeepRare, ConfAgents, and ethical agentic framework boundaries. |
| **`10_medagent_pro_disc_haemorrhage_failure_analysis.md`** | **Tool Degradation Analysis**: Evaluates tool-level indicator failure modes (e.g. 16.5 F1 score drop when retinal disc tools degrade). |

---

# Detailed Modality Phase Breakdowns (`phase_breakdowns/`)

The `phase_breakdowns/` subdirectory details modality-specific ingestion and reasoning mechanisms:

- **`01_perception_foundations.md`**: Foundations of clinical perception and multimodal representation.
- **`02_textual_ehr_ingestion.md`**: Structured EHR data parsing and clinical note ingestion.
- **`03_visual_spatial_imaging.md`**: Radiology & visual modality reasoning (CXR, CT, MRI).
- **`04_time_series_and_genomics.md`**: Dynamic temporal data ingestion and symbolic genomics.
- **`05_active_knowledge_retrieval.md`**: Medical RAG and active knowledge retrieval strategies.
- **`06_multimodal_fusion_and_conflict_resolution.md`**: Multimodal late fusion, agent orchestration, and conflict resolution.
- **`07_omni_modal_late_fusion_synthesis.md`**: Comparative synthesis of unified token spaces vs. agentic late fusion.
