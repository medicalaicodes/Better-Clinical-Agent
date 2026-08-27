# Clinical AI Agent: Implementation Log & Architecture Records

## Date: 2026-08-27
## Scope: Clean-Start Open-World Multi-Agent Clinical AI System

---

### 1. Architectural Principles (Grounding in SOTA Frameworks)
- **Archived Legacy Implementation**: Archived all earlier files and experiments to [`backup/v1/`](file:///c:/PNS/IITMandi_Labwork/backup/v1/).
- **Studied Reference Repositories**:
  - `ref/MedAgent-Pro`: Two-level reasoning (disease-level guidelines + patient-level quantitative tool reasoning).
  - `ref/MedRAX`: Modular LangGraph ReAct tool execution with real PyTorch perception models.
  - `ref/MedRAG`: Structured medical knowledge retrieval.
- **Open-World Diagnostic Scope**: Removed rigid, hardcoded scoring rules (no hardcoded CURB-65, Wells, etc.) or forced shadow math. Built a general-purpose agentic AI loop capable of handling any medical presentation.

---

### 2. Implemented Components

1. **Core Data Schemas ([`src/clinical_agent/schemas.py`](file:///c:/PNS/IITMandi_Labwork/src/clinical_agent/schemas.py))**:
   - `PatientCase`: Patient demographics (age, gender), chief complaint, vitals, clinical notes, image paths, available labs.
   - `FindingItem`: Structured clinical finding with presence status, calibrated confidence, numerical value, and unit.
   - `ToolResult`: Standardized result packet returned by specialist tools.
   - `DifferentialHypothesis`: Condition name, ICD-10 code, suspicion tier (`HIGH_SUSPICION`, `MODERATE_SUSPICION`, `RULE_OUT`), clinical rationale.
   - `DiagnosticOrder`: Target tool name, display name, medical indication, execution parameters.
   - `ClinicalReport`: Primary diagnosis, ranked differential, key findings gathered, clinical impression prose, recommendations, and disposition (`ADMIT_INPATIENT`, `DISCHARGE_WITH_FOLLOWUP`, `ESCALATE_TO_PHYSICIAN`).

2. **Specialist Diagnostic Tools ([`src/clinical_agent/tools/`](file:///c:/PNS/IITMandi_Labwork/src/clinical_agent/tools/))**:
   - `ChestXRayClassifierTool`: Real `TorchXRayVision` DenseNet-121 model (`densenet121-res224-all`) classifying 18 pathologies.
   - `MedSAMSegmenterTool`: Real `MedSAM` ViT-B model segmenting lung fields and quantifying thoracic area symmetry.
   - `LaboratoryTestsTool`: Multi-panel biomarker validator across 20+ laboratory tests (CBC, BMP, Cardiac, Inflammatory, Coagulation).
   - `MedicalKnowledgeRAGTool`: Open-world guideline & knowledge retriever with **Semantic Vector/Cosine Similarity Matching** across clinical practice guidelines (IDSA, ATS, AHA, ACC, ESC, StatPearls).
   - `ClinicalNotesExtractorTool`: Parses free-text doctor/triage notes into structured symptoms and physical exam signs.

3. **Unified Reasoning Engine & Live Token Streaming ([`src/clinical_agent/llm_client.py`](file:///c:/PNS/IITMandi_Labwork/src/clinical_agent/llm_client.py))**:
   - Powered by LiteLLM supporting local Ollama (`medgemma`, `llama3`, `mistral`) and cloud models (Gemini, OpenAI, Claude).
   - Real-time token streaming (`stream=True`) directly to terminal stdout.
   - Sequential diagnostic reasoning: Attending LLM is provided the chronological evidence ledger and specifically reasons on the output of the tool it just called before selecting the next test or concluding with `STOP`.
   - Disabled remote cost map fetching (`os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"`).

4. **Attending Clinical Agent ([`src/clinical_agent/agent.py`](file:///c:/PNS/IITMandi_Labwork/src/clinical_agent/agent.py))**:
   - Pure, sequential diagnostic reasoning-and-acting (ReAct) loop.

5. **LangGraph State Machine ([`src/clinical_agent/graph.py`](file:///c:/PNS/IITMandi_Labwork/src/clinical_agent/graph.py))**:
   - Compiled `StateGraph` workflow: `intake` $\rightarrow$ `reason` $\rightarrow$ `execute` $\rightarrow$ `reason` $\dots \rightarrow$ `synthesize` $\rightarrow$ `END`.

---

### 3. Verification & Test Results
- `pytest tests/ -v`: **13/13 tests passed (100%)**.
- Live Active Inquiry Trace Verified:
  - **Case 0 (CAP)**: 63yo female with acute fever and productive cough $\rightarrow$ Initial ddx $\rightarrow$ Turn 1: CXR DenseNet-121 (consolidation/opacity) $\rightarrow$ Turn 2: Labs (WBC 16.8, Procalcitonin 1.45, CRP 52.0) conditioned on CXR $\rightarrow$ Turn 3: MedSAM symmetry conditioned on Labs $\rightarrow$ Turn 4: Semantic Guidelines RAG $\rightarrow$ Concluded with `Community-Acquired Pneumonia` (`ADMIT_INPATIENT`).
  - **Case 1 (ADHF)**: 71yo male with orthopnea and pedal edema $\rightarrow$ CXR (cardiomegaly/edema) $\rightarrow$ Labs (BNP 1420, NT-proBNP 4850) $\rightarrow$ MedSAM $\rightarrow$ Guidelines $\rightarrow$ Concluded with `Acute Decompensated Heart Failure` (`ADMIT_INPATIENT`).
  - **Case 3 (Costochondritis)**: 34yo male with chest wall tenderness $\rightarrow$ CXR clear $\rightarrow$ Labs normal (Troponin 0.01, D-Dimer 0.18) $\rightarrow$ MedSAM symmetric (1.00) $\rightarrow$ Concluded with `Costochondritis / Musculoskeletal Chest Wall Pain` (`DISCHARGE_WITH_FOLLOWUP`).
