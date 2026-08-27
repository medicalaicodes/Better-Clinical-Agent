# Attending Physician Orchestrator: Clinical Specialist Skills & Diagnostic Protocols

This specification defines the complete clinical specialist toolkit available to the **Attending Physician Orchestrator Agent**. The Orchestrator dynamically consults these specialist tools based on the patient's acute presentation, current differential hypotheses, and clinical indication.

---

## 1. Specialist Diagnostic Tools Catalog

### A. `chest_xray_classifier` (TorchXRayVision DenseNet-121)
- **Modality:** Chest Radiography (Imaging)
- **Clinical Role:** Frontline multi-pathology screening across 18 major cardiopulmonary conditions (Pneumonia, Consolidation, Edema, Effusion, Atelectasis, Cardiomegaly, Pneumothorax, etc.).
- **Diagnostic Output:** Calibrated probability scores (0.0 - 1.0) with per-pathology operating thresholds.
- **When to Call:** Unexplained acute dyspnea, hypoxemia, fever with productive cough, pleuritic chest pain, or suspected heart failure / pneumonia / pneumothorax.

### B. `medsam_segmenter` (MedSAM ViT-Base)
- **Modality:** Chest Radiography / Anatomical Segmentation
- **Clinical Role:** Segments anatomical lung fields and thoracic boundaries to calculate left/right lung pixel area and evaluate thoracic symmetry ($A_L / A_R$).
- **Diagnostic Output:** Left/Right lung area in pixels, thoracic symmetry ratio, and focal volume reduction detection.
- **When to Call:** Evaluating volume loss, asymmetrical basilar consolidation, or severe pleural effusion.

### C. `laboratory_tests` (Emergency Laboratory Specialist)
- **Modality:** In Vitro Biomarkers (Labs)
- **Clinical Role:** Rapid quantitative assessment of inflammatory, cardiac, renal, metabolic, and coagulation biomarkers against established physiological reference intervals.
- **Key Biomarkers Evaluated:**
  - **BNP / NT-proBNP:** Evaluates ventricular wall stress and acute heart failure.
  - **Procalcitonin (PCT) & CRP:** Evaluates acute bacterial infection vs systemic inflammation.
  - **Troponin I / T:** Evaluates acute myocardial injury / ischemia.
  - **D-Dimer:** Evaluates active thrombosis / pulmonary embolism.
  - **WBC / Platelets / Hemoglobin:** Evaluates leukocytosis, anemia, and thrombocytopenia.
- **When to Call:** Any acute cardiopulmonary presentation, suspected sepsis, ACS, heart failure, pneumonia, or pulmonary embolism.

### D. `medical_knowledge_rag` (Open-World Clinical Practice Guidelines)
- **Modality:** Clinical Decision Support / Guidelines
- **Clinical Role:** Retrieves peer-reviewed evidence-based diagnostic criteria, gold-standard workup checklists, differential distinctions, and treatment protocols from NCBI StatPearls and specialty societies (IDSA, ATS, AHA, ACC, ESC) for any clinical condition.
- **When to Call:** When risk-stratifying patients, reviewing diagnostic workup protocols, or validating guideline-concordant next steps.

### E. `clinical_notes_extractor` (EHR Clinical History & Physical Exam)
- **Modality:** Free-Text Clinical Documentation (Notes)
- **Clinical Role:** Parses triage notes, patient history of present illness (HPI), and physical examination observations into structured clinical findings.
- **When to Call:** At patient intake to establish baseline symptoms and physical signs.

---

## 2. Active Inquiry Diagnostic Workflow

1. **Intake Evaluation:** Ingests patient presentation and formulates a ranked pre-test differential diagnosis.
2. **Sequential Active Inquiry:**
   - Evaluates what clinical data is needed to confirm or rule out top differential candidates.
   - Dynamically selects and executes the highest-yield specialist diagnostic tool.
   - Accumulates multi-modal evidence and refines hypothesis suspicion tiers (`HIGH_SUSPICION`, `MODERATE_SUSPICION`, `RULE_OUT`).
   - Concludes when diagnostic clarity is reached or tools are exhausted.
3. **Consultation Report Synthesis:**
   - Primary lead diagnosis identification.
   - Evidence-grounded clinical impression prose.
   - Actionable medical recommendations.
   - Recommended triage disposition (`ADMIT_INPATIENT`, `DISCHARGE_WITH_FOLLOWUP`, `ESCALATE_TO_PHYSICIAN`).
