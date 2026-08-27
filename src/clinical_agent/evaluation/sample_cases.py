"""
Clinical AI Agent: Benchmark Clinical Evaluation Cases

Academic Citations & Foundations:
- Validated Clinical Radiograph Datasets:
  * NIH ChestX-ray14: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks" (CVPR 2017).
  * IEEE COVID-19 & Atypical Pneumonia Dataset: Cohen et al., "COVID-19 Image Data Collection" (arXiv:2003.11597).
  * TorchXRayVision Evaluation Zoo: Cohen et al. (MIDL 2022; arXiv:2111.00595).
- Clinical Vignettes Design:
  * NEJM Case Records of the Massachusetts General Hospital format.
  * SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
  * MedAgent-Pro Benchmark Suite (Wang et al., ICLR 2026; arXiv:2503.18968).
"""

from typing import List
from ..schemas import PatientCase

BENCHMARK_CASES: List[PatientCase] = [
    # Case 0: Streptococcus pneumoniae Lobar Pneumonia (Validated CXR from IEEE dataset)
    PatientCase(
        patient_id="PT_STREP_65M",
        age=65,
        gender="Male",
        chief_complaint="High fever (39.2 C), severe shaking rigors, and rusty/purulent productive cough for 3 days",
        vitals={"HR": 112, "BP": "124/78", "Temp": 39.2, "SpO2": 90, "RR": 26},
        clinical_notes="65-year-old male presents with acute high fever, rigors, and right-sided pleuritic chest pain. Auscultation reveals dense right upper lobe bronchial breath sounds, egophony, and inspiratory crackles. Denies orthopnea or pedal edema.",
        image_path="data/sample_images/streptococcus_pneumonia.jpg",
        available_labs={"WBC": 17.8, "PROCALCITONIN": 1.85, "CRP": 64.0, "BUN": 18.0, "CREATININE": 1.0, "LACTATE": 1.6},
    ),

    # Case 1: Acute Decompensated Heart Failure (Validated CXR from NIH ChestX-ray14)
    PatientCase(
        patient_id="PT_ADHF_71M",
        age=71,
        gender="Male",
        chief_complaint="Progressive exertional breathlessness, 3-pillow orthopnea, and worsening bilateral lower extremity swelling for 1 week",
        vitals={"HR": 92, "BP": "158/94", "Temp": 36.8, "SpO2": 93, "RR": 22},
        clinical_notes="71-year-old male with history of ischemic cardiomyopathy. Physical exam demonstrates elevated jugular venous distension to the angle of the jaw at 45 degrees, bilateral bibasilar inspiratory crackles, and 2+ pitting pedal edema to mid-calf.",
        image_path="data/sample_images/nih_cardiomegaly.png",
        available_labs={"BNP": 1680.0, "NT_PROBNP": 5200.0, "TROPONIN": 0.03, "CREATININE": 1.5, "BUN": 28.0, "SODIUM": 134.0, "POTASSIUM": 4.6},
    ),

    # Case 2: Acute Pulmonary Embolism (Validated clear radiograph)
    PatientCase(
        patient_id="PT_PE_48F",
        age=48,
        gender="Female",
        chief_complaint="Sudden onset sharp right-sided pleuritic chest pain and resting shortness of breath following a 14-hour international flight",
        vitals={"HR": 114, "BP": "118/74", "Temp": 37.1, "SpO2": 92, "RR": 26},
        clinical_notes="48-year-old female with no prior cardiopulmonary history. Developed acute pleuritic chest pain 6 hours after arriving on long-haul flight. Exam reveals right calf tenderness and moderate swelling with circumference 3cm greater than left calf.",
        image_path="data/sample_images/normal_cxr.jpg",
        available_labs={"D_DIMER": 3.85, "TROPONIN": 0.05, "WBC": 8.4, "PROCALCITONIN": 0.08, "CRP": 6.2, "LACTATE": 1.2},
    ),

    # Case 3: Costochondritis / Musculoskeletal Chest Wall Pain (Validated clear radiograph)
    PatientCase(
        patient_id="PT_COSTO_34M",
        age=34,
        gender="Male",
        chief_complaint="Sharp anterior chest pain aggravated by deep inspiration and heavy box lifting at work yesterday",
        vitals={"HR": 72, "BP": "120/78", "Temp": 36.6, "SpO2": 99, "RR": 14},
        clinical_notes="34-year-old male presents with sharp left parasternal chest pain that began after lifting heavy furniture. Pain is acutely reproducible on direct palpation over the 3rd and 4th left costochondral junctions. Clear breath sounds bilaterally, no friction rub.",
        image_path="data/sample_images/normal_cxr.jpg",
        available_labs={"TROPONIN": 0.01, "D_DIMER": 0.18, "WBC": 6.2, "CRP": 1.5, "BUN": 12.0, "CREATININE": 0.8},
    ),

    # Case 4: Pneumocystis jirovecii Pneumonia (PJP/PCP) in Immunocompromised Host (Validated CXR from IEEE dataset)
    PatientCase(
        patient_id="PT_PJP_HIV_40M",
        age=40,
        gender="Male",
        chief_complaint="Subacute dry nonproductive cough, exertional breathlessness, and low-grade fevers for 2 weeks in an immunocompromised patient",
        vitals={"HR": 106, "BP": "116/72", "Temp": 38.2, "SpO2": 84, "RR": 28},
        clinical_notes="40-year-old male with poorly controlled HIV (CD4 count 85 cells/uL). Presents with profound hypoxemia disproportionate to minimal auscultatory findings. Chest examination reveals faint bibasilar crackles without focal consolidation or wheezes.",
        image_path="data/sample_images/pneumocystis_pneumonia.png",
        available_labs={"WBC": 5.4, "PROCALCITONIN": 0.18, "CRP": 38.0, "LACTATE": 2.1, "BUN": 14.0, "CREATININE": 0.9},
    ),

    # Case 5: Klebsiella pneumoniae Severe Necrotizing Pneumonia (Validated CXR from IEEE dataset)
    PatientCase(
        patient_id="PT_KLEB_DM_62F",
        age=62,
        gender="Female",
        chief_complaint="Acute fever (39.5 C), severe dyspnea, and copious thick gelatinous blood-red 'currant-jelly' sputum for 2 days",
        vitals={"HR": 120, "BP": "104/64", "Temp": 39.5, "SpO2": 88, "RR": 30},
        clinical_notes="62-year-old female with poorly controlled type 2 diabetes (HbA1c 10.8%). Acute toxic appearance, profound tachypnea, and productive cough with characteristic thick dark red currant-jelly sputum. Severe right hemithorax dullness and bronchial breathing.",
        image_path="data/sample_images/klebsiella_pneumonia.jpg",
        available_labs={"WBC": 21.4, "PROCALCITONIN": 3.40, "CRP": 112.0, "GLUCOSE": 268.0, "BUN": 32.0, "CREATININE": 1.7, "LACTATE": 2.8},
    ),

    # Case 6: Legionnaires' Disease / Legionella pneumophila (Validated CXR from IEEE dataset)
    PatientCase(
        patient_id="PT_LEGIONELLA_60M",
        age=60,
        gender="Male",
        chief_complaint="High fever (39.8 C), dry cough, confusion, and profuse watery diarrhea for 5 days following a convention stay",
        vitals={"HR": 88, "BP": "110/68", "Temp": 39.8, "SpO2": 91, "RR": 24},
        clinical_notes="60-year-old male with acute pyrexia, mild encephalopathy/disorientation, nausea, and watery non-bloody diarrhea. Pulse rate 88 bpm demonstrates relative bradycardia in the setting of 39.8 C fever (Faget sign). Dense left lower lobe consolidation.",
        image_path="data/sample_images/legionella_pneumonia.png",
        available_labs={"SODIUM": 126.0, "POTASSIUM": 3.6, "WBC": 15.2, "PROCALCITONIN": 2.10, "CRP": 86.0, "BUN": 24.0, "CREATININE": 1.3},
    ),

    # Case 7: COVID-19 Acute Viral Hypoxemic Respiratory Infection (Validated CXR from Cohen et al.)
    PatientCase(
        patient_id="PT_COVID_VIRAL_58M",
        age=58,
        gender="Male",
        chief_complaint="Fever (38.6 C), complete loss of taste and smell (anosmia), fatigue, and progressive exertional shortness of breath for 6 days",
        vitals={"HR": 98, "BP": "128/82", "Temp": 38.6, "SpO2": 89, "RR": 24},
        clinical_notes="58-year-old male presents with persistent pyrexia, severe fatigue, sudden anosmia/ageusia, and hypoxemia. Lung exam reveals bilateral patchy end-inspiratory crackles throughout both mid- and lower-lung fields without localized lobar consolidation.",
        image_path="data/sample_images/covid_viral_pneumonia.jpg",
        available_labs={"WBC": 4.1, "PROCALCITONIN": 0.12, "CRP": 78.0, "D_DIMER": 1.45, "BUN": 16.0, "CREATININE": 1.0, "LACTATE": 1.5},
    ),
]
