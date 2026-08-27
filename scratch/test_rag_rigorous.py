from src.clinical_agent.tools.medical_knowledge import MedicalKnowledgeRAGTool

rag = MedicalKnowledgeRAGTool()

queries = [
    ("Legionella Presentation", "high fever, relative bradycardia, diarrhea, confusion, and hyponatremia with lung consolidation"),
    ("Heart Failure Presentation", "severe exertional dyspnea, orthopnea, elevated jugular venous distension, cardiomegaly, and bnp 1680"),
    ("Pulmonary Embolism Presentation", "sudden sharp pleuritic chest pain, unilateral calf swelling after international flight, elevated d-dimer"),
    ("PJP / PCP Presentation", "subacute dry nonproductive cough, severe hypoxemia in an immunocompromised HIV patient with low CD4"),
    ("Streptococcal Pneumonia", "acute high fever, rusty sputum, right upper lobe lobar consolidation and leukocytosis"),
    ("Klebsiella Necrotizing Pneumonia", "diabetic patient with currant-jelly sputum, upper lobe cavitation and bulging fissure"),
    ("Costochondritis Presentation", "sharp parasternal point tenderness reproducible on palpation over costochondral junction"),
    ("COVID-19 Viral Presentation", "fever, anosmia, loss of taste, bilateral ground glass opacities and lymphopenia"),
]

print("=" * 78)
print("              SEMANTIC RAG VECTOR SIMILARITY RIGOROUS TEST")
print("=" * 78)
for label, q in queries:
    matches = rag.semantic_search(q, top_k=2)
    top_doc, top_score = matches[0]
    second_doc, second_score = matches[1]
    print(f"Test: [{label}]")
    print(f"  Query:  '{q}'")
    print(f"  -> Match 1: {top_doc['disease']} [{top_doc['section']}] (similarity: {top_score:.3f})")
    print(f"     Excerpt: {top_doc['text'][:100]}...")
    print(f"  -> Match 2: {second_doc['disease']} [{second_doc['section']}] (similarity: {second_score:.3f})")
    print("-" * 78)
