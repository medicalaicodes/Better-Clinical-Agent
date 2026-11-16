"""
Clinical diagnostic agent with evidence-grounded medical retrieval.

    retrieval/   the medical texts, the three search engines and the hybrid retriever
    rag/         medical question answering: published baselines and our diagnostic agent (DxAgent)
    evaluation/  the MIRAGE benchmark runner, statistics, and the sample patient cases
    tools/       the clinical agent's tools: chest X-ray, segmentation, blood tests, medical library
    agent.py     the clinical agent (a LangGraph state machine over those tools)
"""
