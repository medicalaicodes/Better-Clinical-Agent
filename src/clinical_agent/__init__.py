"""
Clinical diagnostic agent with evidence-grounded medical retrieval.

Sub-packages (import what you need; nothing heavy is loaded here):
    retrieval/   corpora, BM25 + MedCPT indexes, hybrid retriever
    rag/         medical QA methods: baselines and DDx-RAG
    evaluation/  MIRAGE benchmark runner and statistics
    tools/       the agent's specialist tools (chest X-ray, labs, knowledge)
    agent.py     the sequential diagnostic agent;  graph.py  its LangGraph version
"""
