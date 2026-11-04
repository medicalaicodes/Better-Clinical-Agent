"""
Prompts for the medical QA methods.

The CoT and MedRAG prompts are copied word-for-word from the MedRAG toolkit
(github.com/Teddy-XiongGZ/MedRAG, src/template.py) so our baseline numbers are comparable
with the published ones. The DDx-RAG prompts (PLAN, VERIFY, DECIDE) are ours.
"""

# ----------------------------------------------------------------------------------------
# Baselines (verbatim from MedRAG)
# ----------------------------------------------------------------------------------------

COT_SYSTEM = (
    'You are a helpful medical expert, and your task is to answer a multi-choice medical question. '
    'Please first think step-by-step and then choose the answer from the provided options. '
    'Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), '
    '"answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, '
    'so please have a definite answer.'
)

COT_USER = """
Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
"""

MEDRAG_SYSTEM = (
    'You are a helpful medical expert, and your task is to answer a multi-choice medical question '
    'using the relevant documents. Please first think step-by-step and then choose the answer from '
    'the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": '
    'Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research '
    'purposes only, so please have a definite answer.'
)

MEDRAG_USER = """
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
"""

# Used by i-MedRAG to answer each follow-up query (MedRAG's "simple_medrag" template).
SIMPLE_RAG_SYSTEM = "You are a helpful medical expert, and your task is to answer a medical question using the relevant documents."

SIMPLE_RAG_USER = """Here are the relevant documents:
{context}
Here is the question:
{question}"""

# i-MedRAG follow-up instructions (verbatim), but we ask for the queries as JSON directly
# instead of making a second LLM call to parse them out of free text.
IMEDRAG_SYSTEM = "You are a helpful medical assistant, and your task is to answer the given question following the instructions given by the user."

IMEDRAG_ASK = """{context}

Here is the question:
{question}

{options}

Please first analyze all the information in a section named Analysis (## Analysis). Then, use key terms from previous answers to form specific and direct questions. Generate {n_queries} concise, context-specific queries to search for additional information in an external knowledge base. Each query should be simple and focused, directly relating to the key terms used in the answers.
Finish with the queries as JSON: {{"queries": ["query 1", "query 2", ...]}}"""

# ----------------------------------------------------------------------------------------
# RAG²-style rationale query (Sohn et al., NAACL 2025), written from the question only
# ----------------------------------------------------------------------------------------

RATIONALE_USER = """Here is a medical question (answer options are hidden):
{question}

Write a short rationale (3-4 sentences) explaining the medical knowledge needed to answer it.
Use textbook terminology. Do not hedge."""

# ----------------------------------------------------------------------------------------
# DDx-RAG (ours)
# ----------------------------------------------------------------------------------------

PLAN_SYSTEM = "You are a careful senior physician who plans what to look up before answering."

PLAN_USER = """Here is a medical question. The answer options are hidden on purpose.

{question}

Plan the literature check before answering:
1. "findings": the key facts in the question (patient findings, or the facts being asked about). Max 8, short.
2. "hypotheses": the 2-4 most likely answers (diagnoses, mechanisms, drugs, next steps, ...) ranked from most to least likely. Short names.
3. "queries": short textbook-style search queries (max 12 words each), each with an "intent":
   - "SUPPORT": typical features of one hypothesis
   - "DISCRIMINATE": what distinguishes the top two hypotheses
   - "REFUTE": findings that would argue AGAINST your leading hypothesis
   Give each query an "about" field naming the hypothesis it targets. Write 4-6 queries in total,
   including at least one DISCRIMINATE and one REFUTE query.

Return only JSON:
{{"findings": ["<finding>", ...], "hypotheses": ["<most likely answer>", ...], "queries": [{{"intent": "SUPPORT", "about": "<hypothesis>", "text": "<search query>"}}, ...]}}"""

VERIFY_SYSTEM = "You are a meticulous medical fact-checker. You only state what the given evidence says."

VERIFY_USER = """Question (answer options hidden):
{question}

Key findings: {findings}

Hypotheses to check:
{hypotheses}

Reference evidence:
{evidence}

For EACH hypothesis, list what the evidence says FOR it and AGAINST it, given the findings.
Rules: use only the evidence above; every point must cite one evidence id (e.g. "E3");
if the evidence says nothing about a hypothesis, give empty lists and verdict "no evidence".

Return only JSON:
{{"hypotheses": [{{"name": "<hypothesis>", "for": [{{"point": "<what the evidence says>", "evidence": "E1"}}], "against": [{{"point": "<what the evidence says>", "evidence": "E2"}}], "verdict": "supported | weakened | refuted | no evidence"}}]}}"""

DECIDE_SYSTEM = (
    "You are a helpful medical expert answering a multi-choice question. You weigh reference "
    "evidence carefully and you do not let irrelevant documents distract you."
)

DECIDE_USER = """Here is the question:
{question}

Here are the potential choices:
{options}

Reference evidence:
{evidence}
{verification}{votes}
Instructions:
- Think step by step. Use the evidence where it is relevant; ignore evidence that is off-topic.
- Check your chosen answer against the evidence that argues against it.
{vote_rule}
Organize your output in json: {{"step_by_step_thinking": "<your reasoning>", "answer_choice": "<one option letter>", "evidence_used": ["E1", ...]}}"""

VOTE_RULE = (
    "- The tallies above come from answering WITHOUT references. If you choose a different answer "
    "than the most common one, you must cite the evidence that justifies the change."
)
