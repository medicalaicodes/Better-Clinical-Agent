"""
Prompts for the medical QA methods.

The CoT and MedRAG prompts are copied word-for-word from the MedRAG toolkit
(github.com/Teddy-XiongGZ/MedRAG, src/template.py) so our baseline numbers are comparable
with the published ones. The agent's own prompts live in agent_prompts.py.
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
