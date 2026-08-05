# EHRFlow: A large language model-driven iterative multi-agent electronic health record data analysis workflow

**Publication:** *KDD 2024 Workshop on Artificial Intelligence and Data Science for Healthcare (AIDSH)*.

# Key Features and Objectives
*   **Trustworthy Automation**: EHRFlow automates the analysis of EHR data to reduce the skill barrier and repetitive manual labor. It utilizes existing tools and expert predictive models rather than relying on the LLM to generate unverified clinical answers directly.
*   **Hierarchical Dual-Loop Process**:
    *   *Outer Loop*: Handles coarse-grained planning and task decomposition.
    *   *Inner Loop*: Focuses on fine-grained task execution (dispatching steps, generating/executing code, performing reviews).
*   **Multi-Agent Architecture**: Coordinates four primary agents:
    1.  *PlanAgent*: Decomposes user queries into actionable plans.
    2.  *ToolAgent*: Manages the selection and application of pre-defined tools (APIs, function calls).
    3.  *CodeAgent*: Responsible for writing and executing the necessary code.
    4.  *ReviewAgent*: Evaluates execution results, provides feedback, and initiates refinements or error correction.
*   **Privacy-Preserving Design**: Only the data schema is shared with the LLM, while sensitive operations are executed locally.
