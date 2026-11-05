"""
Clinical AI Agent: Attending Clinical Diagnostic Agent

Academic Citations & Foundations:
- Sequential Reasoning and Acting (ReAct) paradigm:
  * Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023; arXiv:2210.03629).
- Clinical Multi-Turn Active Inquiry loop:
  * SDBench: "A Benchmark for Sequential Diagnosis with Language Models" (Microsoft Research, 2025; arXiv:2506.22405).
  * MedAgent-Pro: "Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow" (Wang et al., ICLR 2026; arXiv:2503.18968).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618).
"""

from typing import Dict, List, Optional, Callable
from .schemas import (
    PatientCase,
    ToolResult,
    ClinicalReport,
)
from .config import get_settings
from .llm_client import ClinicalLLMClient
from .tools import (
    BaseClinicalTool,
    ChestXRayClassifierTool,
    MedSAMSegmenterTool,
    LaboratoryTestsTool,
    MedicalKnowledgeRAGTool,
)


class AttendingClinicalAgent:
    """
    Open-World Attending Clinical AI Agent.
    Implements a sequential diagnostic reasoning-and-acting (ReAct) loop:
    1. Formulates initial differential from intake.
    2. Sequentially orders the most informative diagnostic tests based on prior tool outputs.
    3. Passes targeted parameters/queries to specialist tools.
    4. Accumulates multimodal evidence and refines clinical hypotheses.
    5. Synthesizes a comprehensive consultation report and disposition.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.llm_client = ClinicalLLMClient(model_name=model_name)
        
        # Specialist Tools Hub (Diagnostic Perception, Validation, and Guidelines)
        self.tools: Dict[str, BaseClinicalTool] = {
            "chest_xray_classifier": ChestXRayClassifierTool(),
            "medsam_segmenter": MedSAMSegmenterTool(),
            "laboratory_tests": LaboratoryTestsTool(),
            "medical_knowledge_rag": MedicalKnowledgeRAGTool(),
        }

    def register_tool(self, tool: BaseClinicalTool):
        """Registers a new specialist diagnostic tool into the agent's hub."""
        self.tools[tool.name] = tool

    def diagnose(
        self,
        case: PatientCase,
        max_turns: Optional[int] = None,
        on_turn_callback: Optional[Callable[[int, str, str, Optional[str], ToolResult, int], None]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> ClinicalReport:
        """
        Executes the full active diagnostic inquiry loop on a patient case with token streaming.
        """
        max_steps = max_turns or self.settings.max_inquiry_turns
        evidence_history: List[ToolResult] = []
        completed_tools: List[str] = []

        # 1. LLM Clinical Intake & Pre-Test Differential Formulation
        active_ddx, intake_evidence = self.llm_client.formulate_initial_differential(
            case=case,
            streaming_callback=streaming_callback,
        )
        if intake_evidence:
            evidence_history.append(intake_evidence)


        # 3. Active Sequential Inquiry Loop
        for turn_idx in range(1, max_steps + 1):
            available_tool_names = [name for name in self.tools.keys() if name not in completed_tools]
            if not available_tool_names:
                break

            # Lead Attending LLM evaluates accumulated evidence and selects next diagnostic action
            selected_action, indication, search_query = self.llm_client.select_next_action(
                case=case,
                active_ddx=active_ddx,
                evidence_history=evidence_history,
                available_tools=available_tool_names,
                streaming_callback=streaming_callback,
            )

            if selected_action.upper() == "STOP":
                if on_turn_callback:
                    on_turn_callback(turn_idx, "STOP", indication, None, None, len(evidence_history))
                break

            target_tool = self.tools.get(selected_action)
            if not target_tool:
                selected_action = available_tool_names[0]
                target_tool = self.tools[selected_action]

            # Execute specialist tool on case data with LLM query if specified
            tool_kwargs = {}
            if search_query:
                tool_kwargs["query"] = search_query

            tool_result = target_tool.run(case, **tool_kwargs)
            evidence_history.append(tool_result)
            completed_tools.append(selected_action)

            # Interactive Turn Trace Callback
            if on_turn_callback:
                on_turn_callback(turn_idx, selected_action, indication, search_query, tool_result, len(evidence_history))

        # 4. Final Consultation Report Synthesis
        report = self.llm_client.synthesize_clinical_report(
            case=case,
            active_ddx=active_ddx,
            evidence_history=evidence_history,
            streaming_callback=streaming_callback,
        )

        return report
