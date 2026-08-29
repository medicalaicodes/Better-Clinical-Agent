"""
Clinical AI Agent: LangGraph State Machine Workflow

Academic Citations & Foundations:
- Graph-Based Agent State Machines:
  * LangGraph: "Building Stateful, Multi-Actor Applications with LLMs" (LangChain, 2024).
  * MedRAX: "A Medical Reasoning Agent for Chest X-ray" (Bo Wang Lab, 2025; arXiv:2502.02618) - LangGraph ReAct orchestration over medical tools.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

from .schemas import (
    PatientCase,
    DifferentialHypothesis,
    ToolResult,
    ClinicalReport,
)
from .agent import AttendingClinicalAgent


class ClinicalAgentState(TypedDict):
    case: PatientCase
    active_ddx: List[DifferentialHypothesis]
    evidence_history: Annotated[List[ToolResult], operator.add]
    completed_tools: Annotated[List[str], operator.add]
    selected_action: str
    clinical_indication: str
    search_query: Optional[str]
    turn_count: int
    max_turns: int
    final_report: Optional[ClinicalReport]


def build_clinical_graph(agent: Optional[AttendingClinicalAgent] = None) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph workflow for active clinical inquiry."""
    clinical_agent = agent or AttendingClinicalAgent()

    def intake_node(state: ClinicalAgentState) -> Dict[str, Any]:
        case = state["case"]
        initial_ddx, intake_evidence = clinical_agent.llm_client.formulate_initial_differential(case)
        evidence = [intake_evidence] if intake_evidence else []
        return {
            "active_ddx": initial_ddx,
            "evidence_history": evidence,
            "completed_tools": ["clinical_history_synthesis"] if intake_evidence else [],
            "turn_count": 0,
        }


    def reasoning_node(state: ClinicalAgentState) -> Dict[str, Any]:
        case = state["case"]
        active_ddx = state["active_ddx"]
        evidence_history = state["evidence_history"]
        completed_tools = state["completed_tools"]
        turn_count = state["turn_count"] + 1

        available_tools = [name for name in clinical_agent.tools.keys() if name not in completed_tools]
        if not available_tools or turn_count > state.get("max_turns", 4):
            return {
                "selected_action": "STOP",
                "clinical_indication": "Diagnostic workup limit reached or tools exhausted.",
                "search_query": None,
                "turn_count": turn_count,
            }

        action, indication, search_query = clinical_agent.llm_client.select_next_action(
            case=case,
            active_ddx=active_ddx,
            evidence_history=evidence_history,
            available_tools=available_tools,
        )

        return {
            "selected_action": action,
            "clinical_indication": indication,
            "search_query": search_query,
            "turn_count": turn_count,
        }

    def execution_node(state: ClinicalAgentState) -> Dict[str, Any]:
        case = state["case"]
        action = state["selected_action"]
        search_query = state.get("search_query")
        target_tool = clinical_agent.tools.get(action)
        if not target_tool:
            available = [name for name in clinical_agent.tools.keys() if name not in state["completed_tools"]]
            if available:
                action = available[0]
                target_tool = clinical_agent.tools[action]

        if target_tool:
            kwargs = {}
            if search_query:
                kwargs["query"] = search_query
            result = target_tool.run(case, **kwargs)
            return {
                "evidence_history": [result],
                "completed_tools": [action],
            }
        return {}

    def synthesis_node(state: ClinicalAgentState) -> Dict[str, Any]:
        case = state["case"]
        active_ddx = state["active_ddx"]
        evidence_history = state["evidence_history"]

        report = clinical_agent.llm_client.synthesize_clinical_report(
            case=case,
            active_ddx=active_ddx,
            evidence_history=evidence_history,
        )
        return {"final_report": report}

    def route_action(state: ClinicalAgentState) -> str:
        if state.get("selected_action", "").upper() == "STOP":
            return "synthesize"
        if state.get("turn_count", 0) >= state.get("max_turns", 4):
            return "synthesize"
        return "execute"

    # Build Graph
    workflow = StateGraph(ClinicalAgentState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("synthesize", synthesis_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "reason")
    workflow.add_conditional_edges("reason", route_action, {"execute": "execute", "synthesize": "synthesize"})
    workflow.add_edge("execute", "reason")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


class ClinicalAgentGraph:
    """Wrapper class for compiled LangGraph clinical state machine."""

    def __init__(self, model_name: Optional[str] = None):
        self.agent = AttendingClinicalAgent(model_name=model_name)
        self.graph = build_clinical_graph(self.agent)

    def run(self, case: PatientCase, max_turns: int = 4) -> ClinicalReport:
        initial_state: ClinicalAgentState = {
            "case": case,
            "active_ddx": [],
            "evidence_history": [],
            "completed_tools": [],
            "selected_action": "",
            "clinical_indication": "",
            "search_query": None,
            "turn_count": 0,
            "max_turns": max_turns,
            "final_report": None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_report")

