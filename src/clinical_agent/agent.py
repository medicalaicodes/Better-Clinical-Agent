"""
The clinical agent: works up one patient the way an emergency doctor would, step by step.

    intake      read the notes and vitals, list the key findings, write a first differential
    choose      look at everything gathered so far and pick the most useful next test (or stop)
    run_tool    run that test: chest X-ray classifier, chest segmentation, blood tests, or a
                search of the medical library
    report      write the final report: diagnosis, disposition (admit / discharge / escalate),
                refined differential, impression and recommendations

The steps are wired together with LangGraph as a small state machine:

    intake -> choose -> run_tool -> choose -> run_tool -> ... -> choose -> report
                  (choose goes to "report" when it says STOP, or when the turn limit is reached)
"""

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import clinical_prompts
from .llm import LLM, parse_json
from .schemas import (ClinicalReport, DifferentialHypothesis, DispositionType, FindingItem, FindingStatus,
                      ModalityType, SuspicionTier, ToolResult)
from .tools import ChestXRayClassifierTool, LaboratoryTestsTool, MedicalKnowledgeRAGTool, MedSAMSegmenterTool


class WorkupState(TypedDict):
    """What LangGraph carries from one step to the next. (LangGraph needs the keys declared,
    which is why this one class uses TypedDict.)"""
    case: object
    differential: list
    evidence: list          # every ToolResult so far, in order
    tools_used: list
    next_tool: str          # a tool name, or "STOP"
    reason: str             # why that tool was chosen
    query: str              # the library search, if the next tool is the library
    turns: int
    max_turns: int
    report: object


def tier_from(text):
    """'high suspicion', 'HIGH_SUSPICION', 'rule out', ... -> a SuspicionTier."""
    text = str(text).upper()
    if "HIGH" in text:
        return SuspicionTier.HIGH_SUSPICION
    if "RULE" in text or "LOW" in text:
        return SuspicionTier.RULE_OUT
    return SuspicionTier.MODERATE_SUSPICION


def hypotheses_from(items):
    """The model's list of {"condition", "tier", "clinical_rationale"} -> DifferentialHypothesis objects."""
    hypotheses = []
    if not isinstance(items, list):
        return hypotheses
    for item in items:
        if not isinstance(item, dict) or not item.get("condition"):
            continue
        hypotheses.append(DifferentialHypothesis(
            condition=item["condition"],
            tier=tier_from(item.get("tier", "")),
            clinical_rationale=item.get("clinical_rationale", ""),
        ))
    return hypotheses


def unique_texts(items):
    """Strings with repeats (ignoring capital letters) and empty ones removed, order kept."""
    seen = []
    result = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        if item.strip().lower() in seen:
            continue
        seen.append(item.strip().lower())
        result.append(item.strip())
    return result


def differential_text(differential):
    lines = []
    for hypothesis in differential:
        lines.append(f"- {hypothesis.condition} [{hypothesis.tier.value}]: {hypothesis.clinical_rationale}")
    return "\n".join(lines)


class AttendingClinicalAgent:
    def __init__(self, model_name=None, llm=None):
        self.llm = llm or LLM(model=model_name)

        self.tools = {}
        for tool in [ChestXRayClassifierTool(), MedSAMSegmenterTool(), LaboratoryTestsTool(), MedicalKnowledgeRAGTool()]:
            self.tools[tool.name] = tool

        # Optional hooks for the demo: on_token prints the model's reply as it is written,
        # on_event is told about each step (what was decided, what a tool found).
        self.on_token = None
        self.on_event = None

    def ask(self, prompt):
        """One LLM call that must return a JSON object."""
        reply = self.llm.chat(prompt, on_token=self.on_token, json_mode=True)
        data = parse_json(reply)
        if data is None:
            raise ValueError("The model's reply was not JSON: " + str(reply)[:150])
        return data

    def tell(self, event, details):
        if self.on_event is not None:
            self.on_event(event, details)

    # ------------------------------------------------------------------------------------
    # The four steps (each one is a node in the graph)
    # ------------------------------------------------------------------------------------
    def intake(self, state):
        case = state["case"]
        prompt = clinical_prompts.INTAKE_PROMPT.format(
            age=case.age, gender=case.gender, chief_complaint=case.chief_complaint,
            vitals=json.dumps(case.vitals), clinical_notes=case.clinical_notes or "None")
        data = self.ask(prompt)

        differential = hypotheses_from(data.get("differential"))
        if not differential:
            differential = [DifferentialHypothesis(condition="Undifferentiated presentation",
                                                   clinical_rationale="The model gave no differential.")]

        # The notes' key positives and negatives become the first piece of evidence.
        positives = unique_texts(data.get("pertinent_positives"))
        negatives = unique_texts(data.get("pertinent_negatives"))
        findings = []
        for text in positives:
            findings.append(FindingItem(name="Clinical sign: " + text, confidence=0.98))
        for text in negatives:
            findings.append(FindingItem(name="Pertinent negative: " + text, status=FindingStatus.ABSENT,
                                        confidence=0.98))

        intake_result = ToolResult(
            tool_name="clinical_history_synthesis",
            modality=ModalityType.CLINICAL_NOTES,
            findings=findings,
            summary=f"Clinical intake: {len(positives)} positive feature(s), {len(negatives)} pertinent negative(s).",
            raw_output={"pertinent_positives": positives, "pertinent_negatives": negatives},
        )
        self.tell("intake", {"differential": differential, "intake": intake_result})
        return {"differential": differential, "evidence": [intake_result]}

    def choose(self, state):
        turns = state["turns"] + 1

        available = []
        for name in self.tools:
            if name not in state["tools_used"]:
                available.append(name)

        if not available or turns > state["max_turns"]:
            decision = {"next_tool": "STOP", "reason": "Turn limit reached or every tool used.", "query": "", "turns": turns}
            self.tell("choice", decision)
            return decision

        tool_lines = []
        for name in available:
            tool_lines.append(f"- '{name}': {self.tools[name].description}")

        evidence_lines = []
        for number, result in enumerate(state["evidence"], start=1):
            evidence_lines.append(f"- [Turn {number} - {result.tool_name}]: {result.summary}")

        case = state["case"]
        prompt = clinical_prompts.NEXT_TEST_PROMPT.format(
            age=case.age, gender=case.gender, chief_complaint=case.chief_complaint, vitals=json.dumps(case.vitals),
            differential_str=differential_text(state["differential"]),
            evidence_str="\n".join(evidence_lines) or "No tests ordered yet.",
            tools_str="\n".join(tool_lines))
        data = self.ask(prompt)

        wanted = str(data.get("selected_action", "")).strip()
        reason = str(data.get("clinical_indication", "")).strip()
        query = str(data.get("search_query", "") or "").strip()

        if wanted.upper() == "STOP":
            next_tool = "STOP"
        elif wanted in available:
            next_tool = wanted
        else:
            # A name we don't have (or one already used): take the first tool still available.
            next_tool = available[0]

        decision = {"next_tool": next_tool, "reason": reason, "query": query, "turns": turns}
        self.tell("choice", decision)
        return decision

    def run_tool(self, state):
        name = state["next_tool"]
        options = {}
        if state["query"]:
            options["query"] = state["query"]
        result = self.tools[name].run(state["case"], **options)

        self.tell("tool_result", result)
        return {"evidence": state["evidence"] + [result], "tools_used": state["tools_used"] + [name]}

    def report(self, state):
        case = state["case"]
        used = []
        evidence_lines = []
        for result in state["evidence"]:
            used.append(result.tool_name)
            evidence_lines.append(f"- [{result.tool_name}]: {result.summary}")
        evidence_text = "[Completed evaluations: " + ", ".join(used) + "]\n" + "\n".join(evidence_lines)

        prompt = clinical_prompts.REPORT_PROMPT.format(
            age=case.age, gender=case.gender, chief_complaint=case.chief_complaint, vitals=json.dumps(case.vitals),
            differential_str=differential_text(state["differential"]), evidence_str=evidence_text)
        data = self.ask(prompt)

        # Anything other than a clear "admit" or "discharge" goes to a human: the agent must
        # never silently send a patient home or admit them because a reply was unclear.
        disposition_text = str(data.get("disposition", "")).upper()
        if "DISCHARGE" in disposition_text:
            disposition = DispositionType.DISCHARGE_WITH_FOLLOWUP
        elif "ADMIT" in disposition_text:
            disposition = DispositionType.ADMIT_INPATIENT
        else:
            disposition = DispositionType.ESCALATE_TO_PHYSICIAN

        final_differential = hypotheses_from(data.get("differential_diagnosis"))
        if not final_differential:
            final_differential = state["differential"]

        all_findings = []
        for result in state["evidence"]:
            all_findings += result.findings

        recommendations = data.get("recommendations")
        if not isinstance(recommendations, list):
            recommendations = []

        report = ClinicalReport(
            case_id=case.patient_id,
            patient_id=case.patient_id,
            primary_diagnosis=data.get("primary_diagnosis") or "Undetermined",
            disposition=disposition,
            differential_diagnosis=final_differential,
            key_findings=all_findings,
            clinical_impression=data.get("clinical_impression") or "",
            recommendations=recommendations,
            evidence_trail=state["evidence"],
        )
        self.tell("report", report)
        return {"report": report}

    # ------------------------------------------------------------------------------------
    # Wiring the steps together
    # ------------------------------------------------------------------------------------
    def build_graph(self):
        graph = StateGraph(WorkupState)
        graph.add_node("intake", self.intake)
        graph.add_node("choose", self.choose)
        graph.add_node("run_tool", self.run_tool)
        graph.add_node("report", self.report)

        graph.set_entry_point("intake")
        graph.add_edge("intake", "choose")
        graph.add_conditional_edges("choose", next_step, {"run_tool": "run_tool", "report": "report"})
        graph.add_edge("run_tool", "choose")
        graph.add_edge("report", END)
        return graph.compile()

    def diagnose(self, case, max_turns=4):
        start = {"case": case, "differential": [], "evidence": [], "tools_used": [], "next_tool": "",
                 "reason": "", "query": "", "turns": 0, "max_turns": max_turns, "report": None}
        final_state = self.build_graph().invoke(start)
        return final_state["report"]


def next_step(state):
    """After "choose": run the chosen tool, or write the report if it said STOP."""
    if state["next_tool"] == "STOP":
        return "report"
    return "run_tool"
