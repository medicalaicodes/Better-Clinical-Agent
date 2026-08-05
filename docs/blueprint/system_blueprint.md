# HERALD — A Capability-Centric Blueprint for a Clinically-Oriented Agentic AI System

**Hypothesis-driven, Evidence-graded, Reliability-Aware, Layered Diagnostic agent**

*Technical design document · Internship implementation plan · Research architecture proposal*
*IIT Mandi — Healthcare AI Internship · July 2026*

---

# Document status and how to read it

This is a design document, not a survey and not a literature review. It assumes the reader has
already seen the landscape material (the 27-paper Category 1.1 gap analysis, the ERC coordination
survey, and the comparison table) and wants to know **what to build, in what order, and how to
know whether it worked**.

Three things are deliberately kept separate throughout, because conflating them is the most common
way a project like this fails:

| Track | Owner | Timeline | Success criterion |
|---|---|---|---|
| **A — MVP build** | The intern (you) | Weeks 1–8 | A running system, reproducible numbers, a demo |
| **B — Research extension** | You, after A works | Weeks 8–16 | One defensible novel claim with an ablation |
| **C — Governance add-on** | A second teammate, later | After A is stable | A policy layer attached without touching core code |

Nothing in Track B or C is allowed to block Track A. Section 4 marks every layer as **v1 mandatory**
or **v2 optional** for exactly this reason.

A note on citation style: claims sourced from the literature are attributed inline with enough
identifying information (name, venue, arXiv ID) to be found. Numbers quoted from papers are the
authors' reported numbers and have not been independently reproduced; several of them should be
treated as *targets to verify*, not as facts.

---

# 1. Executive Objective

The purpose of this blueprint is to specify a healthcare agentic system that can be **built by one
person in a semester**, produces **numbers that survive scrutiny**, and can later be extended in two
independent directions — a research contribution and a governance layer — without being rewritten.

The design is organised around a single claim: *the interesting engineering in a clinical agent is
not the reasoning, it is the epistemics.* A modern vision-language model plus five off-the-shelf
chest X-ray tools will already produce fluent, plausible, mostly-correct clinical text. What it will
not do is tell you which parts of that text are load-bearing, which tool it should not have trusted,
what it did not look at, or when it should have stopped. Every documented failure of the current
generation of medical agents is a failure of that second category, not the first. MedAgent-Pro loses
16.5 F1 when one of its indicator tools degrades. CARE's coordinator rewrites correct tool outputs
into incorrect ones. MedAgent-Pro and Cerebra handle missing modalities by skipping or fusing over remaining inputs silently, failing to disclose unresolved hypotheses or discount confidence. Meissa answers confidently at 57% MedQA with no abstention path. On MedFlowBench, requiring that answers be supported by checkable evidence drops average model performance from 0.53 to 0.32 (and on the two hardest task families from 0.37→0.18 and 0.55→0.14). No system in this set has regulatory clearance or reported prospective in-workflow deployment. These are not capability gaps; they are missing control structures.

**Why implementation-first.** The blueprint is written so that a working, evaluable system exists at
the end of Phase 2 — before any research question is settled. This ordering is not modesty, it is
risk management. The research gaps in this field are gaps about *system behaviour under stress*
(does the router pick the right specialist? does a corrupted tool propagate? does the system degrade
or halt?). Those questions are unanswerable without a system to stress. A student who spends eight
weeks refining a problem statement and two weeks building will have neither a result nor a demo. A
student who builds first will find that the problem statement writes itself out of the failure
logs — which is exactly how the strongest papers in this area were produced.

**Why this is not a paper-by-paper collage.** The obvious way to write this document would be to
take the 27 analysed systems, extract one component from each, and staple them together: MDAgents'
complexity router, MMedAgent's tool bus, MedAgent-Pro's indicator agents, CoMMa's Shapley credit,
AURA's counterfactual check, MemSeg's memory tokens. That produces an architecture diagram with
eleven boxes and no thesis, and it inherits every one of those systems' failure modes simultaneously.
The approach here is inverted. We start from a list of **capabilities a clinically meaningful agent
must have** — decide what the task is, decide whether tools are needed, decide *which*, decide
whether to believe the answer, fuse across sources, decide when not to answer, and leave a trail —
and we specify the minimum machinery that delivers each capability. Published systems then appear in
this document in one of three roles: as a component we adopt (TorchXRayVision, MedRAX's tool set), as
a baseline we must beat (MedRAX on ChestAgentBench), or as a documented failure we design against
(Cerebra's/MedAgent-Pro's silent non-disclosure under missingness). No system is copied wholesale.

---

# 2. Problem Framing

# 2.1 What "Agentic AI in Healthcare" means in this blueprint

> **Operating definition.** A clinical agent is a *tool-orchestrated reasoning system that maintains
> an explicit, typed, auditable model of what it currently believes and why, updates that model by
> deliberately acquiring evidence, and is capable of declining to produce an answer.*

Three clauses in that sentence do real work.

*"Explicit, typed, auditable model of what it believes"* — the agent's clinical state does **not**
live in a chat transcript. It lives in a structured **Evidence Ledger** and a **Hypothesis Ledger**.
Free text is an output format, never an internal representation. This is the single most consequential
decision in the blueprint and everything else follows from it.

*"Deliberately acquiring evidence"* — the agent chooses which tool to call based on what would most
change its current beliefs, not based on a fixed script. This is where the differential-diagnosis
paradigm enters (Section 2.4) and it is the capability the AMIE authors explicitly flagged as
untested and hard: *"the goal-directed acquisition of information under uncertainty."*

*"Capable of declining"* — abstention is a first-class, typed output with the same status as an
answer, and it is *specific*: not "I'm not sure", but "I cannot separate hypotheses A and B; the
lateral view would separate them; it is not in this study."

# 2.2 What it is not

| Not this | Why the distinction matters here |
|---|---|
| **A plain medical VLM** (MedGemma, LLaVA-Med, CheXagent) | A VLM maps input to output in one shot. It has no state, no notion of an unused source, no way to say "I did not examine the costophrenic angles." The Illusion of Readiness stress tests (Gu et al., arXiv 2509.18234) showed frontier models answering medical questions **correctly with the image removed** — a VLM cannot detect that it did this; an agent with an evidence ledger structurally cannot do it, because the ledger would be empty. |
| **A static multimodal classifier / supervised fusion model** | These are often *better* than agents at the narrow prediction task, and they must be kept in the evaluation as the non-agent ceiling (MedAgentBoard found multi-agent systems trailing conventional methods on VQA and EHR prediction). What they cannot do is handle an open task space, explain which input drove the call, request a missing input, or be re-pointed at a new question without retraining. |
| **A simple tool-calling wrapper (ReAct + N tools)** | This is MedRAX, and it is a legitimate and strong baseline (63.1% ChestAgentBench, ICML 2025). It is not sufficient because the LLM both selects the tool and interprets its output, with no independent estimate of whether the output deserves belief. MedCTA (arXiv 2606.11702, June 2026) benchmarked 18 models on exactly this pattern and reported a **large controller gap against oracle routing**, with failures concentrated in *under-calling* and *weak local grounding* rather than in final synthesis. |
| **A multi-agent debate system without evidence control** | The empirical record here is poor and the user's own survey documents it: MAM's debate overrode correct answers (70.2% → 64.0%); CoMMa gained +0.5% AUC going from one agent to five; MDAgents underperformed solo GPT-4 on MedQA (72.9% vs 75.1%). MAST (Cemri et al., NeurIPS 2025, 1,600+ annotated traces) found ~42% of multi-agent failures are specification issues and ~37% inter-agent misalignment, and concluded that better base models will not fix the taxonomy. HERALD therefore uses **one reasoning process with many tools**, not many reasoning processes. Disagreement is represented as data in a ledger, not enacted as a conversation. |

# 2.3 Operating assumption

**This is a clinician-supporting, tool-orchestrated reasoning system for research evaluation. It is
not a diagnostic device, it is not deployed, and no claim of clinical readiness is made anywhere in
this document.** The PLOS Medicine methodological piece on benchmarking medical AI agents (July 2026)
states the near-term reality plainly: agents will act in clinician-supporting rather than
clinician-replacing roles. The blueprint is designed for that role. Concretely, the system's
terminal outputs are *a structured findings list, a ranked differential with evidence links, and a
disposition* — the artefacts a resident hands to an attending, not a decision.

# 2.4 Is the differential-diagnosis paradigm actually needed? (Yes — and here is the argument)

The brief asked whether the DDx paradigm should be designed in. It should, and not for realism
reasons — for architectural reasons. Four independent arguments:

**(a) It is the only representation in which abstention is well-defined.** A single-label system can
only abstain by thresholding a confidence number, which is exactly the signal we know to be
untrustworthy ("Entropy Alone is Insufficient for Safe Selective Prediction", arXiv 2603.21172; and
the Robust-TO finding that frontier models lose 15–30 points under corruption *while self-reported
confidence barely moves*). A ranked hypothesis set gives a **shape** to abstain on: flat posterior
mass across clinically divergent hypotheses is a principled, inspectable abstention trigger that does
not depend on any single model's calibration.

**(b) It makes tool selection a well-posed optimisation.** Without hypotheses, "which tool next?" is
a heuristic. With hypotheses, it is *which observation most reduces uncertainty over the current
hypothesis set* — a computable expected-information-gain quantity. This converts the field's hardest
open routing problem into something with an objective function.

**(c) It makes missingness expressible instead of fatal.** In a single-label system, an absent
modality is a hole. In a hypothesis ledger it is an annotation: *"pulmonary embolism — not excluded;
requires CTPA; unavailable."* The hypothesis stays alive and visible. This is precisely the
disclosure capability existing systems (Cerebra, MedAgent-Pro) lack.

**(d) The empirical payoff is documented.** AMIE reached top-10 DDx accuracy of 59.1% against 33.6%
for unassisted clinicians on 302 NEJM CPC cases (*Nature*, 2025); a later lean-orchestration system
reported 46.0% top-1 / 79.1% top-10 on the same 302 cases. More usefully for us, that same work
introduced **Capture@K** — is any differential in the top-K sufficient to make a clinician order the
correct diagnostic test of record? — which measures the differential by *the action it licenses*
rather than by string match. And an ablation study in *npj Digital Medicine* (2025) found that
supplying lab results improves LLM DDx accuracy **by up to 30%**, which is direct evidence that
*evidence acquisition* — not reasoning capacity — is the binding constraint.

**Scope discipline.** Adopting the DDx paradigm does **not** mean building an open-world diagnostic
engine. In v1 the hypothesis space is a **closed, curated set** (the 14–20 CXR findings/conditions
covered by the available tools, plus an explicit `OTHER / OUT-OF-SCOPE` hypothesis that can win).
Closing the hypothesis space is what makes the posterior meaningful and the evaluation possible. The
CXReasonAgent authors (KAIST, arXiv 2602.23276) made the analogous choice — restricting to 12
diagnostic tasks whose image-derived evidence can be *reliably extracted by the tools they have* —
and it is the right instinct.

---

# 3. Blueprint Principles

Ten principles. For each: what it means, why it matters (with the specific failure it prevents), and
what it forces you to do differently in code.

---

# P1 — Evidence before narrative

**Statement.** No clinical claim may appear in an output that is not backed by an Evidence Packet ID
in the ledger. The narrative generator is a *renderer*, not a *reasoner*, and it runs last.

**Why.** This is the direct countermeasure to the most-documented failure in the field. CARE's GPT-4o
coordinator "over-edits" — it takes correct VQA tool outputs and rewrites them into incorrect
answers. The Illusion of Readiness study found frontier models producing "fluent, structured
rationales [that] frequently include non-existent image features"; their conclusion — *fluency in
explanation does not equate to valid inference* — is the sentence this principle exists to enforce.

**Engineering consequence.** The generator's prompt contains only the ledger, and its output is
passed through an **entailment gate**: every generated sentence must be entailed by the packets it
cites (RadFact-style, Bannur et al. 2024). Sentences that fail are dropped, not repaired. This costs
you one extra model call per case and buys you a measurable *unsupported-claim rate*.

---

# P2 — Reliability is a property of evidence, not of the model

**Statement.** Every Evidence Packet carries a calibrated reliability score computed by machinery
*outside* the tool that produced it. A tool's own softmax confidence is an input to that computation,
never the output of it.

**Why.** "The Confidence Dichotomy" (ACL 2026, arXiv 2601.07264) established that tool use changes
agent calibration in *tool-type-dependent* ways: evidence-retrieval tools systematically induce
overconfidence because of noise in what they return, while verification tools with deterministic
feedback improve calibration. Robust-TO (arXiv 2606.26904) names the general disease the **Blind
Trust Problem**: every perception output treated as equally reliable, and the system's confidence
decoupled from the conditions that produced it. In medicine the cost is measured: MedAgent-Pro's
F1 drops 16.5 points when a single weak indicator tool is added to the pipeline.

**Engineering consequence.** A `ReliabilityEstimator` module that no tool can bypass, and a rule that
the Fusion layer reads only `packet.reliability`, never `packet.raw_confidence`. It also means you
must deliberately pair *evidence tools* (classifiers — overconfidence-inducing) with *verification
tools* (deterministic geometric measurements — calibration-improving) wherever you can.

---

# P3 — Graceful degradation, by disclosure rather than imputation

**Statement.** Absence of an input is recorded as an explicit `MissingEvidence` packet. The system
must never halt for missingness, and must never silently synthesise a substitute.

**Why.** Existing systems handle missingness by skipping affected steps (MedAgent-Pro) or fusing over remaining data (Cerebra). Neither *discloses* the consequence: which hypotheses were left unresolved, whether the missing item is obtainable, and whether confidence should be discounted. Generative imputation (Meta-Modal Agent, arXiv 2605.25007) carries a severe risk under heavy missingness: reconstruction "may look plausible while adding unsupported evidence to the pipeline." The workable middle is demonstrated in supervised fusion (arXiv 2511.15847, ICU mortality): mask the missing branch out of the gate, renormalise over observed experts, fall back to a calibrated single-specialist prediction. Reported cost: AUPRC drops 0.09–0.11 while **calibration metrics stay near branch-level values**. That is the target behaviour — *lose accuracy, keep honesty by explicit disclosure*.

**Engineering consequence.** Missingness is a value in the schema, not a null. The Fusion layer
renormalises over available evidence and records the renormalisation. The Disposition layer is
allowed to convert missingness into a `REQUEST` for the specific absent item.

---

# P4 — Abstention is an action, not a refusal

**Statement.** The system's disposition is one of four typed values: `ANSWER`,
`ANSWER_WITH_CAVEAT`, `REQUEST(item)`, `ESCALATE(reason_code)`. There is no "I don't know."

**Why.** Generic refusal is clinically useless and evaluationally invisible. The literature on
medical abstention is almost entirely about single models thresholding themselves ("Knowing When to
Abstain", arXiv 2601.12471; the conformal safety layer for medical summarisation, arXiv 2606.08969),
which leaves the interesting case — *the system knows what it is missing* — unaddressed. A typed
`REQUEST` is measurable: supply the requested item counterfactually and check whether the answer
actually changed. That gives you **request specificity**, a metric nobody currently reports.

**Engineering consequence.** A `Disposition` object, a reason-code enumeration, and an evaluation
harness that can replay a case with additional evidence injected.

---

# P5 — Conflict is preserved, not averaged

**Statement.** When independent sources disagree, the disagreement is stored and surfaced. The system
does not compute a mean and discard the spread.

**Why.** The user's ERC survey already identified this as the empty cell in the literature ("no
standard metric exists yet, which is itself a contribution opportunity"). The mechanism behind the
danger is documented: in multi-agent debate, inter-agent correlation converges to 1 and terminal
confidence saturates *independently of error*, with 17× smaller variance than accuracy ("Diversity
Collapse and Manufactured Overconfidence"). Averaging away discordance is how a system manufactures
the confidence it has not earned.

**Engineering consequence.** A `ConflictDetector` between Evidence and Fusion, a `conflicts[]` field
on the case state, and a Disposition rule that unresolved high-severity conflict forces
`ESCALATE` regardless of posterior sharpness.

---

# P6 — Modularity with a stable contract, not modularity by folder structure

**Statement.** Every module communicates only through the versioned Pydantic schemas in Section 5.
Tools are registered, not imported. Any tool can be removed at runtime and the system must still
produce a disposition.

**Why.** MAST attributes roughly 42% of multi-agent failures to specification issues — ambiguous
prompts, role mis-specification, inadequate contracts. The practitioner consensus is blunt about the
fix: treat specifications like API contracts, JSON schemas for everything. This is also what makes
Track C possible: a governance layer can only intercept what has a schema.

**Engineering consequence.** Pydantic v2 models with `schema_version`, a `ToolRegistry` with
capability cards, and a `test_tool_removal` suite that runs the whole pipeline with each tool
disabled in turn.

---

# P7 — Backbone-agnostic

**Statement.** The reasoning LLM is a configuration value. Swapping MedGemma-4B for Qwen2.5-VL-7B or
a hosted model requires changing one line and re-running evaluation.

**Why.** Two reasons, one scientific and one legal. Scientific: HeartAgent (arXiv 2603.10764) reported
top-3 DDx of 0.556 / 0.556 / 0.567 on the NEJM subset across Llama-3.3-70B, Qwen-2.5-32B and
MedGemma-27B respectively — the *scaffolding*, not the backbone, produced the ~29–36% improvement.
If your contribution is the scaffolding, you must be able to demonstrate it across backbones.
Legal: see P8.

**Engineering consequence.** A thin `LLMClient` interface over an OpenAI-compatible endpoint
(vLLM serves MedGemma, Qwen, and Llama identically). No framework-specific model objects leak into
module code.

---

# P8 — Local-first, because the data licence says so

**Statement.** The default deployment runs entirely on local hardware with open-weight models. Hosted
frontier models are an *optional comparison arm*, used only on data whose licence permits it.

**Why.** This is not a preference. The PhysioNet Credentialed Health Data Use Agreement (v1.5.0,
governing MIMIC-CXR and MIMIC-IV) **explicitly prohibits sending the data to third-party LLM APIs or
online platforms**, and states that MIMIC data must not be retained by third-party LLM services. Any
architecture whose reasoning core is `openai.chat.completions` is non-compliant on MIMIC by
construction. Separately, as of mid-2026 PhysioNet warns that credentialed-access applications face
**significant delays** — so the MVP's critical path must not depend on that approval arriving.

**Engineering consequence.** MedGemma-4B (or Qwen2.5-VL-7B) served by vLLM on a single 16–24 GB GPU
is the default. The MVP's primary datasets are the openly downloadable ones (VinDr-CXR, PadChest,
NIH ChestX-ray14, CXR-LT 2026 splits, Synthea). MIMIC is a *nice-to-have upgrade path*, gated behind
a local-only inference configuration.

---

# P9 — Auditability by design, using a standard others already speak

**Statement.** Every LLM call, tool call, packet, reliability score, conflict, and disposition emits
an OpenTelemetry GenAI span at the moment it happens. Nothing is reconstructed after the fact.

**Why.** Two forcing functions. Regulatory: EU AI Act Article 12 requires high-risk AI systems to
"technically allow for the automatic recording of events (logs) over the lifetime of the system,"
with Articles 19/26(6) imposing at least six months' retention — and diagnosis and clinical decision
support sit squarely in the high-risk bracket. (The Digital Omnibus on AI, given final Council
approval on 29 June 2026, defers Annex III high-risk obligations to **2 December 2027** and Annex I
to 2 August 2028; Article 50 transparency obligations were *not* deferred and apply from 2 August
2026. The deferral buys time; it does not change the schema you need.) Practical: the OTel GenAI
semantic conventions already define `invoke_agent`, `chat` and `execute_tool` spans, and — crucially
— they put message content in **span events rather than indexed attributes**, precisely so that
sensitive content can be filtered or dropped at the Collector without touching application code.
That is a PHI-redaction hook you get for free if you adopt the convention on day one, and an
expensive retrofit if you do not. (Caveat to record honestly: the GenAI conventions were still at
Development stability as of semconv v1.41 in mid-2026, so attribute names may move.)

**Engineering consequence.** OTel SDK wired in Phase 1, not Phase 6. Langfuse or Grafana Tempo
self-hosted as the backend. A `trace_id` on every artefact the system produces.

---

# P10 — Human override compatibility and governance-ready seams

**Statement.** The pipeline exposes named, no-op-by-default **interception points** before every
irreversible or externally-visible action. In v1 they are pass-through functions with logging. In the
governance track they become policy enforcement.

**Why.** EU AI Act Article 14 requires human oversight including the ability to intervene in or
interrupt operation. LangGraph gives durable checkpointing and native human-in-the-loop interrupts,
so the cost of building the seams in v1 is close to zero — but only if you build them in v1. Adding
an interception point to a pipeline that assumed synchronous completion is a rewrite.

**Engineering consequence.** Five named hooks — `pre_ingest`, `pre_tool_call`, `pre_fusion`,
`pre_response`, `pre_persist` — each with a documented signature, each a `lambda x: x` in v1.

---

# Principle interaction summary

| Principle | Prevents (documented failure) | v1 cost | Blocks if deferred? |
|---|---|---|---|
| P1 Evidence before narrative | CARE coordinator over-editing; fabricated rationales | Medium | **Yes** — retrofit is a rewrite |
| P2 Reliability on evidence | MedAgent-Pro −16.5 F1; Blind Trust Problem | Medium | **Yes** |
| P3 Disclose-and-degrade | Cerebra halt; fabricated imputation | Low | No |
| P4 Typed abstention | Meissa silent failure at 57% MedQA | Low | No |
| P5 Conflict preservation | Manufactured overconfidence in debate | Low | No |
| P6 Schema contracts | MAST specification failures (~42%) | Low | **Yes** |
| P7 Backbone-agnostic | Result attributable to model, not method | Low | No |
| P8 Local-first | PhysioNet DUA violation | Low | **Yes** (legal) |
| P9 OTel audit spine | EU AI Act Art. 12; unreproducible traces | Low | **Yes** — retrofit is a rewrite |
| P10 Interception seams | Governance layer requires core rewrite | Very low | **Yes** |

Read the right-hand column as the actual build discipline: five things must be right in Phase 1 even
though nothing depends on them yet.

---

# 4. High-Level Architecture

# 4.1 The control loop

HERALD runs a bounded loop, not an open-ended ReAct scratchpad:

```
HYPOTHESISE  →  ACQUIRE  →  VERIFY  →  ARBITRATE  →  DISPOSE
     ↑                                        
      (bounded: ≤ N rounds) 
```

The loop terminates on one of four conditions, all of which are logged: the hypothesis posterior is
sufficiently sharp; no remaining tool has positive expected information gain; the round budget is
exhausted; or a hard conflict / integrity failure forces escalation. Explicit termination conditions
are a direct response to MAST failure mode 1.5 (*unaware of termination conditions*) and to the
practitioner rule of thumb that agent loops need a tight consecutive-reply cap by default.

# 4.2 Text pipeline diagram

```
                          
                            L11  GOVERNANCE INTERFACE (v2)                        
                            policy engine · RBAC · redaction · consent · review   
                          
                              pre_   pre_tool_  pre_fusion pre_       pre_
                              ingest call                  response   persist
                                                                      
           
   CLINICAL     L1  INGESTION       L2  TRIAGE &         L3  TOOL REGISTRY  
    INPUT     & NORMALISATION  TASK FRAMING      (capability cards)
   img/text      Modality Det.       Query Parser         competence priors 
   /labs         Quality checks      Task type          
       De-ID stub          Hypothesis seed              
                                 
                                                                      
                                                                      
                 
                               L0  CASE STATE MANAGER  (typed, versioned)      
                         
                    EVIDENCE LEDGER   HYPOTHESIS LEDGER   CONFLICT LOG   
                         
                 
                                                                     
                                              
                                                
                     L4  TOOL ROUTER /             
                                      ACQUISITION                  
                                      PLANNER (EIG)                
                                                
                                               selects tool + args  
                                                                    
                
            L5  EVIDENCE EXTRACTION (Specialist Tool Wrappers)      
                      
             classifier segmenter  geometric  retrieval     
              (xN dom.)  lung/hrt   measures   KB/RAG       
                      
                   each returns a raw ToolResult                    
                
                                                                     
                
            L6  VERIFICATION & RELIABILITY                          
              EvidencePacketBuilder → ReliabilityEstimator          
              competence · agreement · integrity → r ∈ [0,1]        
              ConflictDetector                                      
                
         writes EvidencePackets 
                                     
                     
                      L7  FUSION / ARBITRATION           
                       reliability-weighted hypothesis   
                       update · renormalise over observed
                       · record conflicts & missingness  
                     
                                       posterior + conflicts + coverage
                                      
                     
                      L8  DISPOSITION (Abstention Engine)
                       ANSWER  CAVEAT  REQUEST  ESCALATE
                       conformal risk control threshold  
                     
                                      
                     
                      L9  RESPONSE COMPOSITION           
                       grounded generator (cites pkt IDs)
                       → ENTAILMENT GATE → final output  
                     
                                      
   
    L10  PROVENANCE & AUDIT SPINE  (cross-cutting, writes at every step)     
     OTel GenAI spans: invoke_agent / chat / execute_tool  ·  case bundle     
     content in span EVENTS (droppable) · manifest hash · replayable trace    
   
```

# 4.3 Layer specifications

# L0 — Case State Manager *(cross-cutting; v1 mandatory)*
**In:** every module's writes. **Out:** the current `CaseState` to every module.
The single source of truth. Holds `EvidenceLedger`, `HypothesisLedger`, `ConflictLog`,
`AcquisitionHistory`, `Disposition`, and `trace_id`. Append-only within a case: packets are never
mutated, only superseded (with a `supersedes` pointer). This is what makes replay possible and what
makes the audit trail truthful rather than reconstructed.

# L1 — Input Ingestion & Normalisation *(v1 mandatory)*
**In:** raw study — images (DICOM/PNG/JPG), free text (indication, prior report), structured values
(labs, vitals, demographics). **Out:** `NormalisedCase` + `AvailabilityMap`.
Responsibilities: modality/view detection; technical quality assessment (rotation, inspiration,
projection, inclusion — the CXReasonAgent image-quality task set is the right list); unit and code
normalisation; a de-identification stub (no-op in v1 on public data, a real hook for later).
**The AvailabilityMap is the important output**: an explicit record of what modalities and fields
exist *and which expected ones do not*. Missingness enters the system here or it never does.

# L2 — Task Understanding / Triage *(v1 mandatory)*
**In:** `NormalisedCase` + user query. **Out:** `TaskFrame` = {task_type, target_findings,
seed_hypotheses, urgency_flag, out_of_scope_flag}.
Classifies the request into a small closed set (`SCREEN_FINDINGS`, `ANSWER_QUESTION`,
`DIFFERENTIAL`, `COMPARE_PRIOR`, `SUMMARISE`) and seeds the Hypothesis Ledger with priors.
Also does **scope gating**: if the request falls outside the tool-covered hypothesis space, that is
determined here and produces an immediate `ESCALATE(OUT_OF_SCOPE)`. Deliberately *not* a
complexity-based routing classifier of the MDAgents kind — the user's own gap analysis found that
its complexity classifier correlated poorly with physician ratings and routed tasks incorrectly.

# L3 — Tool Registry *(v1 mandatory)*
**In:** tool registrations at startup. **Out:** candidate tool set for a given `TaskFrame`.
A declarative catalogue of **Capability Cards** (schema in §5.4). Each card states what the tool
produces, what inputs it requires, its training provenance, and its **measured competence profile**
sliced by case descriptor (view, source domain, demographic bucket, OOD score). Registration is
config, not code. In v2 the registry is served over MCP so tools become process-isolated.

# L4 — Tool Router / Acquisition Planner *(v1 mandatory)*
**In:** `TaskFrame`, `HypothesisLedger`, `AvailabilityMap`, registry candidates.
**Out:** `AcquisitionAction` = {tool_id, arguments, expected_information_gain, rationale}.
This is the layer MedCTA identified as the bottleneck. v1 uses a **hybrid**: hard preconditions from
the capability cards eliminate ineligible tools (an ineligible tool is never a "choice"), then a
scoring function ranks the survivors by `competence_prior × expected_information_gain / cost`. The
LLM proposes and may override with a logged rationale; it does not select unconstrained. Explicitly
tracks **under-calling** — if a hypothesis has no supporting or refuting evidence and an eligible
tool exists, the planner is required to call it before the loop may terminate.

# L5 — Evidence Extraction / Specialist Tool Wrappers *(v1 mandatory)*
**In:** `AcquisitionAction` + case data. **Out:** raw `ToolResult` + execution telemetry.
Uniform wrapper contract around heterogeneous models: timeout, retry policy, input-precondition
assertion, deterministic seeding, and *mandatory* capture of whatever native uncertainty signal the
tool offers (logits, ensemble spread, mask area variance, retrieval score). Wrappers never
interpret; they transport.

# L6 — Evidence Verification & Reliability Scoring *(v1 mandatory — this is the differentiator)*
**In:** `ToolResult` + `NormalisedCase` + registry card. **Out:** `EvidencePacket` with calibrated
`reliability`, plus `Conflict` records.
Three sub-steps: **build** (normalise to the packet schema, attach provenance), **score** (compute
the reliability triple — §5.8), **cross-check** (compare against existing packets; emit conflicts).
Reliability is calibrated on a held-out split so that `r` means something on a probability scale.
This layer exists because of a specific empirical result: error injection propagates to a wrong final
answer with probability ≈0.62 in tool-using agents (AgentProp-Bench, arXiv 2604.16706), and
*detection* and *recovery* are statistically independent capabilities. L6 is detection; L4/L7 are
recovery.

# L7 — Fusion / Arbitration *(v1 mandatory)*
**In:** Evidence Ledger + Hypothesis Ledger + Conflict Log. **Out:** updated posterior + `coverage`
+ `resolved/unresolved conflicts`.
A reliability-weighted log-odds update over the closed hypothesis set, renormalised over *observed*
evidence only (the masking-and-renormalising pattern from the ICU fusion work). Deliberately a
transparent, inspectable arithmetic step rather than an LLM judgement call — you cannot audit a
narrative fusion. Records **coverage**: which hypotheses have been actively examined vs. merely not
raised. Unexamined ≠ excluded.

# L8 — Abstention / Escalation (Disposition Engine) *(v1 mandatory)*
**In:** posterior, conflicts, coverage, AvailabilityMap, cost budget. **Out:** `Disposition`.
Decision order (first match wins, all logged):
1. integrity failure (unusable image, precondition violation) → `ESCALATE(TECHNICAL)`
2. unresolved high-severity conflict → `ESCALATE(CONFLICT)`
3. top hypotheses separable by a named, obtainable observation → `REQUEST(item)`
4. selective-risk threshold not met at target error rate → `ESCALATE(UNCERTAIN)`
5. threshold met but coverage incomplete → `ANSWER_WITH_CAVEAT`
6. otherwise → `ANSWER`
The threshold in (4) is set by **conformal risk control / Learn-then-Test** on a calibration split
(Angelopoulos et al.), which gives a distribution-free guarantee on the error rate among *accepted*
cases under exchangeability. Being able to state that guarantee, and to show it breaking under
distribution shift, is worth more than any accuracy number in this project.

# L9 — Response Generation *(v1 mandatory)*
**In:** ledgers + disposition. **Out:** `ClinicalOutput` = structured findings + ranked differential
with per-hypothesis evidence links + disposition + prose summary.
The generator sees only the ledger. Its output passes the **entailment gate**; unsupported sentences
are dropped and counted. Structured output is primary; prose is a rendering of it.

# L10 — Provenance & Logging *(cross-cutting; v1 mandatory)*
Writes at every step. OTel GenAI spans (`invoke_agent` root, `chat` per LLM call, `execute_tool` per
tool) with content in span **events** rather than indexed attributes so it can be dropped at the
Collector. Plus a per-case **Case Bundle**: input manifest hash, model + weight versions, tool
versions, full ledger, disposition, wall-clock, token and GPU accounting. Replay test: re-executing
a bundle must reproduce the disposition.

# L11 — Governance Interface *(v2 optional; seams mandatory in v1)*
Five no-op hooks in v1. In v2 they host: consent/authorisation checks, RBAC, PHI redaction, policy
evaluation, output filtering, human-review queue. Detailed in §10.

# 4.4 Layer summary table

| # | Layer | Input | Output | v1? | Primary failure it addresses |
|---|---|---|---|---|---|
| L0 | Case State Manager | all module writes | `CaseState` | **v1** | Loss of conversation history (MAST 1.4); unreproducible traces |
| L1 | Ingestion & Normalisation | raw study | `NormalisedCase`, `AvailabilityMap` | **v1** | Missing-modality invisibility; garbage-in |
| L2 | Triage / Task Framing | case + query | `TaskFrame`, seed hypotheses | **v1** | Task mis-specification (MAST 1.1); out-of-scope answering |
| L3 | Tool Registry | registrations | Capability Cards, competence priors | **v1** | Untracked tool provenance; unmeasurable routing |
| L4 | Tool Router / Planner | frame + hypotheses | `AcquisitionAction` | **v1** | Behaviour-routing deficiency; under-calling (MedCTA) |
| L5 | Specialist Tool Wrappers | action + data | raw `ToolResult` | **v1** | Silent tool failure; non-determinism |
| L6 | Verification & Reliability | `ToolResult` | `EvidencePacket(reliability)`, `Conflict` | **v1** | Blind Trust Problem; error cascade (p≈0.62) |
| L7 | Fusion / Arbitration | ledgers | posterior, coverage | **v1** | Averaging away conflict; halt on missingness |
| L8 | Disposition Engine | posterior + conflicts | `Disposition` | **v1** | Silent failure; no abstention (Meissa) |
| L9 | Response Composition | ledgers + disposition | `ClinicalOutput` | **v1** | Coordinator over-editing (CARE); fabricated rationale |
| L10 | Provenance & Audit Spine | everything | OTel spans + Case Bundle | **v1** | EU AI Act Art. 12; irreproducibility |
| L11 | Governance Interface | intercepted objects | filtered/blocked/queued | *v2* (seams v1) | Governance retrofit cost |

# 4.5 Data flow, narrated

A chest radiograph and the indication text *"52M, acute dyspnoea, r/o pneumonia"* arrive.

**L1** detects a single frontal PA view, runs quality checks (adequate inspiration, mild rotation
flagged), finds no lateral view and no prior study, and writes an `AvailabilityMap` recording
`lateral: ABSENT`, `prior_study: ABSENT`, `labs: ABSENT`. **L2** frames the task as `DIFFERENTIAL`
with target `pneumonia`, seeds the Hypothesis Ledger with pneumonia, pulmonary oedema, pleural
effusion, pneumothorax, atelectasis, normal, and `OTHER`, and marks urgency. **L4** consults the
registry: five classifier variants are eligible, but the card for the RSNA-trained pneumonia model
records the highest competence prior for this exact indication and view; a lung/heart segmenter and
the geometric measurement tool are also eligible and cheap. It calls the segmenter and the
measurement tool first — they are deterministic *verification* tools whose outputs improve the
calibration of everything downstream (the Confidence Dichotomy result, exploited deliberately).

**L5** returns raw results. **L6** builds packets: the cardiothoracic ratio is 0.44 (deterministic,
integrity-verified, reliability 0.93); the pneumonia classifier returns 0.71 (softmax) but the case
descriptor puts it slightly outside its training domain and test-time augmentation gives a spread of
±0.19, so its calibrated reliability is 0.58, not 0.71. A second classifier trained on a different
source hospital returns 0.34 for the same finding. **L6** emits a `Conflict(severity=MEDIUM)`.

**L7** updates the posterior: pneumonia 0.41, pleural effusion 0.19, oedema 0.14, others lower,
renormalised over observed evidence, with `coverage.pneumothorax = NOT_EXAMINED`. Because two
independent classifiers disagree and no view exists that would separate consolidation from effusion,
**L8** does not reach the conformal acceptance threshold. It finds that the *lateral view* is the
named, obtainable observation with the highest expected information gain over the top two
hypotheses, and issues `REQUEST(lateral_view)` with `ESCALATE(CONFLICT)` as the fallback if
unavailable. **L9** renders: a findings list where every line cites a packet, a ranked differential
where pneumonia is annotated *"supported by classifier A (moderate reliability); not corroborated by
classifier B"*, an explicit statement that pneumothorax was not assessed, and the request. Every
step is already in the trace.

That case is the whole thesis of the blueprint in one paragraph. A conventional agent would have
returned "Findings consistent with pneumonia."

---

# 5. Core Modules

Sixteen modules: the fourteen requested, plus two additions that the differential-diagnosis
requirement makes unavoidable (§5.9 Hypothesis Manager, §5.10 Acquisition Planner — the latter is a
separable half of the Tool Router and deserves its own contract because it is where the strongest
research extension lives).

Schemas are given in Pydantic-flavoured pseudocode. Every model carries `schema_version: str` and
`trace_id: str`; these are omitted below for brevity but are not optional.

---

# 5.1 Query Parser

**Objective.** Turn a free-text clinical request plus study metadata into a closed-vocabulary
`TaskFrame`, and refuse to invent scope.

| | |
|---|---|
| **Input** | `{query_text: str, study_meta: StudyMeta, user_role: Role}` |
| **Output** | `TaskFrame{task_type: Enum, target_findings: list[ConceptID], seed_hypotheses: list[HypothesisSeed], urgency: Enum, out_of_scope: bool, ambiguity_flags: list[str]}` |
| **Strategy** | Constrained LLM extraction into a Pydantic schema (structured output / JSON mode), with a deterministic post-validator that rejects any concept not in the controlled vocabulary. Concepts are UMLS CUIs — PadChest's taxonomy is already UMLS-mapped, which removes a week of ontology work. |
| **MVP** | Five task types; vocabulary restricted to the ~20 findings the tool set covers; a rule-based urgency keyword list. |
| **Advanced** | Multi-turn clarification (ask before framing); paraphrase-invariance ensembling (parse the query three ways, flag disagreement) — motivated by the PSF-Med finding of a **14.6% answer-flip rate** for MedGemma-4B under paraphrase on binary MIMIC-CXR questions. |
| **Failure modes** | Silently widening scope to fit an unsupported request; over-eager urgency; hallucinated target findings; brittleness to phrasing. |
| **Metrics** | Frame accuracy vs. 200 hand-labelled queries; out-of-scope detection precision/recall; **paraphrase stability** (frame agreement across 3 paraphrases of the same query); vocabulary-violation rate (must be 0). |

---

# 5.2 Modality Detector

**Objective.** Determine what data is actually present, what view/modality each item is, whether it is
technically adequate, and — critically — **what expected data is absent**.

| | |
|---|---|
| **Input** | `{files: list[FileRef], structured: dict, text_blobs: list[str]}` |
| **Output** | `AvailabilityMap{present: list[ModalityRecord], absent_expected: list[ModalityGap], quality: list[QualityFinding]}` where `ModalityRecord{modality, view, resolution, checksum, quality_score}` and `ModalityGap{modality, why: Enum[NOT_ACQUIRED, NOT_PROVIDED, UNREADABLE], impact_note}` |
| **Strategy** | Deterministic first: DICOM headers, file magic, image statistics. Learned second: a small view/modality classifier (MedSigLIP zero-shot embedding similarity is sufficient and is already trained on CXR, histopathology, dermatology and fundus). Quality checks follow the CXReasonAgent image-quality task list — inspiration, rotation, projection, inclusion — implemented as rule-based geometric computations on a lung/heart segmentation mask. |
| **MVP** | CXR only: frontal/lateral discrimination, the four quality checks, and an `absent_expected` list derived from a per-task expectation template. |
| **Advanced** | Cross-modality expectation learned from the corpus ("for this indication, a lateral is present 78% of the time"); artefact detection (lines, tubes, pacemakers) as evidence rather than noise. |
| **Failure modes** | Classifying a rotated frontal as a lateral; declaring adequate quality on a clipped film; the `absent_expected` template being wrong, which makes the whole missingness story vacuous. |
| **Metrics** | View classification accuracy (VinDr/PadChest have view metadata); quality-check agreement with a small radiologist-adjudicated subset or with published CheXStruct-style criteria; **gap-detection recall** on synthetically ablated studies (this one you can generate at scale — hold out the lateral and check it is reported). |

---

# 5.3 Case State Manager

**Objective.** Own the single mutable object in the system and guarantee it is append-only,
serialisable, versioned, and replayable.

| | |
|---|---|
| **Input** | Typed write requests from every module. |
| **Output** | `CaseState{case_id, task_frame, availability, evidence: EvidenceLedger, hypotheses: HypothesisLedger, conflicts: list[Conflict], acquisitions: list[AcquisitionAction], disposition: Disposition | None, budget: BudgetState, trace_id}` |
| **Strategy** | LangGraph typed state (`TypedDict` + reducers) for the in-flight object; SQLite (or DuckDB) + JSONL Case Bundles for persistence. Packets are immutable; corrections create a new packet with `supersedes=<id>`. A `state_hash` after every write makes divergence detectable. |
| **MVP** | In-memory during the run, serialised to a JSONL bundle at the end. LangGraph checkpointer enabled from day one (it is one line and it is what makes human-in-the-loop interrupts possible later). |
| **Advanced** | Cross-encounter longitudinal state; state diffing for counterfactual replay ("what would the disposition have been without packet 7?"). |
| **Failure modes** | In-place mutation destroying the audit trail (the single worst bug you can write in this project); unbounded growth; state written by two modules with different assumptions. |
| **Metrics** | **Replay determinism rate** — fraction of bundles that re-execute to an identical disposition; ledger integrity assertions (no mutation, no orphan references, every cited packet exists); bundle size p95. |

---

# 5.4 Tool Registry

**Objective.** Make tools declarative, describable, and — above all — *measurable*, so that routing
and reliability have something to be computed from.

**Capability Card schema:**

```python
class CapabilityCard(BaseModel):
    tool_id: str
    version: str
    produces: list[ConceptID]            # what findings/measurements it can speak to
    output_kind: Enum[LABEL, SCORE, MASK, MEASUREMENT, TEXT, RETRIEVAL]
    requires: Preconditions              # modality, view, min resolution, quality floor
    provenance: Provenance               # training corpus, source institutions, year, licence
    cost: CostProfile                    # latency_ms_p50/p95, vram_mb, gpu_seconds
    uncertainty_signal: Enum[LOGIT, ENSEMBLE, TTA, NONE, DETERMINISTIC]
    tool_class: Enum[EVIDENCE, VERIFICATION]     # see P2 — this field is load-bearing
    competence: list[CompetenceSlice]    # measured, not declared
class CompetenceSlice(BaseModel):
    descriptor: CaseDescriptor           # {view, source_domain, age_band, sex, ood_bucket}
    n: int
    auroc: float; f1: float
    ece: float                           # calibration on this slice
    measured_on: str                     # dataset + split + date
```

| | |
|---|---|
| **Input** | YAML registrations at startup + a competence profile artefact produced by an offline profiling job. |
| **Output** | Eligible candidate set for a given `TaskFrame` and `AvailabilityMap`. |
| **Strategy** | The competence profile is **not** hand-written. Phase 3 includes a one-off profiling run: evaluate every registered tool on every held-out slice, store AUROC/F1/ECE per slice. TorchXRayVision makes this cheap because it ships several classifiers trained on *different* source hospitals (`nih`, `chex`, `mimic_ch`, `mimic_nb`, `pc`, `rsna`, `all`) plus dataset wrappers and built-in utilities for constructing covariate-shift splits. You get a pool of genuinely heterogeneous specialists with genuinely different per-domain competence, for free, without training anything. |
| **MVP** | 6–8 tools, YAML cards, competence sliced on 3 dimensions (source domain × view × OOD bucket). |
| **Advanced** | MCP-served tools (process isolation, language independence, and a natural authorisation boundary — RadFabric already demonstrates MCP as a CXR tool bus); dynamic registration; competence updated online from observed outcomes. |
| **Failure modes** | Declared-but-unmeasured competence (fatal — it makes routing a fiction); stale cards after a tool upgrade; preconditions too loose, so ineligible tools stay in the candidate set. |
| **Metrics** | Card completeness (% of required fields with measured, non-default values); precondition violation rate at runtime (target 0); card–reality drift (re-profile quarterly and measure the delta). |

---

# 5.5 Tool Router

**Objective.** Choose *which* eligible tool to invoke, given the current hypothesis state.
(The *whether-and-when-to-stop* question belongs to §5.10.)

| | |
|---|---|
| **Input** | `{task_frame, hypotheses, availability, candidates: list[CapabilityCard], history: list[AcquisitionAction], budget}` |
| **Output** | `AcquisitionAction{tool_id, arguments, expected_information_gain, competence_prior, cost, rationale, selection_mode: Enum[SCORED, LLM_OVERRIDE, FORCED_COVERAGE]}` |
| **Strategy** | Three-stage funnel. **(1) Hard filter** on preconditions from the capability card — an ineligible tool is never presented as an option, which removes the single largest class of routing error by construction. **(2) Score** survivors: `score = competence_prior(descriptor) × EIG(hypotheses) / normalised_cost`. **(3) LLM proposal with logged override** — the reasoning model may name a different tool, but must supply a rationale, and the override is recorded and evaluated separately. |
| **MVP** | Stages 1 and 2, with a hand-tuned scoring function and a coverage rule: any hypothesis with zero evidence and an eligible tool forces a call before termination (an explicit anti-under-calling guard, since MedCTA found under-calling to be a dominant failure). |
| **Advanced** | Learned router (contextual bandit over case descriptors, or a small model trained on the profiling corpus); ATTC-style pre-call trust signals (arXiv 2604.08281) using the model's confidence in its own argument construction as a prior on whether the call will be sound. |
| **Failure modes** | Behaviour-routing deficiency — matching subtasks to the wrong capability under normal conditions, now a named failure class in the agent literature; redundant calls; oscillation between two tools; over-calling to inflate apparent thoroughness. |
| **Metrics** | **Tool-selection accuracy** against an oracle route; **controller gap** = oracle-routed outcome − actual-routed outcome (the MedCTA framing); under-calling rate (eligible + informative tool never called); redundant-call rate; argument validity rate; LLM-override win rate (does overriding the score ever help? report it honestly). |

---

# 5.6 Specialist Tool Wrapper

**Objective.** Provide one uniform, defensive, instrumented calling convention over wildly
heterogeneous models — and capture every uncertainty signal the underlying tool is willing to emit.

| | |
|---|---|
| **Input** | `AcquisitionAction` + `NormalisedCase` |
| **Output** | `ToolResult{tool_id, version, raw_output, native_uncertainty: dict, perturbation_outputs: list | None, preconditions_met: bool, latency_ms, error: ToolError | None, input_hash}` |
| **Strategy** | Abstract base class enforcing: precondition assertion before invocation; fixed seed; timeout with a typed timeout error (never a silent empty result); explicit error taxonomy (`PRECONDITION`, `TIMEOUT`, `RUNTIME`, `EMPTY_OUTPUT`, `SCHEMA_VIOLATION`); mandatory `native_uncertainty` capture; optional test-time-augmentation replicates. **The wrapper never interprets, summarises, or thresholds.** |
| **MVP** | Wrappers for: TorchXRayVision classifiers ×3–5 domains (18 pathology labels), TorchXRayVision/CheXmask segmentation (lungs, heart), a deterministic geometric measurement tool (cardiothoracic ratio, rotation index, inspiration level, costophrenic angle) built on the segmentation mask, a MedSigLIP-based similar-case retriever, and a guideline retriever over a small local corpus. |
| **Advanced** | MCP transport with per-tool sandboxing; GPU batching; async fan-out for independent calls; a rate-limited "expensive tool" tier. |
| **Failure modes** | Silent degradation (returning a default when the model fails); preprocessing mismatch between wrapper and model (a classic and invisible source of a 10-point accuracy loss); non-determinism from unseeded augmentation; swallowing exceptions. |
| **Metrics** | Wrapper overhead (ms above raw inference); error-taxonomy coverage (no `UNKNOWN` errors); determinism check (identical input → identical output, 100 repeats); **precondition-violation catch rate** using deliberately malformed inputs. |

---

# 5.7 Evidence Packet Builder

**Objective.** Convert a raw `ToolResult` into the system's universal currency: a typed, provenanced,
individually-addressable unit of belief.

**Schema (the most important object in the system):**

```python
class EvidencePacket(BaseModel):
    packet_id: str
    concept: ConceptID                    # UMLS CUI, e.g. pneumonia / cardiomegaly
    assertion: Enum[PRESENT, ABSENT, MEASURED, INDETERMINATE]
    value: float | str | None             # e.g. CTR = 0.44
    units: str | None
    polarity: Enum[SUPPORTS, REFUTES, NEUTRAL]
    source: SourceRef                     # tool_id + version + input_hash
    modality: Modality
    localisation: BBox | Mask | None      # WHERE in the image — required if available
    raw_confidence: float | None          # what the tool said
    reliability: ReliabilityScore         # what we believe (§5.8) — the decision-relevant field
    derivation: list[str]                 # parent packet_ids, for derived evidence
    supersedes: str | None
    timestamp: datetime
class MissingEvidence(BaseModel):        # a first-class packet type, not a null
    packet_id: str
    concept: ConceptID
    reason: Enum[MODALITY_ABSENT, TOOL_INELIGIBLE, TOOL_FAILED, BUDGET_EXHAUSTED, OUT_OF_SCOPE]
    would_have_resolved: list[HypothesisID]   # what this absence leaves unresolved
    obtainable: bool                          # can a human get it?
```

| | |
|---|---|
| **Input** | `ToolResult` + `CapabilityCard` + `NormalisedCase` |
| **Output** | `list[EvidencePacket | MissingEvidence]` |
| **Strategy** | Deterministic adapters per `output_kind` — a classifier score becomes one packet per label above a floor; a mask becomes a measurement packet via the geometric tool; retrieval becomes reference packets with distinct polarity semantics. **Localisation is mandatory when the tool can produce it**, because ungrounded assertions cannot be entailment-checked later and MedCTA identified *weak local grounding* as a leading failure. |
| **MVP** | Adapters for LABEL, SCORE, MASK, MEASUREMENT; `MissingEvidence` emitted by L1 and by the router when it declines a call. |
| **Advanced** | Cross-modal packet linking (this text finding and this image region concern the same entity); temporal packets against a prior study. |
| **Failure modes** | Threshold choices smuggled into the builder (thresholding is a *decision*, it belongs in L7/L8); losing localisation; concept-mapping errors that silently split one finding into two. |
| **Metrics** | Schema validity 100%; **localisation coverage** (% of packets from localisable tools that carry a box/mask); concept-mapping accuracy against the PadChest UMLS mapping; `MissingEvidence` emission recall on ablated studies. |

---

# 5.8 Reliability Estimator

**Objective.** Produce, for every packet, a calibrated scalar `r ∈ [0,1]` meaning *"the probability
that this assertion is correct, given everything we know about this tool on this kind of case."*
This module is the technical core of the blueprint.

**Schema:**

```python
class ReliabilityScore(BaseModel):
    r: float                       # calibrated, decision-relevant (competence + integrity)
    competence: float              # prior: how good is this tool on cases like this one?
    integrity: float               # was the input valid; is the output stable?
    method: str                    # which estimator version produced r
    calibration_ref: str           # which calibration split r was fitted on
    ood_score: float               # distance from the tool's training distribution
```

**The signals, and why each one (D3 / D4 fixes):**

- **Competence** — looked up from the capability card's `CompetenceSlice` matching this case
  descriptor, then attenuated by an OOD score (embedding distance from the tool's training
  distribution, computed with MedSigLIP or the tool's own penultimate features). This is the only
  signal available *before* the call, so it also feeds the router.
- **Integrity** — input validity (preconditions, quality checks from L1) plus output stability under
  perturbation. Two cheap implementations: test-time augmentation spread for image models, and a
  RadFlag-style resampling check for text outputs (resample at higher temperature, count how many
  samples support each assertion, flag the unsupported ones — arXiv 2411.00299; training-free and
  black-box).
- **Packet Immutability & Agreement Placement (D3 Fix)** — To preserve packet immutability and prevent double-counting of corroboration, `agreement` is **moved out of packet creation** into the Fusion layer (L7). At packet creation, `r` relies strictly on `competence` and `integrity` (computable from that single packet alone).
- **Source Independence Caveat (D4 Fix)** — TorchXRayVision weights `mimic_ch` and `mimic_nb` are trained on the same MIMIC-CXR images (differing only in NLP labellers), and `chex` shares the CheXpert labeller; realistically ~3 independent source families exist (NIH, CheXpert/MIMIC, PadChest), not 5–6. Furthermore, weights do not support the identical set of 18 labels; inter-tool agreement is defined strictly on the supported label intersection, which must be enumerated empirically on Day 1 of Phase 0.

**Combination.** v1: logistic regression over competence and integrity plus the raw confidence, fitted on a
held-out split where packet correctness is known, then temperature-scaled. Simple, inspectable,
reportable. v2: a small gradient-boosted model or a learned per-tool head; optionally conformal so
that `r` supports a distribution-free guarantee rather than a point estimate.

| | |
|---|---|
| **MVP** | Competence from the profiling run; integrity from quality checks + 5-replicate TTA; logistic combiner; report reliability-vs-correctness AUROC and ECE. |
| **Advanced** | Per-tool-class estimators exploiting the Confidence Dichotomy finding (evidence tools need *downward* correction, verification tools do not); online updating; conformal reliability intervals. |
| **Failure modes** | Reliability that is itself miscalibrated — this makes everything downstream worse than trusting raw scores, and it is the main risk of the whole design; agreement collapsing when the tools share training data (correlated errors masquerading as corroboration — the "diversity collapse" phenomenon in a new costume); OOD score dominated by acquisition artefacts rather than pathology. |
| **Metrics** | **AUROC of `r` for predicting packet correctness** (the headline number for this module); ECE and Brier of `r`; Spearman ρ(reliability, error); **separation between clean and corrupted tool outputs** (feed the system a deliberately degraded tool and check `r` drops); ablation of individual signals. |

---

# 5.9 Differential Hypothesis Manager *(added module)*

**Objective.** Maintain the ranked hypothesis set that gives abstention, routing, and missingness
their meaning.

```python
class Hypothesis(BaseModel):
    hypothesis_id: str
    concept: ConceptID
    prior: float
    posterior: float
    supporting: list[str]          # packet_ids
    refuting: list[str]
    status: Enum[ACTIVE, EXCLUDED, NOT_EXAMINED, UNRESOLVABLE]
    unresolvable_because: list[str]        # MissingEvidence packet_ids
    discriminating_observations: list[ConceptID]   # what would separate this from its rivals
    severity: Enum[ROUTINE, URGENT, CRITICAL]      # "can't-miss" flag
```

| | |
|---|---|
| **Input** | `TaskFrame` (seeds), evidence updates from L7. |
| **Output** | `HypothesisLedger` with posterior, coverage, and per-hypothesis explanation. |
| **Strategy (D1 & D5 Fixes)** | **Two-level hypothesis space:**<br>1. *Findings level*: Multi-label independent probabilities over ~18 CXR findings with **no renormalisation** (co-occurrence like effusion + consolidation is expected). Metrics: per-label AUROC, mAP.<br>2. *Explanation level*: Small mutually-exclusive set (~8–12: pneumonia, oedema, effusion, pneumothorax, COPD/hyperinflation, mass/nodule, normal, OTHER) with a hand-written `P(finding | explanation)` matrix. Only the explanation level is a simplex (renormalised, ranked, abstained on).<br>**Capture@K (D5 Fix)**: Evaluated strictly at the explanation level via an explicit ~20-row action-mapping table (`explanation -> next diagnostic test / action`). |
| **MVP** | Fixed findings set + mutually-exclusive explanation set, prevalence priors, hand-authored discrimination table and action-mapping table, `severity` flags on pneumothorax / pneumomediastinum / free air. |
| **Advanced** | Open hypothesis space with retrieval-grounded candidate generation; discrimination table learned from co-occurrence statistics; guideline-derived priors. |
| **Failure modes** | Premature exclusion (posterior driven to ~0 by one unreliable packet — mitigated by flooring exclusion at a reliability threshold); anchoring on the indication text; `NOT_EXAMINED` being reported as `EXCLUDED`, which is the clinically dangerous confusion this module exists to prevent. |
| **Metrics** | DDx **Top-1 / Top-3 / Top-10** accuracy at explanation level; **Capture@K** via action-mapping table; *can't-miss recall* (never drop a CRITICAL hypothesis without explicit refuting evidence); calibration of the posterior itself (reliability diagram over hypotheses). |

---

# 5.10 Acquisition Planner *(added module — and the strongest research seam)*

**Objective.** Decide *whether to acquire more evidence at all*, and which observation is worth the
most, given the current hypothesis set.

| | |
|---|---|
| **Input** | `HypothesisLedger`, `AvailabilityMap`, remaining budget, registry. |
| **Output** | `PlanDecision{action: Enum[ACQUIRE, STOP_SUFFICIENT, STOP_EXHAUSTED, REQUEST_HUMAN], target: ConceptID | None, eig: float, justification: str}` |
| **Strategy** | Expected information gain over the hypothesis posterior. For a candidate observation `o` from tool `t` with reliability profile `r_t`, estimate the expected reduction in posterior entropy, discounted by `r_t` — an unreliable observation is worth less *even if it addresses the right question*, which is the coupling that makes this interesting. Stop when max EIG falls below a cost-calibrated threshold. |
| **MVP** | Discrete EIG over the closed hypothesis set with a lookup table of `P(observation | hypothesis)` from training-set co-occurrence; a hard round cap of 4. |
| **Advanced** | Learned value-of-information; budget-aware planning under a latency SLA; planning that can propose *human* actions ("order a lateral") as well as tool actions — which is what makes it clinically real. |
| **Failure modes** | EIG estimates dominated by a bad co-occurrence table; endless acquisition (mitigated by the hard cap); stopping too early on a sharp-but-wrong posterior. |
| **Metrics** | Calls-to-correct-answer vs. a call-everything baseline (efficiency); **regret vs. oracle acquisition order**; fraction of stops that were premature (would one more call have changed the disposition? — measurable by exhaustive replay); EIG–actual-information-gain correlation. |

---

# 5.11 Conflict Detector

**Objective.** Find and classify contradictions between packets, and refuse to let them be silently
averaged.

```python
class Conflict(BaseModel):
    conflict_id: str
    packets: list[str]
    kind: Enum[DIRECT_CONTRADICTION, MAGNITUDE_DISAGREEMENT, CROSS_MODAL, TEMPORAL, ONTOLOGICAL]
    severity: Enum[LOW, MEDIUM, HIGH]
    resolution: Enum[UNRESOLVED, RESOLVED_BY_RELIABILITY, RESOLVED_BY_EVIDENCE, DEFERRED_TO_HUMAN]
    note: str
```

| | |
|---|---|
| **Input** | The Evidence Ledger after every packet write. |
| **Output** | New/updated `Conflict` records. |
| **Strategy** | Rule-based on concept + assertion + polarity: same concept with opposite assertions is a `DIRECT_CONTRADICTION`; numeric packets outside each other's tolerance are `MAGNITUDE_DISAGREEMENT`; image-vs-text mismatch is `CROSS_MODAL`. Severity scales with the reliability of the *conflicting* packets — two high-reliability packets in contradiction is the serious case, and it should raise severity, not be resolved by picking the larger number. |
| **MVP** | The three rule types above; a per-concept numeric tolerance table; severity from `min(r_i, r_j)`. |
| **Advanced** | NLI-based contradiction detection for text packets; ontology-aware conflict (a parent concept asserted absent while a child is asserted present — this is what SNOMED/UMLS relations buy you); conflict typing learned from adjudicated cases. |
| **Failure modes** | Over-firing on legitimate co-occurrence (effusion *and* consolidation is not a conflict); missing cross-modal conflicts because concepts were mapped inconsistently; resolving by majority when the majority shares training data. |
| **Metrics** | Conflict detection precision/recall against injected contradictions (you can generate these: corrupt one tool and check the conflict fires); false-conflict rate on clean cases; **conflict preservation rate** — of detected conflicts, what fraction survive into the final output rather than being flattened (a metric the ERC survey correctly identified as not yet existing). |

---

# 5.12 Fusion Coordinator

**Objective.** Update the hypothesis posterior from the evidence, transparently and reproducibly.

| | |
|---|---|
| **Input** | Evidence Ledger, Hypothesis Ledger, Conflict Log. |
| **Output** | Updated posteriors + `FusionTrace` (the per-packet contribution to each hypothesis) + `coverage`. |
| **Strategy (D2 & D3 Fixes)** | Reliability-weighted log-odds accumulation at the explanation level: `logit(h) += w(packet) × llr(packet, h)`, where `w = f(reliability)` is explicitly defined (e.g. $w = \max(0, \ln(r / (1 - r)))$ or $w = r$), scaling the likelihood update. Print $w$ and $f$ explicitly in the trace. Cross-source corroboration/agreement is counted here once across observed packets (D3 fix). Renormalise over *observed* evidence only, masking absent modalities out of the update and recording the mask. Deliberately **not** an LLM judgement: the fusion step must be arithmetic you can print. |
| **MVP** | Naive-Bayes-style log-odds with explicit reliability weighting function $f(r)$, single-counted agreement corroboration, and a per-packet contribution trace. |
| **Advanced** | Learned fusion head trained on the profiling corpus, with the reliability weights as features; Shapley-style per-packet attribution (CoMMa's contribution, worth borrowing, and cheap in a closed hypothesis space); explicit correlation modelling between tools that share training data. |
| **Failure modes** | Double-counting correlated evidence (three classifiers trained on overlapping data are not three independent votes — this is the single biggest correctness risk in the fusion layer); reliability weights that saturate; a fusion trace that does not sum to the posterior (a bug that will bite you and is worth an assertion). |
| **Metrics** | Posterior calibration (reliability diagram, ECE over hypotheses); fusion-trace additivity assertion; ablation vs. unweighted fusion and vs. best-single-tool; **calibration-under-missingness curve** (the headline plot for the graceful-degradation claim). |

---

# 5.13 Abstention Engine

**Objective.** Convert the fused state into a typed, actionable disposition with a stated risk
guarantee.

```python
class Disposition(BaseModel):
    kind: Enum[ANSWER, ANSWER_WITH_CAVEAT, REQUEST, ESCALATE]
    reason_code: Enum[OK, INCOMPLETE_COVERAGE, MISSING_MODALITY, CONFLICT,
                      UNCERTAIN, TECHNICAL, OUT_OF_SCOPE, BUDGET]
    requested_item: ConceptID | None
    caveats: list[str]
    selective_risk_target: float          # the α the threshold was calibrated for
    threshold_used: float
    calibration_ref: str
```

| | |
|---|---|
| **Input** | Posterior, conflicts, coverage, availability, budget. |
| **Output** | `Disposition`. |
| **Strategy (D8 Fix)** | The ordered decision rules from §4.3-L8. The acceptance threshold is set by **split conformal risk control / Learn-then-Test** on a calibration set, targeting a user-specified error rate α among accepted cases (Angelopoulos et al., ICLR 2024 / Ann. Appl. Stat. 2025). **Cascading rule (D8 Fix)**: `REQUEST` (rule 3) occurs before conformal thresholding (rule 4). Hard cases routed to `REQUEST` must be counted as abstentions in all risk-coverage plots, and a `REQUEST`-disabled baseline must be evaluated alongside to prevent selective risk distortion. |
| **MVP** | Rule ordering + split-conformal threshold at α ∈ {0.05, 0.10, 0.20}; risk–coverage curve (with `REQUEST` counted as abstention) as the primary deliverable. |
| **Advanced** | Cost-sensitive deferral (asymmetric costs for missing a CRITICAL hypothesis); cascade to a larger model before escalating to a human, in the style of the cascaded-LLM deferral work (NeurIPS 2025, arXiv 2506.11887); conditional conformal by subgroup. |
| **Failure modes** | Over-abstention that makes the system useless (the mirror failure of the one everyone worries about, and the more likely one here); threshold overfitting to the calibration split; the guarantee silently voided under shift and reported anyway. |
| **Metrics** | Risk–coverage curve and **AURC** (counting `REQUEST` as abstention); selective risk at fixed coverage; empirical error rate among accepted cases vs. target α (with and without `REQUEST` enabled); over-abstention rate on cases a strong baseline gets right; **request specificity** — replay the case with the requested item supplied and measure how often the disposition actually changes. |

---

# 5.14 Explanation Generator

**Objective.** Render the ledgers as clinical text without adding anything to them.

| | |
|---|---|
| **Input** | `CaseState` (ledgers + disposition). **Not** the original free-text query, and **not** the raw images. |
| **Output** | `ClinicalOutput{structured_findings, ranked_differential, disposition_statement, prose, citations: dict[sentence_idx, list[packet_id]], dropped_sentences: list[str]}` |
| **Strategy** | Structured output is generated deterministically from the ledger (templating). Only the prose summary is LLM-generated, from a prompt containing the ledger and an instruction to cite packet IDs. Then the **entailment gate**: split into sentences; for each, check entailment against its cited packets (RadFact-style, an NLI model or a constrained LLM judge); drop failures and record them. The generator is explicitly forbidden from receiving the images, which structurally prevents it from re-perceiving and contradicting the evidence layer — the exact failure CARE exhibits. |
| **MVP** | Deterministic structured section + templated prose + LLM polish + entailment gate with a local NLI model. |
| **Advanced** | Audience-conditioned rendering (radiologist / referring clinician / patient) with the same ledger; visual evidence overlays; uncertainty-calibrated language ("possible" vs "probable" mapped to posterior bands, and validated against how clinicians read those words). |
| **Failure modes** | Confidence inflation in prose relative to the posterior (say the number, not just the adjective); the entailment gate being too lenient to matter; dropping so many sentences the output is incoherent — which is a signal the ledger is thin, not that the gate is wrong. |
| **Metrics** | **Evidence coverage** (% of clinical claims with a valid citation); **unsupported-claim rate** post-gate (target ≈0); entailment-gate precision/recall on injected unsupported sentences; RadGraph-F1 / GREEN / RadFact against reference reports where available (noting GREEN's documented length bias); hedging–posterior correlation. |

---

# 5.15 Audit Logger

**Objective.** Emit a complete, standards-shaped, PHI-safe record as the run happens.

| | |
|---|---|
| **Input** | Every module event. |
| **Output** | OTel spans + a per-case `CaseBundle` (JSONL). |
| **Strategy** | OpenTelemetry GenAI semantic conventions: `invoke_agent` root span, `chat` child spans per LLM call with `gen_ai.request.model` / `gen_ai.usage.*` / `gen_ai.response.finish_reasons`, `execute_tool` spans per tool call. **Message and image content goes in span *events*, not indexed attributes**, following the convention's own rationale — attributes are always indexed, size-limited, and would expose sensitive content, whereas events can be filtered or dropped at the Collector without touching application code. That single choice is the PHI-redaction hook the governance track will need. Backend: self-hosted Langfuse or Grafana Tempo. Record honestly that the GenAI conventions were still at Development stability (semconv v1.41, mid-2026), so pin the version. |
| **MVP** | OTel SDK + Langfuse; Case Bundle with input manifest hash, model/tool versions, full ledger, disposition, and resource accounting. |
| **Advanced** | Hash-chained bundles for tamper evidence; W3C PROV / PROV-AGENT export; retention policy enforcement; Collector-level redaction processors. |
| **Failure modes** | Logging after the fact (reconstruction is not provenance); PHI in indexed attributes (the mistake the convention exists to prevent); unbounded bundle growth; trace IDs that do not connect the LLM spans to the packets. |
| **Metrics** | **Audit completeness** against a field checklist (see §10.4); replay determinism rate; logging overhead as % of wall-clock; PHI-leak scan pass rate on synthetic identifiers deliberately injected into the input. |

---

# 5.16 Governance Connector

**Objective.** Exist in v1 as five documented no-ops so that v2 is an addition, not a refactor.

| | |
|---|---|
| **Input** | `(hook_name, payload, context{user_role, purpose, jurisdiction, consent_ref})` |
| **Output** | `GovernanceDecision{allow: bool, modified_payload, obligations: list[str], reason}` |
| **Strategy** | Five named hooks (`pre_ingest`, `pre_tool_call`, `pre_fusion`, `pre_response`, `pre_persist`), each a `lambda payload, ctx: GovernanceDecision(allow=True, modified_payload=payload)` in v1, each with a documented, frozen signature and each already logged. LangGraph's native human-in-the-loop interrupt is the mechanism for the blocking case in v2. |
| **MVP** | No-ops + logging + one integration test per hook proving that a deny decision is honoured end-to-end (test the mechanism now, ship the policy later). |
| **Advanced** | OPA/Rego or Cedar policy engine; RBAC from role claims; PHI redaction at `pre_response` and at the OTel Collector; consent verification at `pre_ingest`; human-review queue at `pre_persist`. |
| **Failure modes** | Hooks placed after the decision they are supposed to govern (the classic and fatal error — see §11 R8); signature churn breaking the later integration; hooks that can be bypassed by a code path added later. |
| **Metrics** | Hook coverage (every irreversible action preceded by a hook — assert in CI); deny-honoured rate 100%; governance latency overhead. |

---

# 6. Recommended Initial Implementation Scope (MVP)

# 6.1 The choice: chest radiography + indication text → structured findings, ranked differential, disposition

**Modality.** Frontal chest radiograph (PA/AP), plus lateral when present, plus a short free-text
indication and, where the dataset provides it, basic demographics.

**Task.** Given a CXR study and an indication, produce (a) a structured findings list with
localisation where available, (b) a ranked differential over a closed hypothesis set with per-
hypothesis evidence links, and (c) a typed disposition. Secondary task for external comparability:
multiple-choice CXR reasoning questions, so results can be placed against ChestAgentBench.

# 6.2 Why this scope and not one of the alternatives

| Candidate | Why not first |
|---|---|
| **Ophthalmology (fundus / glaucoma)** | Attractive because MedAgent-Pro's failure is well documented there, and the vCDR/DH indicator structure is a clean tool pipeline. But the open tool ecosystem is thin — you would train the specialists yourself, which is a month you do not have. Good **second** domain for a portability claim. |
| **Pathology (WSI)** | Gigapixel I/O, heavy compute, slow iteration, and tiling decisions dominate the results. Wrong for a first build. |
| **General multimodal VQA** | Maximally impressive-sounding, minimally measurable. Ground truth is a letter choice; you cannot evaluate reliability estimation, missingness, or grounding against it. This is precisely the benchmark family the Illusion of Readiness study showed to be shortcut-solvable. |
| **EHR / clinical notes** | Genuinely good and MedAgentBench makes it tractable — but it is *unimodal text*, so missing-modality and cross-modal-conflict claims become impossible to make. Correct as the **v2 second modality**, where it turns the missing-modality story from synthetic to real. |
| **3D CT / MRI** | The largest gap in the field (2D→3D chasm) and the worst first project: volumetric I/O, no cheap open specialists, and no way to get results in eight weeks. |

**Why CXR wins, concretely — five reasons, in order of importance:**

1. **It is the only domain where you get a heterogeneous pool of specialists for free.**
   TorchXRayVision ships DenseNet classifiers over 18 pathologies trained on *different source
   hospitals* (`nih`, `chex`, `mimic_ch`, `mimic_nb`, `pc`, `rsna`, `all`), plus segmentation, plus
   dataset wrappers, plus built-in utilities for constructing covariate-shift and
   spurious-correlation splits. A tool router and a reliability estimator are only meaningful if
   different tools are genuinely better on different cases — here they demonstrably are, and you
   train nothing to get that.
2. **Verification tools are cheap and deterministic.** Cardiothoracic ratio, rotation index,
   inspiration level and costophrenic angle are rule-based geometric computations on a segmentation
   mask. Following the Confidence Dichotomy result — verification tools with deterministic feedback
   improve agent calibration where evidence tools degrade it — this gives you a free, high-reliability
   evidence class to pair against the classifiers. CXReasonAgent's CheXStruct-based design is the
   proof of concept.
3. **Open data with real localisation.** VinDr-CXR gives 18,000 images with **radiologist bounding
   boxes** (fully released training split) — the grounding ground truth you need for entailment and
   localisation metrics. PadChest gives 160k+ images with a hierarchical taxonomy **already mapped to
   UMLS**, which removes an ontology sub-project. Neither requires credentialed access.
4. **A ready-made OOD stress split with published numbers.** The CXR-LT 2026 challenge assembled
   145,000+ images from PadChest + NIH specifically to work around MIMIC-CXR access restrictions,
   with Task 1 = 30 known classes (winning mAP 0.5854) and Task 2 = **6 unseen rare classes**
   (winning mAP 0.4315). That is a distribution-shift arm you do not have to construct or justify.
5. **A real baseline exists and is beatable on the axes that matter.** MedRAX (ICML 2025) is an
   open-source ReAct agent over 8 CXR tools reporting 63.1% on ChestAgentBench. You are not claiming
   to beat it on accuracy — you are claiming to match it on accuracy while adding calibrated
   reliability, typed abstention, and graceful degradation, and then to beat it decisively under
   stress. That is a much easier and much more interesting claim.

# 6.3 Concrete MVP specification

| Element | Choice |
|---|---|
| **Primary data** | VinDr-CXR (18k, boxes, open), PadChest subset (~30k sampled, UMLS-mapped labels, Spanish reports — translate offline or use label-only), NIH ChestX-ray14 (weak labels, scale) |
| **OOD / stress data** | CXR-LT 2026 splits (PadChest + NIH, incl. 6 unseen rare classes); a held-out source-domain split (train profiling on `nih`+`chex`, evaluate on `pc`) |
| **External comparability** | ChestAgentBench (2,500 six-choice questions over 675 Eurorad cases) and CheXbench; SLAKE / VQA-RAD for sanity |
| **Reference reports** | CheXpert Plus or PadChest reports for report-level metrics; MIMIC-CXR only if credentialing lands in time (do not plan on it) |
| **Reasoning backbone** | MedGemma-4B multimodal (default), Qwen2.5-VL-7B-Instruct (swap arm) — (D7: 27B bf16 requires ~54GB VRAM and is excluded from the single-GPU v1 stack) |
| **Encoder / retrieval** | MedSigLIP (400M) for view detection, OOD scoring, and similar-case retrieval |
| **Evidence tools** | TorchXRayVision classifiers ×4 domains (18 labels each); TorchXRayVision / CheXmask segmentation (lungs, heart) |
| **Verification tools** | Geometric measurement suite (CTR, rotation, inspiration, costophrenic angle, tracheal deviation) built on the masks |
| **Retrieval tools** | MedSigLIP k-NN similar-case retriever over a labelled index; a small local guideline corpus (open CXR interpretation references) |
| **Hypothesis space** | ~18 findings + `OTHER/OUT_OF_SCOPE`; CRITICAL flags on pneumothorax, pneumomediastinum, free subdiaphragmatic air |
| **Orchestration** | LangGraph (typed state, checkpointing, interrupts) |
| **Serving** | vLLM, one 16–24 GB GPU; everything local |
| **Scale** | 2,000-case dev set, 1,000-case test set, 1,000-case calibration set (disjoint), 500-case OOD set |

# 6.4 What is deliberately excluded from v1

- 3D volumetric data (CT/MRI) — the field's biggest gap and the wrong first project.
- Multi-turn dialogue and simulated patient agents.
- Any RL or fine-tuning of the backbone. The claim is architectural; training confounds it.
- Real EHR integration and any write action. (Synthea / MedAgentBench read-only in v2.)
- Pathology, genomics, ECG, ultrasound.
- Real-time inference targets. Measure latency, do not optimise it yet.
- Any clinical deployment, IRB submission, or claim of clinical validity.
- The governance policy engine — hooks only.
- Learned routers, learned fusion, conformal reliability intervals. All are v2 research, and each is
  a candidate extension in §9.

# 6.5 What makes the MVP genuinely *agentic* (and not a pipeline with extra steps)

This is the question a supervisor will ask, so answer it precisely. Four properties, all present in
v1, none of which a static pipeline has:

1. **The tool sequence is not fixed.** The Acquisition Planner chooses the next observation by
   expected information gain over the current hypothesis set. Two cases with the same inputs and
   different indications will follow different tool paths, and the path is logged.
2. **The system can decide to stop, or to not answer.** `STOP_SUFFICIENT` and the four disposition
   types are decisions the system makes, not thresholds a human set at the top.
3. **The system can ask for something it does not have.** `REQUEST(lateral_view)` is an action
   directed at the world, and its usefulness is independently measurable.
4. **The system reasons about its own instruments.** The reliability estimator is a model of the
   tools, not of the disease. Nothing in a conventional pipeline has an opinion about whether its
   own components should be believed.

Note what is *not* on that list: multiple LLM personas talking to each other. That is theatre, and
the evidence (MAST, MedAgentBoard, CoMMa's +0.5% AUC for 5 agents) says it usually costs more than
it returns.

# 6.6 Real vs. simulated in the MVP — stated plainly

| Component | Status | Note |
|---|---|---|
| Images and labels | **Real** | Public datasets, real radiologist annotations (VinDr boxes) |
| Specialist tools | **Real** | Pretrained models, real inference, real per-domain competence differences |
| Reliability estimation | **Real** | Fitted on held-out data, evaluated against actual packet correctness |
| Differential + disposition | **Real** | Real algorithm; hypothesis space is closed and curated |
| **Missing modality** | **Simulated** | Ablation of laterals / priors / indication text. Honest framing: "controlled missingness", not "real-world missingness" |
| **Tool corruption** | **Simulated** | Controlled injection, following AgentProp-Bench's methodology — this is the correct way to measure propagation causally, and should be described as a strength, not a limitation |
| **Clinician escalation** | **Simulated** | Escalations are counted and rubric-scored offline; no clinician is in the loop |
| **EHR context** | **Absent in v1** | Synthea / MedAgentBench in v2 |
| **Deployment / workflow** | **Absent** | No DICOM router, no PACS, no HL7 |

Being explicit about this table in the write-up is worth more than pretending otherwise. Every
paper in this space that overclaimed on this axis is now a cautionary citation.

---

# 7. Suggested Technical Stack

Everything below is open-source, runs on one workstation with a single 16–24 GB GPU, and requires no
paid API. That is a hard constraint, for the licence reasons in P8 and the budget reasons in reality.

# 7.1 Stack table

| Concern | Choice | Rationale |
|---|---|---|
| **Reasoning backbone** | MedGemma-4B multimodal (default); Qwen2.5-VL-7B-Instruct (swap) | MedGemma 4B fine-tuned variant reports SOTA CXR report generation (RadGraph-F1 30.3). **Hardware constraint (D7)**: MedGemma-27B in bf16 (~54GB VRAM) cannot co-reside with tool wrappers on a 24GB GPU and is excluded from v1. **Spatial grounding constraint (MedFlowBench)**: MedGemma 27B / 4B IT models score ~0 on spatial localization (0.00 Strict / 0.00 Loc); all spatial coordinates must come from tools (segmentation/bounding box wrappers), never from prompting the LLM directly. |
| **Model serving** | vLLM (OpenAI-compatible endpoint); Ollama acceptable for laptop dev | One interface, three backbones, zero code change (P7). |
| **Orchestration** | **LangGraph** | Orchestration as an explicit typed state machine with durable checkpointing, time-travel debugging, and native human-in-the-loop interrupts. Consistently identified as the production choice where auditability and deterministic control matter. Note AutoGen has moved to maintenance in favour of Microsoft's unified Agent Framework — do not start there. CrewAI is fine for a throwaway demo and wrong for this. |
| **Tool transport** | Direct Python calls in v1 → **MCP** servers in v2 | MCP gives process isolation, language independence, and a natural authorisation boundary. RadFabric already demonstrates MCP as a CXR tool bus; an MCP wrapper for TorchXRayVision already exists. Be clear-eyed: MCP is a transport protocol, **not** a compliance guarantee — HIPAA-grade behaviour has to be implemented at the server (OAuth 2.1, RBAC, per-call audit, TLS/AES, PHI minimisation, narrow workflow-specific tools rather than wildcard scopes). |
| **Imaging tools** | `torchxrayvision`, MONAI (transforms/metrics), CheXmask masks, OpenCV for the geometric suite | Pip-installable, no training required. |
| **Schemas** | Pydantic v2 | Runtime validation is the enforcement mechanism for P6. |
| **Data storage** | Parquet for tabular; DuckDB for analysis; JSONL Case Bundles for traces; filesystem + manifest hashes for images; DVC or plain content hashing for dataset versioning | No database server to administer. DuckDB reads the trace JSONL directly, which makes evaluation queries trivial. |
| **Observability / audit** | OpenTelemetry SDK + **Langfuse (self-hosted)** or Grafana Tempo | Standards-shaped from day one (P9). Pin the semconv version — GenAI conventions were at Development stability at v1.41. |
| **Experiment tracking** | MLflow (self-hosted) or Weights & Biases (free tier) | MLflow if data-residency discipline matters, which it eventually will. |
| **Evaluation harness** | Custom `herald-eval` package; `crfm-helm`/MedHELM for text-task comparability | MedHELM is Apache-2.0, has a clinician-validated 5-category / 22-subcategory / 121-task taxonomy, and its LLM-jury protocol reached ICC 0.47 with clinicians — above clinician–clinician agreement (0.43) and well above ROUGE-L (0.36). It requires Python 3.10. |
| **Judge / entailment** | A local NLI model (DeBERTa-MNLI class) for the entailment gate; a 3-model LLM jury for open-ended scoring | AgentProp-Bench showed substring judging agrees with humans at **κ = 0.049 (chance)** while a three-LLM ensemble reaches **κ = 0.432**. Never report a judge you have not validated against human labels. |
| **Interface** | Gradio (single-case explorer: image, packet table with reliability bars, hypothesis ranking, disposition, trace viewer) + a CLI for batch runs | The demo *is* the deliverable for the internship presentation. Build it in Phase 2, not Phase 7. |
| **Containerisation** | Docker + docker-compose (app, vLLM, Langfuse, MLflow); NVIDIA Container Toolkit | Reproducibility, and the MedAgentBench environment ships as a container too. |
| **CI** | GitHub Actions: schema validation, tool-removal suite, replay-determinism test, hook-coverage assertion | These four tests are the ones that keep the architecture honest. |
| **Config** | Hydra | Backbone/tool/threshold sweeps without code edits. |

# 7.2 Audit log format

Two artefacts, one canonical:

- **OTel spans** (streaming): `invoke_agent` → `chat` / `execute_tool`, with `gen_ai.*` attributes and
  content in **span events** so a Collector processor can redact or drop it before it leaves the
  machine.
- **Case Bundle** (canonical, one JSONL per case): schema version, input manifest with per-file
  hashes, model and weight versions, tool versions, complete Evidence and Hypothesis Ledgers, all
  conflicts, all acquisitions with their EIG and rationale, the disposition with its calibration
  reference, the final output with citations, dropped sentences, and resource accounting
  (wall-clock, tokens, GPU-seconds). The bundle is the unit of replay and the unit of audit.

# 7.3 Hardware and a realistic cost line

One workstation with a 24 GB GPU (RTX 4090 / A5000 class) runs everything: MedGemma-4B in bf16 with
room for the CXR tool models resident alongside. A 16 GB card works with 4-bit quantisation and a
measurable quality cost that should be reported, not hidden. Recurring cost: zero. This is not a
frugality flex — it is the configuration that keeps the door open to MIMIC later without violating
the data use agreement.

---

# 8. Evaluation Framework

This section is the one that determines whether the project produces a result or a demo. Two
methodological commitments run through all of it:

> **Commitment 1 — Every automatic judge is validated against human labels before its numbers are
> reported.** Substring matching agrees with human judgement at chance level (κ = 0.049); a
> three-model ensemble reaches κ = 0.432 (AgentProp-Bench). Budget 100–200 human-labelled items and
> report the κ alongside every judged metric.
>
> **Commitment 2 — Every headline number is reported three times**: clean, stressed, and
> shifted. A single number from the clean setting is exactly the artefact the Illusion of Readiness
> study taught the field to distrust.

# 8.1 The eight evaluation axes

---

# Axis 1 — Clinical / task performance

| | |
|---|---|
| **Metrics** | Per-finding AUROC and F1 (macro + per-class, 18 labels); DDx **Top-1 / Top-3 / Top-10**; **Capture@K** (does any top-K hypothesis license the correct next diagnostic test?); **can't-miss recall** (CRITICAL hypotheses never silently dropped); report-level RadGraph-F1, GREEN, RadFact where reference reports exist; ChestAgentBench accuracy for external comparability |
| **Stress tests** | Rare/long-tail classes (CXR-LT Task 2's 6 unseen classes); ambiguous cases; cases where the indication text contradicts the image |
| **Baselines** | (a) best single TorchXRayVision classifier; (b) MedGemma-4B zero-shot, no tools; (c) MedRAX-style ReAct agent over the same tools, no reliability layer; (d) supervised late-fusion model on the same features (the **non-agent ceiling** — keep it, or you will misattribute fusion gains to agency, exactly as MedAgentBoard warned) |
| **Expected failures** | Long-tail findings; conditions requiring a lateral; distinguishing consolidation from effusion on a single frontal; GREEN's known length bias inflating verbose outputs |

---

# Axis 2 — Tool-routing performance

| | |
|---|---|
| **Metrics** | Tool-selection accuracy vs. an oracle route; **controller gap** = oracle-routed performance − actual-routed performance (MedCTA's framing, and the number that best characterises this layer); **under-calling rate** (an eligible, informative tool that was never invoked); redundant-call rate; argument-validity rate; average calls-to-disposition; LLM-override win rate |
| **Stress tests** | Add a decoy tool with a plausible description and poor competence; remove the best tool and check the router degrades rather than fails; shift the case distribution so the previously-best tool is no longer best (this is the test that separates real routing from a memorised preference) |
| **Baselines** | Random eligible tool; always-call-all; fixed expert-authored script; **oracle router** (upper bound) |
| **Expected failures** | Preferring high-competence-prior tools on cases where they are out of distribution; under-calling verification tools because their EIG looks low in isolation; oscillation |

---

# Axis 3 — Reliability estimation quality

This is the axis on which the whole blueprint stands or falls.

| | |
|---|---|
| **Metrics** | **AUROC of `r` for predicting per-packet correctness** (headline); ECE and Brier of `r`; Spearman ρ(reliability, error); **clean-vs-corrupted separation** (AUROC of `r` distinguishing packets from an intact tool vs. a deliberately degraded one); per-signal ablation (competence / agreement / integrity, each removed in turn); per-tool-class breakdown testing the Confidence Dichotomy prediction that evidence tools need more downward correction than verification tools |
| **Stress tests** | Swap one classifier for a version trained on a mismatched domain; inject Gaussian noise / JPEG artefacts / rotation into the image; substitute a lateral where a frontal is expected; feed a non-CXR image entirely |
| **Baselines** | Raw softmax confidence; temperature-scaled softmax; ensemble variance alone; a constant `r` (does the estimator beat "trust everything equally"? — if not, say so) |
| **Expected failures** | Correlated errors across classifiers with overlapping training data inflating the agreement signal; OOD score responding to acquisition artefacts rather than pathology; miscalibration transfer failure from calibration split to OOD split (which is a *finding*, and should be reported as one) |

---

# Axis 4 — Abstention quality

| | |
|---|---|
| **Metrics** | Risk–coverage curve and **AURC** (counting `REQUEST` as abstention — D8 fix); selective risk at coverage ∈ {90%, 80%, 70%}; empirical error rate among accepted cases vs. target α, reported separately for ID and OOD; **over-abstention rate**; disposition-type distribution; **request specificity**; `REQUEST`-disabled variant evaluated alongside to ensure un-distorted risk curves |
| **Stress tests** | Sweep α; run the conformal guarantee on the OOD split, where exchangeability is violated by construction, and report the violation; adversarially easy cases to detect over-abstention; conflicting-evidence cases to check `ESCALATE(CONFLICT)` fires |
| **Baselines** | No abstention; softmax threshold; entropy threshold (known insufficient on its own); random abstention at matched coverage; **oracle abstention** (abstain exactly on the errors — the upper bound) |
| **Expected failures** | Over-abstention under distribution shift making the system useless precisely when it is most needed; conformal validity collapsing OOD; requests that are specific but not actionable |

---

# Axis 5 — Robustness to missing modalities

| | |
|---|---|
| **Metrics** | **Halt rate** (must be 0 — system must never halt on missingness); **graceful-degradation curve**: accuracy AND calibration vs. fraction of inputs removed, plotted together; **disclosure rate** (did the output name what was missing?); coverage honesty (`NOT_EXAMINED` never rendered as `EXCLUDED`); **calibration retention** — the target is the ICU-fusion pattern of losing ~0.09–0.11 AUPRC while calibration metrics stay near single-branch values |
| **Stress tests** | Ablate systematically: lateral only, indication text only, no text, no priors, image only, and 1-of-N tool outputs dropped at random; degrade rather than remove (a heavily under-inspired film instead of a missing film) |
| **Baselines** | Zero-imputation; mean-imputation; a generative reconstruction arm (to demonstrate the failure mode the Meta-Modal Agent authors warn about — plausible-looking fabricated evidence); single-modality specialist |
| **Expected failures** | Silent accuracy loss without disclosure (the failure this axis exists to catch); the availability template being wrong so nothing is ever reported missing; over-caveating until the output is unreadable |

---

# Axis 6 — Robustness to weak / noisy tool outputs

| | |
|---|---|
| **Metrics** | **Propagation rate** — P(wrong final disposition \| injected tool error). AgentProp-Bench's human-calibrated baseline for tool-using agents is ≈**0.62** (range 0.46–0.73); anything materially below that is the result. **Detection rate** and **recovery rate reported separately**, because they are statistically independent capabilities (Spearman ρ = 0.126, p = 0.747) — a system that notices a bad tool but cannot route around it is a different failure from one that never notices. Also: performance drop per corrupted tool (the MedAgent-Pro −16.5 F1 analogue), and time-to-detection in loop rounds |
| **Stress tests** | Controlled injection, one tool at a time and in pairs: label flip, confidence inflation (correct label, wrong confidence — the nastiest case), systematic bias, empty output, timeout, schema violation, and a *stale* tool returning last case's result |
| **Baselines** | Same pipeline with the reliability layer disabled (the crucial ablation); majority vote; single-tool |
| **Expected failures** | Confidence-inflation attacks that agreement cannot catch when tools are correlated; recovery failing because no alternative tool is eligible; detection firing but the router having nowhere to go |

---

# Axis 7 — Latency and cost

| | |
|---|---|
| **Metrics** | Wall-clock p50/p95 per case; tool calls per case; LLM calls and tokens per case; GPU-seconds; **reliability-layer overhead** as a % of total (the honest accounting of what P2 costs); cost per correctly-dispositioned case; cost per avoided error at each α |
| **Stress tests** | Concurrent load; worst-case (all hypotheses active, maximum rounds); the degenerate case where every tool is called |
| **Baselines** | Single VLM call; MedRAX-style ReAct; supervised fusion model (milliseconds — include it to keep the overhead honest) |
| **Expected failures** | The entailment gate and TTA replicates dominating latency; unbounded loops (the round cap exists for this); a reliability layer that costs more than it saves — which is a legitimate negative result and should be published as one if it happens |

---

# Axis 8 — Traceability and evidence completeness

| | |
|---|---|
| **Metrics** | **Evidence coverage** (% of clinical claims in the output with a valid packet citation); **unsupported-claim rate** post-entailment-gate (target ≈0); entailment-gate precision/recall against injected unsupported sentences; **localisation coverage**; **regeneration stability** (resampling k=5 to assess TaskFrame and Disposition agreement — D6 fix; bitwise replay reads from content-addressed LLM response cache); **audit completeness** against the §10.4 field checklist; **conflict preservation rate** (detected conflicts that survive into the output rather than being flattened) |
| **Stress tests** | Inject plausible-but-unsupported sentences and measure gate recall; delete a packet and confirm the citation breaks loudly rather than silently; replay 500 bundles and diff the dispositions; scan traces for synthetic PHI deliberately planted in the inputs |
| **Baselines** | Free-text VLM output (evidence coverage ≈ 0 by construction — this comparison is the point); MedRAX output; a citation-prompted LLM without a gate (the "asked nicely" baseline, which is what most systems actually do) |
| **Expected failures** | Gate too lenient (high coverage, low precision); dropped sentences producing incoherent prose; LLM response cache miss rate under non-deterministic continuous batching |

# 8.2 The three evaluation settings

| | **S1 — Clean benchmark** | **S2 — Stress test** | **S3 — Realistic workflow-inspired** |
|---|---|---|---|
| **Purpose** | Comparability with published work; sanity | Causal measurement of failure propagation | Face validity; the thing you demo |
| **Data** | VinDr-CXR + PadChest held-out; ChestAgentBench; SLAKE | Same cases, perturbed | Consecutive unfiltered cases from a held-out source, unbalanced, with real prevalence |
| **Manipulations** | None | Input ablation (remove image — the Illusion-of-Readiness shortcut probe); paraphrased queries; shuffled options; image corruption; tool corruption injection; modality ablation; tool removal | None applied; realism comes from the sampling |
| **Primary axes** | 1, 2, 7 | 3, 4, 5, 6 | 1, 4, 7, 8 |
| **Key question** | Are we in the right ballpark? | Does the architecture do what it claims when it matters? | Would a clinician find the output usable, and how often does it hand back work? |
| **Headline output** | Accuracy table vs. baselines | Propagation / degradation / risk–coverage curves | Disposition distribution + a clinician-rubric review of 100 cases |

**On S1, one warning.** Report the input-ablation control *on the clean set*: run the multiple-choice
benchmark with the image removed. If HERALD's accuracy stays high, the benchmark is shortcut-solvable
and no result on it means anything. Frontier models were shown to do exactly this. Publishing that
control for your own system is cheap, honest, and rare.

**On S3, one design note.** Realistic does not mean real. There is no hospital, no PACS, no
clinician. What S3 buys is *unbalanced prevalence and unfiltered difficulty*, which is enough to
expose over-abstention and to make the disposition distribution meaningful. Describe it exactly that
way.

# 8.3 Reporting template (fix this before you run anything)

Every experiment produces one row:

```
system | backbone | setting | axis | metric | value | 95% CI | n | judge_kappa | calib_ref | trace_manifest
```

Confidence intervals by bootstrap over cases. `judge_kappa` is mandatory and null-forbidden for any
judged metric. `calib_ref` identifies which calibration split any threshold came from.
`trace_manifest` is the hash of the Case Bundle set, so any number in the final report can be traced
to the exact runs that produced it. This costs an afternoon and makes the write-up trivial.

---

# 9. Research Extension Opportunities

Seven candidates, each a natural growth of a module that already exists in the MVP. Three of them
(R1, R2, R3) address the mandated themes: reliability-aware tool orchestration, abstention under
conflicting evidence, and graceful degradation under missing modality.

Novelty and difficulty are the author's honest estimates, not guarantees; the "prior art you must
beat" column is the part to read carefully, because it is where a naive version of each idea dies.

---

# R1 — Calibrated Tool-Trust Priors for Clinical Tool Orchestration
*(reliability-aware orchestration — mandated theme)*

**Problem statement.** Given a clinical case *c*, a set of specialist tools *T* with heterogeneous
and case-dependent competence, and a budget, learn a mapping *ρ(t, c) → [0,1]* that predicts the
correctness of tool *t*'s output on case *c* **before invoking it**, and use *ρ* jointly for tool
selection and evidence weighting. Quantify the reduction in the oracle-routing controller gap and the
reduction in error-cascade propagation attributable to *ρ*.

**Why it matters.** Routing is the empirically identified bottleneck: MedCTA (June 2026) reports a
large controller gap against oracle routing across 18 multimodal models, with failures concentrated
in under-calling and weak grounding rather than synthesis; the user's own survey records ToolSelect
at 72% against a 96% oracle. Meanwhile the *evidence-weighting* side is documented separately —
MedAgent-Pro loses 16.5 F1 to one weak tool. Nobody has connected the two: the same quantity that
tells you which tool to call should tell you how much to believe what it returns, and estimating it
once should improve both.

**Prior art you must beat.** ToolSelect (Attentive Neural Processes for model selection); Roumeliotis et al., *Trust-Aware Orchestration for LLM Agents* (arXiv:2507.10571, IEEE Access 2026 — global trust modulation via ECE/OCR/CG); Robust-TO's confidence-aware tool orchestration in video (arXiv 2606.26904) — which is *not* medical and *not* joint with abstention; the Confidence Dichotomy paper's RL approach (arXiv 2601.07264).

**What it modifies.** §5.4 Tool Registry (competence slices become a learned model), §5.5 Tool
Router, §5.8 Reliability Estimator.

**Novelty** medium-high (narrows to **per-case, case-descriptor-conditioned** competence priors in the **medical** domain, used **jointly for tool routing and evidence weighting** with measured transfer) · **Difficulty** medium · **Student feasibility** high.

**Minimum publishable claim.** "A learned per-case tool-trust prior reduces the controller gap by X
points and error propagation from 0.62 to Y on a controlled-injection benchmark, and the same prior
improves packet-level reliability AUROC by Z — one estimator, two uses."

---

# R2 — Conflict-Preserving Selective Prediction: abstention under contradictory evidence
*(abstention under conflicting evidence — mandated theme)*

**Problem statement.** Current selective prediction abstains on *low confidence*. Define and evaluate
abstention on *structured disagreement*: given a set of evidence packets with calibrated reliabilities
and detected conflicts, decide between answering, answering with the conflict surfaced, requesting the
specific discriminating observation, or escalating — with a distribution-free risk guarantee on the
accepted set. Introduce a **conflict-preservation** metric and show that conflict-triggered
abstention dominates confidence-triggered abstention on the risk–coverage curve where the two
disagree.

**Why it matters.** This is the empty cell the ERC survey identified explicitly ("no standard metric
exists yet, which is itself a contribution opportunity"). The theoretical motivation is strong:
multi-agent debate drives inter-agent correlation to 1 and terminal confidence saturates
*independently of error* (17× smaller variance than accuracy), so a confidence threshold is exactly
the wrong instrument when sources disagree. Clinically, unresolved discordance is often the most
informative thing the system knows, and averaging it away destroys it.

**Prior art you must beat.** Conformal abstention (Abbasi-Yadkori et al., NeurIPS 2024 WS); the
conformal safety layer for medical summarisation (arXiv 2606.08969); "Knowing When to Abstain"
(arXiv 2601.12471); cascaded LLMs with deferral (NeurIPS 2025, arXiv 2506.11887). All of these are
single-model or single-cascade; none operate on structured cross-source conflict.

**What it modifies.** §5.11 Conflict Detector, §5.13 Abstention Engine, and the Axis-4 evaluation.

**Novelty** high · **Difficulty** medium-high (the conformal guarantee under a conflict-conditioned
threshold needs care — exchangeability is easy to break accidentally) · **Feasibility** medium-high.

**Minimum publishable claim.** "Conflict-conditioned selective prediction achieves the same selective
risk at N points higher coverage than a confidence threshold, and preserves M% of clinically
significant disagreements that a fusion baseline eliminates."

---

# R3 — Missingness disclosure and calibration under incomplete evidence
*(graceful degradation under missing modality — mandated theme)*

**Problem statement.** Formalise three responses to a missing modality — **skip silently** (MedAgent-Pro), **fuse over observed data silently** (Cerebra), **impute generatively**, and **disclose-and-degrade** — and characterise their accuracy/calibration/trust trade-offs across a missingness sweep. Show that a disclosing agent that names what is absent, keeps the affected hypotheses alive as `UNRESOLVABLE`, and requests the specific missing item preserves calibration where imputation destroys it and silent handling hides it.

**Why it matters.** Existing systems handle missingness by skipping affected steps (MedAgent-Pro) or fusing over remaining modalities (Cerebra), but fail to disclose what hypotheses were left unresolved. On the other side, generative imputation (Meta-Modal Agent, arXiv 2605.25007) carries a severe risk under heavy missingness: reconstruction "may look plausible while adding unsupported evidence to the pipeline." The target behaviour of losing accuracy while retaining calibration is established in *supervised* fusion (mask, renormalise, fall back to a calibrated single specialist; ~0.09–0.11 AUPRC cost with calibration retained) but has never been demonstrated in an *agent* with explicit missingness disclosure and request capabilities.

**What it modifies.** §5.2 Modality Detector, §5.12 Fusion Coordinator, §5.13 Abstention Engine.

**Novelty** medium (narrowed: graceful degradation via skipping/fusion exists; *disclosing missingness and measuring its impact on calibration* does not) · **Difficulty** low-medium · **Feasibility** very high.

**Minimum publishable claim.** "Across a missingness sweep, disclose-and-degrade retains calibration
(ECE +ε) at an accuracy cost of Δ, while imputation retains accuracy but degrades ECE by 3ε and
produces unsupported evidence citations in K% of cases."

---

# R4 — Evidence-Value Planning: information-gain tool selection for differential diagnosis

**Problem statement.** Treat tool invocation as sequential experiment design over a hypothesis
posterior. Select the next observation to maximise *reliability-discounted* expected information
gain, and evaluate against oracle acquisition ordering, a fixed expert script, and call-everything.

**Why it matters.** This is the capability the AMIE authors named as untested and hard — "the
goal-directed acquisition of information under uncertainty" — and the *npj Digital Medicine*
ablation showing lab results improve DDx accuracy by up to 30% is direct evidence that acquisition,
not reasoning, is the binding constraint. The reliability discount is the novel coupling: an
observation that would resolve the question but comes from an unreliable instrument is worth less
than the naive EIG suggests, and no existing planner models that.

**What it modifies.** §5.10 Acquisition Planner, §5.9 Hypothesis Manager.

**Novelty** medium-high · **Difficulty** medium-high (EIG estimation is only as good as your
`P(observation | hypothesis)` table) · **Feasibility** medium.

---

# R5 — Provenance-Verified Generation: entailment-gated clinical narrative

**Problem statement.** Quantify how much clinical narrative generated from a structured evidence
ledger is *not* entailed by that ledger, and evaluate an entailment gate as a mitigation. Report
unsupported-claim rate, the accuracy cost of dropping unsupported content, and reader-rated
usefulness of gated vs. ungated output.

**Why it matters.** CARE's coordinator demonstrably rewrites correct tool outputs into incorrect
answers. The Illusion of Readiness study found fluent rationales citing non-existent image features
and concluded that reasoning fidelity must be independently validated. Yet essentially every medical
agent's final step is an ungated LLM summarisation of its own tool outputs. The instruments to fix
this exist (RadFact, TBFact, GREEN) and are used for *evaluating* systems, not for *gating* them.

**What it modifies.** §5.14 Explanation Generator.

**Novelty** medium (the technique is borrowed; the systematic clinical measurement is not) ·
**Difficulty** low-medium · **Feasibility** very high. Good first paper; excellent workshop paper.

---

# R6 — MedCascadeBench: a controlled-injection benchmark for error propagation in clinical tool agents

**Problem statement.** Build a benchmark that injects controlled, typed corruptions into the tool
layer of medical agents (label flip, confidence inflation, systematic bias, stale result, timeout,
schema violation) and measures propagation, detection and recovery as separate quantities across
multiple agent architectures.

**Why it matters.** AgentProp-Bench established the methodology in general domains and produced the
key numbers — propagation ≈0.62, and detection/recovery statistically independent (ρ = 0.126,
p = 0.747). No medical equivalent exists, and the medical case is where the asymmetric cost lives.
MedCTA evaluates tool *use*; it does not corrupt the tools.

**What it modifies.** Nothing in HERALD — it *uses* HERALD's wrapper layer as the injection point,
which is why the wrapper contract in §5.6 is worth getting right.

**Novelty** medium-high (benchmarks are undervalued and over-cited) · **Difficulty** medium ·
**Feasibility** high · **Best team fit**: two people, one building HERALD and one building the
benchmark against it.

---

# R7 — Governance-Evidence Sufficiency for clinical agents *(bridges to Track C)*

**Problem statement.** Define the minimal set of emitted metadata sufficient to reconstruct, audit,
and contest a clinical agent's decision after the fact, and measure whether current medical agent
architectures emit it. Map the field checklist to EU AI Act Articles 12/14/19 obligations and score
a set of published open-source medical agents against it.

**Why it matters.** Article 12 mandates automatic event logging over the system lifetime with ≥6
months retention, and Article 14 mandates the ability to intervene — but the regulation specifies
objectives, not a data model. The OTel GenAI conventions supply telemetry vocabulary but not clinical
evidence semantics. The gap between "the law requires traceability" and "here is the schema" is real,
current, and unusually well-suited to a governance-focused teammate.

**Novelty** medium · **Difficulty** low-medium technically, high on the legal-interpretation side ·
**Feasibility** high **for the second teammate**, and a poor fit for the implementation intern.

---

# 9.1 Which one to pick

| | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|
| Novelty |  |  |  |  |  |  |  |
| Difficulty |  |  |  |  |  |  |  |
| Falls out of MVP |  |  |  |  |  |  |  |
| Solo-feasible | yes | yes | yes | maybe | yes | yes | teammate |

**Recommendation: R1 as primary, R3 as insurance, R5 as the guaranteed-shippable minor result.**
R1 is where the strongest claim lives and it is directly enabled by the Phase 3 profiling run. R3 is
the one that will work even if R1's learned prior fails to beat the hand-tuned one. R5 is a complete
workshop paper that can be written from the MVP's logs with two extra days of work, which means the
project cannot end with nothing.

---

# 10. Governance Add-On Readiness

**Design stance: do not build governance now. Build the *seams* now, and make them impossible to
bypass.** The system in Track A is a research prototype on public data with no PHI and no clinical
use; a policy engine would be premature. What is not premature is ensuring that when a teammate
arrives to add one, they can do it without touching a single line of the reasoning core.

# 10.1 The five insertion points

| Hook | Fires before | v1 behaviour | v2 governance responsibilities |
|---|---|---|---|
| `pre_ingest` | any data enters L1 | log + pass | consent verification; purpose limitation; jurisdiction check; de-identification; data-minimisation (drop fields the task does not need) |
| `pre_tool_call` | any tool invocation | log + pass | RBAC on tool scope; egress control (does this tool leave the machine?); PHI minimisation on tool arguments; rate/cost limits; tool allow-list per role |
| `pre_fusion` | posterior update | log + pass | policy constraints on evidence admissibility (e.g. exclude tools not validated for this population); fairness monitors; provenance completeness assertion |
| `pre_response` | output leaves L9 | log + pass | output redaction; role-conditioned disclosure; required-disclaimer injection; blocking on policy violation; confidence-language policy |
| `pre_persist` | anything written to durable storage | log + pass | retention class assignment; encryption; human-review queue routing; log-integrity sealing |

Signature, frozen in v1:

```python
def hook(payload: T, ctx: GovernanceContext) -> GovernanceDecision[T]: ...
class GovernanceContext(BaseModel):
    user_role: Role; purpose: Purpose; jurisdiction: str
    consent_ref: str | None; case_id: str; trace_id: str
class GovernanceDecision(BaseModel, Generic[T]):
    allow: bool; modified_payload: T
    obligations: list[str]       # e.g. ["log_to_review_queue", "attach_disclaimer"]
    reason: str; policy_version: str
```

The v1 test that makes this real: **an integration test per hook that returns `allow=False` and
asserts the pipeline halts with the correct disposition and a complete trace.** Test the mechanism
now; the policy is someone else's problem later.

# 10.2 Where each governance concern attaches

| Concern | Attach at | Notes |
|---|---|---|
| Privacy / compliance filtering | `pre_ingest`, `pre_persist`, OTel Collector | Three-layer: at entry, at exit, and in telemetry. The Collector layer is the one people forget. |
| Role-based access control | `pre_tool_call`, `pre_response` | Roles gate *which tools may run* and *what detail is disclosed*, not just who can log in. |
| Audit trail | L10 (already there) | Governance consumes it; it does not need to build it. This is the payoff for P9. |
| Output redaction | `pre_response` | Must run **after** the entailment gate, or you redact the citations and break provenance. |
| Policy engine | all five hooks | OPA/Rego or Cedar; policies versioned and referenced by ID in the trace. |
| Evidence accountability | L6 + L10 | Every packet already carries tool ID, version, input hash. Accountability is a query over the ledger, not new machinery. |
| Human review checkpoint | `pre_persist`, and `ESCALATE` dispositions | LangGraph's checkpointing + interrupts make a genuine blocking review possible; the disposition type already identifies the cases that need one. |

# 10.3 Text-only governance is not enough — and this is the part most projects get wrong

Almost every governance discussion in this space is about text: redact names, filter outputs, check
for PHI strings. In a multimodal tool-orchestrating system that covers a minority of the surface.

**What text-only governance misses:**

- **Images carry identifiers that text filters cannot see.** DICOM headers contain names, MRNs,
  accession numbers, institution names and dates. Burned-in pixel annotations survive header
  scrubbing entirely. A facial reconstruction from a head CT is a re-identification vector no regex
  will catch.
- **Tool outputs are new PHI.** A segmentation mask, a measurement, an embedding — these are derived
  patient data. An embedding is *especially* awkward: it is not human-readable, so it evades review,
  and it may be invertible. Governance must classify tool outputs, not just model outputs.
- **Retrieval leaks across patients.** A similar-case retriever returns *other patients' data*. That
  is a cross-patient disclosure event and needs its own authorisation, its own audit record, and a
  policy on whether retrieved cases may appear in the output at all.
- **Tool calls are egress events.** Any tool not running locally is a data transfer. Under the
  PhysioNet DUA, sending MIMIC content to a third-party LLM API is a licence violation — governance
  must be able to *enforce* the local-only constraint, not merely document it.
- **Small-cell re-identification.** A rare-finding case in a small subgroup can be identifying even
  with every direct identifier removed. Redaction cannot fix this; suppression policy can.

**What this demands architecturally:** every hook receives *typed* payloads, so `pre_tool_call` can
distinguish "sending an image to a local classifier" from "sending an image to a remote endpoint",
and `pre_response` can distinguish a measurement packet from a retrieved-case packet. This is why P6
(schema contracts) is a governance prerequisite, not just a code-hygiene preference.

# 10.4 Metadata each module must emit for governance to be possible later

This is the concrete ask on the implementation intern. Emit these fields in v1 and the governance
layer is an addition; omit them and it is a rewrite.

| Module | Required emissions |
|---|---|
| L1 Ingestion | source ref, modality, acquisition date (shifted), consent ref (nullable), de-ID method + version, **per-file content hash** |
| L2 Triage | task type, requested scope, user role, purpose, out-of-scope flag |
| L3 Registry | tool ID + **version**, training provenance, licence, **locality flag (local / remote)**, validated populations |
| L4 Router | tool chosen, tools rejected + reason, EIG, competence prior, selection mode, override rationale |
| L5 Wrapper | input hash, output hash, latency, error class, **egress flag**, precondition status |
| L6 Reliability | reliability triple, calibration reference, OOD score, estimator version |
| L7 Fusion | per-packet contribution, renormalisation mask, coverage map |
| L8 Disposition | disposition, reason code, threshold + α + calibration reference, requested item |
| L9 Generation | citations per sentence, dropped sentences, generator model + version, gate model + version |
| L10 Audit | trace ID, span tree, bundle hash, retention class, schema version |

Two rules that make the difference between a log and an audit trail: **(a) every version is
recorded, never assumed** — "MedGemma-4B" is not a version, a weight hash is; **(b) every rejection
is recorded with its reason** — a trace that shows what the system did but not what it declined to do
cannot answer the question a regulator or a plaintiff will actually ask.

# 10.5 What is explicitly out of scope

No claim of HIPAA or GDPR compliance. No conformity assessment. No CE marking. No IRB. No clinical
validation. The governance track's deliverable is *demonstrated attachability* — a policy engine
wired into five hooks, with tests proving deny decisions are honoured and a completeness score
against §10.4 — not compliance.

---

# 11. Risks and Failure Modes

| # | Risk | Cause | Likely impact | Mitigation | MVP-acceptable? |
|---|---|---|---|---|---|
| **R1** | Wrong tool chosen | Competence priors estimated on too little data; case descriptor too coarse to distinguish the cases where tools differ; LLM override unconstrained | Wrong evidence enters the ledger with a plausible reliability score — the worst kind of error, because it is *confidently* wrong | Hard precondition filter (an ineligible tool is never a choice); logged overrides evaluated separately; report the controller gap rather than hiding it; forced-coverage rule against under-calling | **Yes** — provided the gap is measured and reported |
| **R2** | Weak tool trusted too much | Reliability estimator itself miscalibrated; correlated errors across tools sharing training data inflating the agreement signal; OOD score responding to acquisition artefacts | Error cascade — the MedAgent-Pro −16.5 F1 failure reproduced inside our own system, with the added insult that we claimed to prevent it | Calibrate on held-out data and report ECE of `r` itself; measure inter-tool error correlation explicitly and down-weight agreement between correlated tools; run the corrupted-tool stress test every build; report detection and recovery separately | **Yes**, if measured — this risk *is* the research question |
| **R3** | Correct evidence overwritten by the narrative layer | Generator re-reasons instead of rendering; entailment gate too lenient; prose inflates confidence beyond the posterior | The CARE failure. Destroys the entire value proposition, silently | Generator sees the ledger only, never the images; mandatory entailment gate with measured precision/recall; unsupported-claim rate as a build-gating metric; hedging–posterior correlation check | **Yes** with the gate; **No** without it |
| **R4** | Missing modality causes collapse | Fusion assumes complete evidence; a required tool's precondition fails and the loop cannot proceed | The Cerebra failure — a halt is worse than a degraded answer in every clinical context | `MissingEvidence` as a first-class packet type; mask-and-renormalise fusion; halt rate as a build-gating metric (target 0); tool-removal test in CI | **Yes** — this one is genuinely solved by design if the schema is right |
| **R5** | Over-abstention | Conformal threshold overfit to a calibration split that does not match the test distribution; conflict detector over-firing on legitimate co-occurrence; every axis independently adding caution | A system nobody uses. **This is the more likely failure than over-confidence**, and it is under-discussed | Report the over-abstention rate as a headline metric, not a footnote; sweep α and publish the whole curve; false-conflict rate on clean cases; compare against oracle abstention to show the headroom | **Yes**, if reported. Silent over-abstention is not acceptable |
| **R6** | Under-coverage of edge cases | Closed hypothesis space; tools cover 18 findings; anything else is invisible | Confidently normal reports on pathology outside the hypothesis set — clinically the most dangerous single failure in the design | Explicit `OTHER/OUT_OF_SCOPE` hypothesis that can win and triggers escalation; scope gating in L2; report performance on the CXR-LT unseen-class split, which is exactly this test | **Yes** with `OTHER` and the unseen-class evaluation; **No** without them |
| **R7** | Benchmark inflation | Optimising against a shortcut-solvable benchmark; tuning on the test set; LLM judge agreeing with itself | Numbers that do not survive contact with S2/S3 — the precise pathology the Illusion of Readiness study documented | Three-setting protocol; input-ablation control run on the clean benchmark and published; judge κ against human labels mandatory; frozen test set touched once | **Yes** with the protocol |
| **R8** | Governance hooks too late in the pipeline | Hooks placed after the decision they should govern; a code path added later that bypasses them | The governance layer becomes advisory theatre; Track C requires a core rewrite | Five named hooks in Phase 1; deny-honoured integration test per hook; **CI assertion that every irreversible action is preceded by a hook call** | **Yes** with the CI assertion; **No** with hooks alone |
| **R9** | Reliability layer costs more than it returns | Three signals, TTA replicates, entailment gate — all compute; benefit may be smaller than the latency penalty | The honest possibility that the architecture is not worth it | Measure reliability-layer overhead as an explicit % and report cost-per-avoided-error. **If it loses, publish that** — a well-measured negative result on a widely-assumed-good idea is a real contribution | **Yes** — and reporting it is mandatory |
| **R10** | Data access blocks the project | PhysioNet credentialed-access delays; dataset licence changes mid-project | Weeks lost at exactly the wrong time | Critical path uses only openly downloadable data (VinDr, PadChest, NIH, CXR-LT, Synthea); apply for MIMIC on day 1 and treat it strictly as an upgrade | **Yes** — this is why §6.3 is written the way it is |
| **R11** | Scope creep into a second modality | The blueprint is extensible, which is a temptation | Nothing finished; two half-systems | Modality is frozen for Phases 0–5. EHR arrives only after the full evaluation suite runs green on CXR | **Yes**, enforced by the roadmap |

---

# 12. Recommended Final Blueprint

# 12.1 The three alternatives that were considered

Before naming a recommendation, the alternatives should be on the record, because each is defensible
and each was rejected for a specific reason.

**Alternative A — "MedRAX+": tool-augmented ReAct with a better tool set.**
Take the existing open-source CXR agent, add tools, tune prompts, report a higher ChestAgentBench
number. *Rejected because* it inherits the exact architecture whose limits MedCTA measured across 18
models — the LLM both selects the tool and interprets its output, with no independent estimate of
whether the output deserves belief. It is fast to build and produces a number nobody can interpret.
There is no research seam because there is no new mechanism.

**Alternative B — "Clinical MDT": a multi-agent board of specialist personas.**
A radiologist agent, a clinician agent, and a moderator, debating to consensus. *Rejected because*
the empirical record is genuinely bad and the user's own survey documents it: MAM's debate overrode
correct answers (70.2% → 64.0%); CoMMa gained +0.5% AUC for 5× the agents; MDAgents underperformed
solo GPT-4 on MedQA. MAST attributes ~79% of multi-agent failures to specification and coordination
issues and concludes better base models will not fix them. Debate also *manufactures* confidence:
inter-agent correlation converges to 1 while terminal confidence saturates independently of error.
It demos beautifully and it is the wrong architecture.

**Alternative C — "Supervised-plus-explanation": a strong multimodal fusion model with a post-hoc
narrator.** Likely the most *accurate* option on the narrow prediction task, and cheap at inference.
*Rejected as the primary system* because it cannot handle an open task space, cannot request a
missing input, cannot be re-pointed without retraining, and has nothing to say about tool reliability
— there are no tools. **But it is retained as the non-agent ceiling baseline in §8**, and keeping it
there is what prevents misattributing fusion gains to agency.

# 12.2 The recommendation

> # **HERALD — Hypothesis-driven, Evidence-graded, Reliability-Aware, Layered Diagnostic agent**
>
> **One sentence:** A single-reasoner, many-tools clinical agent that maintains a typed evidence
> ledger and a ranked differential, scores the reliability of every piece of evidence independently
> of the tool that produced it, and terminates in one of four typed dispositions — answer, answer
> with caveat, request a named missing observation, or escalate — with every claim in its output
> traceable to a specific evidence packet.

**Why it is the best starting point.**

It is the only one of the four options in which each of the field's documented failure modes maps to
a *named module you can test in isolation*. Error cascade → Reliability Estimator, with a measurable
propagation rate. Missing-modality collapse → `MissingEvidence` packets and mask-and-renormalise
fusion, with a halt rate that must be zero. Silent failure → the Disposition Engine, with a
risk–coverage curve and a conformal guarantee. Coordinator over-editing → the entailment gate, with
an unsupported-claim rate. Routing failure → the controller gap. That one-to-one correspondence is
what makes it a research architecture rather than an application: every module has a metric, and
every metric has a paper's worth of headroom behind it.

**What makes it technically meaningful.**

The non-obvious commitment is that **the agent's clinical state is a typed data structure, not a
conversation.** Almost every published medical agent carries its beliefs in a chat transcript, which
is why almost none of them can tell you what they did not look at, why they believed a tool, or which
sentence in their output is load-bearing. Moving state into a ledger converts five separate hard
problems — provenance, missingness, conflict, abstention, and governance attachment — into
consequences of one schema decision. The second commitment is that **reliability is computed outside
the tool that produced the evidence**, which is the direct instantiation of a result now established
in an adjacent field (confidence-aware tool orchestration for video reasoning) and untested in
medicine.

**What makes it suitable for an internship build.**

Nothing in v1 requires training a model. The tools are pip-installable and pretrained. The data is
openly downloadable. The heterogeneous specialist pool that makes routing and reliability meaningful
comes free with TorchXRayVision's multi-source model zoo. The whole thing runs on one GPU. There is a
published open-source baseline (MedRAX, 63.1% ChestAgentBench) to be measured against, and a
published OOD split (CXR-LT 2026) to be stressed on. And there is a genuine demo — an image, a table
of evidence packets with reliability bars, a ranked differential, and a disposition that sometimes
says *"I need the lateral view"* — which is far more memorable in a presentation than an accuracy
number.

**What makes it extensible into research and governance.**

Research: three of the seven extensions in §9 are one module each, and the training data for the
strongest of them (R1) is generated as a side effect of Phase 3. Governance: the five hooks, the
typed payloads, and the OTel spine mean a second person can attach a policy engine without opening
the reasoning code — and §10.4 tells them exactly what metadata to expect.

**What it is not.** It is not a claim that this architecture is more accurate than a fine-tuned VLM.
It very possibly is not. The claim is that it is more *honest under stress*, and that this is
measurable. If the accuracy is comparable and the stress behaviour is decisively better, that is the
result. If the reliability layer turns out not to pay for itself, that is also a result, and §11-R9
commits to reporting it.

---

# 13. Build Roadmap

Eight phases. Weeks are indicative for one person working roughly full-time; the *ordering* is the
part that matters and is not negotiable.

---

# Phase 0 — Literature-to-capability mapping *(≈3 days)*

**Goals.** Convert the existing survey material into an engineering artefact: a table mapping each
documented failure mode to the module that must prevent it and the metric that will prove it did.
Freeze the hypothesis space, the concept vocabulary, and the tool shortlist.

**Deliverables.** `capability_map.md` (failure → module → metric → target); frozen concept
vocabulary with UMLS CUIs (bootstrapped from PadChest's mapping); `datasets.md` with licences,
access status, and download commands; MIMIC credentialing application submitted *and treated as
optional*.

**Dependencies.** The three existing survey documents.

**Show in a presentation.** The failure→module→metric table. It is the single slide that explains
why the architecture looks the way it does, and it is the slide a supervisor will remember.

---

# Phase 1 — MVP architecture skeleton *(≈1 week)*

**Goals.** All schemas, all module stubs, the LangGraph state machine, the OTel spine, and the five
governance hooks — with **no clinical intelligence whatsoever**. An end-to-end run on a single image
that produces a valid, complete, empty Case Bundle.

**Deliverables.** Pydantic schemas (§5); LangGraph graph with all nodes as pass-throughs; OTel +
Langfuse in docker-compose; five no-op hooks with their deny-honoured tests; CI running schema
validation, replay determinism, and the hook-coverage assertion.

**Dependencies.** Phase 0.

**Show.** A trace viewer displaying an empty but structurally complete case. Unglamorous and the most
important week in the project — this is where P1, P6, P9 and P10 are either done or permanently lost.

---

# Phase 2 — First agentic prototype *(≈1.5 weeks)*

**Goals.** Real tools, real ingestion, a fixed tool order (no routing yet), naive fusion, no
reliability scoring. The point is to close the loop and get a number.

**Deliverables.** Working wrappers for 3 classifiers + segmentation + the geometric suite; L1
ingestion with view detection and quality checks; templated L9 output; **Gradio demo**; first
accuracy numbers on the S1 clean setting; MedRAX baseline reproduced on the same subset.

**Dependencies.** Phase 1 schemas; datasets downloaded.

**Show.** The demo. From here on there is always something to present, which changes the character
of the internship.

---

# Phase 3 — Tool routing + evidence packets + tool profiling *(≈2 weeks)*

**Goals.** Turn the fixed pipeline into an agent. Also: run the profiling job that everything
downstream depends on.

**Deliverables.** Tool Registry with **measured** competence slices (the profiling run: every tool ×
every slice → AUROC/F1/ECE); Evidence Packet Builder with mandatory localisation; Tool Router with
the three-stage funnel; Hypothesis Manager with the hand-authored discrimination table (~40 rows);
Acquisition Planner v1 with discrete EIG; Axis-2 routing metrics including the controller gap.

**Dependencies.** Phase 2 wrappers.

**Show.** Two cases with identical images and different indications taking different tool paths, with
the EIG and rationale for each choice visible in the trace. This is the slide that proves "agentic"
means something here.

*Note: the profiling artefact produced in this phase is also the training set for research extension
R1. Build it carefully.*

---

# Phase 4 — Reliability estimation and abstention *(≈2 weeks)*

**Goals.** The differentiator. Reliability triple, conflict detection, reliability-weighted fusion,
conformal disposition, entailment gate.

**Deliverables.** Reliability Estimator with all three signals and a fitted, temperature-scaled
combiner; Conflict Detector with the three rule types; reliability-weighted log-odds fusion with a
contribution trace; Disposition Engine with a split-conformal threshold at α ∈ {0.05, 0.10, 0.20};
entailment gate; Axis-3 and Axis-4 results including risk–coverage curves.

**Dependencies.** Phase 3 profiling; a held-out calibration split reserved and untouched.

**Show.** Two plots. (i) Reliability vs. actual correctness, showing the estimator beats raw softmax.
(ii) The risk–coverage curve with the conformal guarantee marked. These two plots are the technical
core of the internship report.

---

# Phase 5 — Benchmarks and stress tests *(≈2 weeks)*

**Goals.** The full three-setting protocol. This is where the project stops being a system and starts
being a result.

**Deliverables.** Stress harness: modality ablation sweep, tool-corruption injection (6 corruption
types), image degradation, paraphrase variation, tool removal, OOD split; all eight axes measured in
S1/S2/S3; the input-ablation control published for the clean benchmark; judge validation against
100–200 human-labelled items with κ reported; complete results table with bootstrap CIs.

**Dependencies.** Phase 4; a frozen test set.

**Show.** The propagation-rate comparison (HERALD vs. reliability-layer-disabled vs. MedRAX-style
baseline, against the ≈0.62 literature reference) and the graceful-degradation curve with accuracy
and calibration on the same axes. If only two results survive from the whole project, these are the
two.

---

# Phase 6 — Governance integration hooks *(≈1 week)*

**Goals.** Prove attachability. Do not build policy.

**Deliverables.** Governance completeness audit against the §10.4 checklist with a score; a
demonstration policy (one RBAC rule, one redaction rule, one human-review routing rule) wired
through OPA or plain Python; Collector-level PHI-redaction processor with a synthetic-identifier leak
test; a written handover document for the governance teammate, including the modality-aware
requirements from §10.3.

**Dependencies.** Phase 1 hooks; Phase 5 traces to audit.

**Show.** A side-by-side of the same case rendered for two roles under two policies, with the trace
proving where the difference was introduced.

---

# Phase 7 — Research extension selection and execution *(≈4+ weeks)*

**Goals.** Convert the system into a contribution.

**Deliverables.** A decision memo choosing between R1 / R3 / R5 based on where Phase 5's numbers
showed the largest measured headroom (not on which sounded best in Phase 0); the extension
implemented; the ablation that isolates its contribution; a paper draft.

**Dependencies.** Everything. Do not start early.

**Show.** The ablation table. One row per component removed, one column per axis. This is what a
reviewer reads first and it is the artefact the whole roadmap was designed to produce.

---

# 13.1 Roadmap summary

| Phase | Weeks | Gate to pass before moving on | CV / demo artefact |
|---|---|---|---|
| 0 Capability map | 0.5 | **P0 Checkpoint**: TorchXRayVision supported-label intersection ≥ 6 labels across ≥ 3 distinct sources | The table & label matrix |
| 1 Skeleton | 1 | Empty bundle replays from response cache; regeneration stability measured; hooks honour deny | Trace viewer |
| 2 Prototype | 1.5 | End-to-end accuracy within range of a single-tool baseline | Gradio demo |
| 3 Routing | 2 | **P3 Checkpoint**: Between-slice competence variance sufficient to route on; controller gap measured | Divergent tool paths |
| 4 Reliability | 2 | **P4 Checkpoint**: Reliability AUROC beats temperature-scaled softmax; conformal guarantee holds ID | Two core plots |
| 5 Stress | 2 | Propagation rate materially below the ≈0.62 reference; halt rate 0 | Stress curves |
| 6 Governance | 1 | Completeness score ≥ target; deny honoured end-to-end | Two-role rendering |
| 7 Research | 4+ | Ablation isolates the contribution | Ablation table + draft |

**Three Go/No-Go Checkpoints:**
1. **P0 Gate (Day 1–2):** Supported-label intersection across TorchXRayVision sources is ≥6 labels across ≥3 genuinely distinct sources. *If not*, the multi-source premise fails — pivot modality or claim before building.
2. **P3 Gate:** Between-slice competence variance is large enough to route on. *If not*, routing has no headroom — report as a negative result and pivot contribution to reliability weighting alone.
3. **P4 Gate:** Reliability AUROC for predicting packet correctness beats temperature-scaled softmax. *If not*, that is the reportable negative result.

Start `NEGATIVE_RESULTS.md` on day one. Report honest findings if any gate fails.

---

# 14. Final Recommendation to the Student

You have already done the hard part that most people skip — you read the field before deciding what
to build, and your gap analysis is sharper than what is in several of the papers you analysed. The
risk now is the opposite one: knowing eleven gaps and trying to address them.

**Build first: the skeleton, then the loop, then the reliability layer.** In that order. Phase 1
looks like a wasted week because nothing clinical happens in it, and it is the week that determines
whether Phases 4, 6 and 7 are possible at all. Typed schemas, an append-only ledger, OTel spans from
the first line, and five no-op hooks. If you write the reasoning first and promise yourself you will
add provenance later, you will not — nobody ever has.

Then get to a working end-to-end run on real images with a fixed tool order as fast as you can, even
if the answers are mediocre. A running system that is wrong is worth ten times a design document that
is right, because it generates the failure logs that will tell you which of your eleven gaps is
actually the one worth your semester.

**Do not overcomplicate these four things.** *Multi-agent personas* — you do not need them; the
evidence says they usually cost more than they return, and one reasoner with many tools is both
simpler and better supported by the data. *The LLM* — MedGemma-4B is enough; if your result depends
on a bigger backbone, it is not your result. *The hypothesis space* — keep it closed at ~18 findings
plus `OTHER`; an open space makes the posterior meaningless and the evaluation impossible.
*The fusion mathematics* — reliability-weighted log-odds is enough and you can print it; the moment
you cannot print your fusion step, you have lost auditability, which was the point.

**Which gap looks strongest after the MVP works.** Your own analysis picked out Gap 3 (error cascade
in tool pipelines) and Gap 5 (safety and trust deficit), and I think you were right, but for a reason
worth making explicit: they are the same gap seen from two ends. A system that cannot estimate
whether to believe its tools cannot know when to abstain, and a system that abstains on nothing but
its own softmax is not estimating anything. **R1 — calibrated tool-trust priors used jointly for
routing and for evidence weighting — is the strongest single claim available to you**, and the
Phase 3 profiling run generates its training data as a side effect. The adjacent-domain result exists
(confidence-aware tool orchestration works in video reasoning), the medical version does not, the
benchmark methodology exists (controlled injection, with a published ≈0.62 propagation baseline), and
the evaluation is clean. That is an unusually good setup.

Keep **R3 (disclose-and-degrade)** as insurance. It is the lowest-risk of the three mandated
extensions, it will work, and if R1's learned prior fails to beat your hand-tuned scoring function
you will still have a paper. Keep **R5 (entailment-gated generation)** as the guaranteed minor
result — it can be written from your Phase 5 logs in two days, which means the project cannot end
with nothing regardless of what else happens.

**Defer to a teammate or a later paper:** the governance policy engine (build the hooks, hand over
§10.3 and §10.4, and let someone who cares about the legal interpretation own it); 3D and volumetric
imaging (the field's biggest gap and the worst possible second thing to attempt); any RL or
fine-tuning (it confounds the architectural claim you are making); EHR integration (correct as v2,
and it is what turns your synthetic missing-modality story into a real one); and real clinical
validation, which is not a semester project and should not be implied to be one.

**Two habits worth adopting on day one.** Write down the metric before you build the module — if you
cannot name how you would know it works, you do not yet understand what it does. And keep a running
`NEGATIVE_RESULTS.md`. The most valuable thing this project can produce is a well-measured account of
where reliability-aware orchestration does *not* pay for itself, and that file is where a real
contribution will accumulate while you are busy looking for one somewhere else.

The field currently has a lot of systems that are impressive and almost none that are honest. Build
the honest one. It is a smaller system and a much better paper.

---

# Appendix A — Key sources by design decision

Grouped by the section they support. Numbers quoted in this document are the cited authors' reported
figures and have not been independently reproduced.

**Benchmark validity and stress testing (§2, §8)**
- Gu et al., *The Illusion of Readiness in Health AI* — arXiv:2509.18234 (Microsoft Research). Input
  ablation, distractor perturbation, visual substitution; clinician-guided benchmark rubrics.
- Hager et al., *Nature Medicine* 2024;30(9):2613–2622 — limitations of LLMs in clinical decision-making.
- Bedi et al., *MedHELM* — arXiv:2505.23802 / *Nature Medicine* 2026, s41591-025-04151-2. Apache-2.0.
- *How to benchmark medical AI agents* — PLOS Medicine, 2026, pmed.1005170.
- Jin et al., *Hidden flaws behind expert-level accuracy of multimodal GPT-4V in medicine* —
  npj Digital Medicine 7:190 (2024).

**Agentic clinical systems and tool benchmarks (§2, §6)**
- Fallahpour et al., *MedRAX* + ChestAgentBench — arXiv:2502.02673, ICML 2025; github.com/bowang-lab/MedRAX.
- Ashraf et al., *MedCTA: A Benchmark for Clinical Tool Agents* — arXiv:2606.11702 (June 2026).
- Lee, Yoon & Choi, *CXReasonAgent* — arXiv:2602.23276 (KAIST, Feb 2026); CheXStruct-based deterministic evidence.
- *RadFabric* — arXiv:2506.14142 (MCP-based CXR multi-agent).
- Jiang et al., *MedAgentBench* — arXiv:2501.14654; NEJM AI 2025. FHIR-compliant virtual EHR, 300 tasks.
- *FHIR-AgentBench* — arXiv:2509.19319 (Synthea-grounded).
- Schmidgall et al., *AgentClinic* — npj Digital Medicine 2026, s41746-026-02674-7.
- Curated lists: github.com/Nanboy-Ronan/awesome-medical-imaging-agents ;
  github.com/AgenticHealthAI/Awesome-AI-Agents-for-Healthcare.

**Tool reliability, routing, and error cascades (§3-P2, §5.8, §9-R1, §9-R6)**
- *The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents* —
  ACL 2026 Long; arXiv:2601.07264. Evidence tools induce overconfidence; verification tools reduce it.
- *Confidence-Aware Tool Orchestration for Robust Video Understanding* (Robust-TO) — arXiv:2606.26904.
  The "Blind Trust Problem"; shared evidence format with calibrated reliability scores.
- *Evaluating Tool-Using Language Agents ... AgentProp-Bench* — arXiv:2604.16706. Propagation ≈0.62;
  detection and recovery independent (ρ=0.126, p=0.747); substring judging at κ=0.049.
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST) — arXiv:2503.13657, NeurIPS 2025.
- *When to Trust Tools? Adaptive Tool Trust Calibration* — arXiv:2604.08281.
- *Silent Failure in LLM Agent Systems* — arXiv:2606.08162 (behaviour routing deficiency).

**Abstention, calibration, conformal risk control (§3-P4, §5.13, §9-R2)**
- Angelopoulos et al., *Conformal Risk Control* — ICLR 2024; *Learn Then Test* — Ann. Appl. Stat. 2025;19(2):1641–1662.
- Abbasi-Yadkori et al., *Mitigating LLM Hallucinations via Conformal Abstention* — NeurIPS 2024 WS.
- *CARE: A Conformal Safety Layer for Medical Summarization* — arXiv:2606.08969 (som-shahlab).
- *Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty* — arXiv:2601.12471.
- *Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution
  shift* — Scientific Reports 2026, s41598-026-40637-w.
- *Cascaded Language Models for Cost-Effective Human-AI Decision-Making* — NeurIPS 2025, arXiv:2506.11887.
- *Entropy Alone is Insufficient for Safe Selective Prediction in LLMs* — arXiv:2603.21172.
- *RadFlag: Black-Box Hallucination Detection for Medical VLMs* — arXiv:2411.00299.

**Missing modality and graceful degradation (§3-P3, §5.12, §9-R3)**
- *Transparent Early ICU Mortality Prediction ... Per-Case Modality Attribution* — arXiv:2511.15847
  (mask-and-renormalise; AUPRC −0.09/−0.11 with calibration retained).
- *Handling and Interpreting Missing Modalities in Patient Clinical Trajectories* — arXiv:2604.18753.
- *Meta-Modal Agent: Sequential Evidence Routing for Missing-Modality Candidate Reranking* —
  arXiv:2605.25007 (reconstruction as a liability under heavy missingness).
- *MREF-AD* — arXiv:2512.10966 (gating mask + renormalisation).

**Differential diagnosis (§2.4, §5.9, §9-R4)**
- McDuff, Tu et al., *Towards Accurate Differential Diagnosis with LLMs* (AMIE) — arXiv:2312.00164; *Nature* 2025.
- *Toward Accurate and Actionable Differential Diagnosis with Lean LLM Orchestration* —
  medRxiv 2025.10.28.25338335 (Capture@K).
- *Preliminary analysis of the impact of lab results on LLM-generated differential diagnoses* —
  npj Digital Medicine 2025, s41746-025-01556-8 (up to +30% from labs).
- *HeartAgent* — arXiv:2603.10764 (scaffolding > backbone).

**Report and evidence-fidelity metrics (§5.14, §8-Axis 8)**
- Bannur et al., *MAIRA-2 / RadFact* — arXiv:2406.04449.
- Ostmeier et al., *GREEN* — arXiv:2405.03595 (note documented length bias).
- Smit et al., *CheXbert* (2020); Jain et al., *RadGraph* (2021); Delbrouck et al. (2022); Yu et al., *RadCliQ* (2023).

**Backbones, tools, and data (§6, §7)**
- Sellergren et al., *MedGemma Technical Report* — arXiv:2507.05201 (MedGemma 4B/27B, MedSigLIP 400M).
- Cohen et al., *TorchXRayVision* — arXiv:2111.00595, MIDL 2022; github.com/mlmed/torchxrayvision.
- Nguyen et al., *VinDr-CXR* — Scientific Data 2022, s41597-022-01498-w.
- Bustos et al., *PadChest* (UMLS-mapped taxonomy); Chambon et al., *CheXpert Plus* — arXiv:2405.19538.
- *CXR-LT 2026 Challenge Overview* — arXiv:2602.22092 (PadChest+NIH; 6 unseen OOD classes).
- *CheXmask* — arXiv:2307.03293.
- PhysioNet Credentialed Health Data License 1.5.0 and the PhysioNet guidance on responsible use of
  MIMIC with LLMs and online services (third-party API prohibition).

**Governance and audit (§3-P9, §10)**
- EU AI Act, Regulation (EU) 2024/1689 — Articles 9, 10, 11, 12, 14, 19, 26, 50, 72; Annexes III, IV.
- Digital Omnibus on AI: Commission proposal 19 Nov 2025; provisional agreement 7 May 2026;
  Parliament 16 June 2026; **Council final approval 29 June 2026**. Annex III high-risk → 2 Dec 2027;
  Annex I → 2 Aug 2028; Article 50 transparency unchanged at 2 Aug 2026 (50(2) → 2 Dec 2026 for
  systems already on market).
- OpenTelemetry GenAI Semantic Conventions (semconv v1.41, Development stability as of mid-2026);
  `invoke_agent` / `chat` / `execute_tool` spans; content in span events.
- ISO/IEC 42001; IEC 62304; prEN ISO/IEC 24970 (AI system event logging, in development).
- *Decision Trace Schema for Governance Evidence* — arXiv:2604.09296; *DEMM-Bench* — arXiv:2606.20634;
  *Runtime Governance for AI Agents: Policies on Paths* — arXiv:2603.16586.

**User-supplied source documents**
- `internship_presentation.md` — 27-paper Category 1.1 deep analysis; the 11-gap taxonomy.
- `ERC_survey.md` — coordination-vs-robustness evaluation distinction; the six ERC principles;
  the conflict-preservation gap.
- `ComparisionTable.md` — the empty Memory × Governance intersection.

---

# Appendix B — Repository layout

```
herald/
 docker-compose.yml              # app · vllm · langfuse · mlflow
 conf/                           # hydra: backbones, tools, thresholds, datasets
 herald/
    schemas/                    # Pydantic v2 — the contract (P6). Change with care.
       evidence.py             # EvidencePacket, MissingEvidence, ReliabilityScore
       hypothesis.py           # Hypothesis, HypothesisLedger
       case.py                 # CaseState, AvailabilityMap, TaskFrame
       tools.py                # CapabilityCard, ToolResult, AcquisitionAction
       disposition.py          # Disposition, Conflict, GovernanceDecision
    graph/                      # LangGraph nodes + edges + checkpointer
    ingest/                     # L1: modality detection, quality checks, de-ID stub
    triage/                     # L2: query parser, scope gate, hypothesis seeding
    registry/                   # L3: capability cards + competence profiles
    routing/                    # L4: router funnel + acquisition planner (EIG)
    tools/                      # L5: wrappers — xrv_*, segment_*, geometry_*, retrieve_*
    reliability/                # L6: packet builder, estimator, conflict detector
    fusion/                     # L7: log-odds fusion + contribution trace
    disposition/                # L8: conformal thresholds + decision rules
    generate/                   # L9: templating, prose, entailment gate
    audit/                      # L10: OTel wiring + Case Bundle writer
    governance/                 # L11: five hooks (no-ops in v1) + context types
 eval/
    axes/                       # one module per evaluation axis (§8)
    stress/                     # ablation · corruption injection · OOD · paraphrase
    judges/                     # entailment gate + LLM jury + κ validation harness
    report/                     # results table builder (§8.3 template)
 profiling/                      # Phase 3 job → competence slices (also R1 training data)
 app/                            # Gradio explorer
 tests/
    test_schemas.py
    test_tool_removal.py        # pipeline survives each tool being disabled
    test_replay_determinism.py  # bundles re-execute to identical dispositions
    test_hook_coverage.py       # every irreversible action is preceded by a hook
 NEGATIVE_RESULTS.md             # start it on day one
 docs/
     capability_map.md           # Phase 0 deliverable: failure → module → metric
     datasets.md                 # licences, access status, hashes
     governance_handover.md      # Phase 6 deliverable for the second teammate
```

*End of blueprint.*
