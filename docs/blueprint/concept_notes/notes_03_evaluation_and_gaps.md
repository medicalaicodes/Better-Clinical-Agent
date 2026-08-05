# Research Notes — Batch 3: Evaluation Methodology & Confirmed Gaps

# MedCTA: A Benchmark for Clinical Tool Agents (arXiv 2606.11702, 10 Jun 2026)
- 107 real-world clinician-verified clinical tasks, executable trajectories over **5 deployed tools**,
  multimodal inputs (radiology images, pathology slides, reports). "Step-implicit" tasks — the
  agent is not told which steps to take.
- **Process-aware** evaluation axes: tool selection, argument validity, execution stability,
  trajectory fidelity, outcome quality.
- 18 open + closed multimodal models benchmarked. Findings:
  - reliable clinical tool use is **unsolved**;
  - **gold-standard (oracle) tool routing exposes a large "controller gap"**;
  - most failures come from **protocol instability, under-calling, and weak local grounding**,
    NOT from final synthesis.
  → Confirms: (1) the router is the bottleneck, not the writer; (2) "controller gap vs oracle
    routing" is an accepted metric; (3) under-calling (failing to invoke a needed tool) is a
    first-class failure mode we must measure.

# AgentProp-Bench (arXiv 2604.16706) — error propagation, quantified
- 2,000 tasks / 2,300 traces / 4 domains / 9 production LLMs / 100 human-validated labels.
- Method: **controlled injection** of errors (rather than waiting for natural ones) → enables
  causal measurement of propagation.
- RO1 Judge reliability: **substring matching agrees with humans at κ = 0.049 (chance level)**;
  a **three-LLM ensemble judge reaches κ = 0.432** with conservative bias.
  → MANDATE: validate any automatic judge against a human-labelled subset before reporting numbers.
- RO2 Propagation: a **parameter-level injection propagates to a wrong final answer with
  human-calibrated probability ≈ 0.62 (range 0.46–0.73 across models)**.
  → Baseline expectation for our "corrupted tool" stress test.
- RO2b: **rejection rate and recovery rate are statistically independent** (Spearman ρ = 0.126,
  p = 0.747). Robustness to tool errors is **at least two-dimensional**.
  → MANDATE: report *detection* and *recovery* as separate metrics. A system that notices a bad
    tool but can't route around it is a different (and commoner) failure than one that never notices.

# "How to benchmark medical AI agents" — PLOS Medicine, 9 July 2026 (pmed.1005170)
- Very recent methodological framing piece. Notes agents will act in **clinician-supporting roles
  rather than replacing them** for the foreseeable future.
- Cites: AgentClinic (now *npj Digital Medicine* 2026, s41746-026-02674-7); Bean et al.,
  *Nature Medicine* 2026;32(2):609–15 — randomized preregistered study of LLM reliability as
  medical assistants for the general public; Hager et al. *Nat Med* 2024;30(9):2613–22;
  Chiu, Pitis & van der Schaar, NeurIPS 2025 D&B — simulated viva voce exams for clinical reasoning.

# Orchestration framework landscape (mid-2026)
- **LangGraph**: orchestration as an explicit state machine — pure nodes, typed edges, shared
  typed state (TypedDict + reducers), durable checkpointing, time-travel debugging, native
  human-in-the-loop interrupts, LangSmith/OTel tracing. Consistently described as the default for
  regulated / auditable / stateful production workflows. Learning curve ~10–14 engineer-days.
- **CrewAI**: role/goal/backstory hierarchy; fastest prototype; murkier debugging. Built on LangChain.
- **AutoGen**: group-chat paradigm; **Microsoft has moved it to maintenance** in favour of the
  unified Microsoft Agent Framework (AutoGen + Semantic Kernel). Do not start new projects on it.
- **OpenAI Agents SDK**: lowest friction if OpenAI-locked; no model flexibility → unusable under
  PhysioNet DUA constraints.
- Repeated practitioner warning: many "agent" problems are better solved by a well-structured chain
  with a few tool calls; agents add non-determinism, debugging cost, and money. Also:
  **set termination conditions tightly** (e.g. max_consecutive_auto_reply=3) — matches MAST failure
  mode 1.5 "unaware of termination conditions".

# Consolidated statement of the gap this blueprint targets
Confirmed by independent sources, none of which solve it together for clinical multimodal agents:
1. Tools are trusted blindly (Robust-TO "Blind Trust Problem"; MedAgent-Pro's −16.5% F1 from one
   weak tool; CARE's over-editing coordinator).
2. Tool augmentation *changes* calibration in tool-type-dependent ways — evidence tools induce
   overconfidence, verification tools reduce it (Confidence Dichotomy, ACL 2026). Nobody exploits
   this deliberately in a clinical pipeline.
3. Routing is the empirical bottleneck and there is a large measured oracle-vs-actual controller
   gap (MedCTA 2026; ToolSelect 72% vs 96% oracle from the user's own survey).
4. Errors propagate at ≈0.62 once injected; detection and recovery are independent capabilities
   (AgentProp-Bench). No medical agent reports either.
5. Benchmarks reward shortcuts; models answer correctly with the image removed (Illusion of
   Readiness). No medical agent reports input-ablation sensitivity.
6. Missingness is handled by halting (Cerebra) or by imputation that fabricates unsupported
   evidence (Meta-Modal Agent warning). "Disclose and degrade with preserved calibration" is
   demonstrated in supervised fusion (arXiv 2511.15847) but not in an agent.
7. Abstention in medicine is studied for single models (conformal abstention, CARE conformal safety
   layer, Knowing When to Abstain) but **not at the coordination level**, and not as a *typed,
   actionable request for a specific missing datum*.
8. Governance: Art. 12-style logging is a legal requirement with no engineering schema; OTel GenAI
   semconv is the natural carrier but is still Development-stability and covers telemetry, not
   clinical evidence semantics.

→ The blueprint's contribution is the *integration*: a typed evidence ledger with calibrated
  per-packet reliability, a hypothesis-level fusion layer that can preserve conflict, a typed
  disposition (answer / caveat / request / escalate), and an entailment-gated narrative — all
  emitted as OTel-compatible spans so the governance layer is a later attachment, not a rewrite.
