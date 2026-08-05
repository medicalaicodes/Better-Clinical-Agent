# Literature Survey — Part 1
# Tracing the Lineage of MedAgent-Pro: Improvements in Agentic AI for Medical Diagnosis

*Compiled July 25, 2026. This is Deliverable 1 of the low-effort/high-reward starting phase for the research programme on "Agentic AI Frameworks in Medicine with Complete Governance."*

---

# 0. Method and honesty note

I searched Semantic Scholar, arXiv, ACM DL, ResearchGate, TechRxiv, Preprints.org, and general web sources for (a) papers that explicitly cite Wang et al.'s *MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow* (arXiv:2503.18968, submitted March 2025) and (b) papers describing themselves as improving on, extending, or positioning against it. I could **not** run a full citation-graph export (no Semantic Scholar API tool available in this environment), so the list below is a **representative, not exhaustive**, sample built from search coverage. A few honesty caveats before the list:

- **Venue status varies a lot.** A few of these are peer-reviewed (ACL/EMNLP Findings, ACM Multimedia, and — notably — one in *Nature*). Most are **arXiv preprints, not yet peer-reviewed**, some very recent (2026). I flag venue status for each so you don't over-weight preprint claims in your own paper.
- **"Improvement" claims are self-reported** by the citing papers' own abstracts/intros unless I note otherwise — I have not independently reproduced any benchmark numbers.
- Several papers cite MedAgent-Pro only as **related-work context** (a baseline to be compared against or a data point in a survey), not as a system they build on top of. I've separated these two categories below.
- This is arXiv:2503.18968 v3 (July 2025) — the version you uploaded. Some citing papers cite an earlier v1/v2 with a slightly different title ("Towards Multi-modal Evidence-based..." vs "Towards Evidence-based Multi-modal..."); this is the same paper.

---

# 1. What MedAgent-Pro actually claims (baseline recap)

For calibration, MedAgent-Pro's core contributions were: (1) a **two-level hierarchy** — disease-level RAG-based guideline planning + patient-level step-by-step reasoning; (2) **evidence-based step gating** (Continue/Terminate/Complete) so unreliable intermediate outputs halt the pipeline rather than propagating; (3) **tool-integrated quantitative analysis** (segmentation/grounding/coding models) rather than relying on VLM qualitative judgment alone; (4) validation across 10+ modalities, 20+ anatomies, 50+ diseases, with clinician evaluation. Its own stated limitations: dependence on availability of visual tools per domain, and residual hallucination risk in the qualitative (VLM-driven) reasoning steps. This is the honest baseline against which "improvement" should be judged — most citing papers target exactly these two weak points.

---

# 2. Papers that directly build on or improve MedAgent-Pro's mechanisms

# 2.1 Reasoning-structure improvements (attacking the "flat sequential plan" limitation)

**Tree-of-Reasoning (ToR)** — Peng, Cui, Xie, Cai, Li. *ACM Multimedia 2025* (peer-reviewed, pp. 1744–1753; also arXiv:2508.03038).
MedAgent-Pro's patient-level reasoning is a **linear chain** of steps with a single terminate/continue/complete gate. ToR's explicit critique is that linear/single-agent reasoning chains "lack sufficient reasoning depth" and lose information or make logical jumps on complex, multi-disease differential cases. Their fix: a **tree structure** that records multiple candidate reasoning branches and their supporting evidence simultaneously, plus a **cross-validation mechanism across multiple agents** to reconcile disagreement before committing to a branch — essentially replacing MedAgent-Pro's single evidence-gated chain with a searchable, auditable reasoning graph. This is a fairly direct structural upgrade for differential diagnosis (multiple-choice-style, multi-disease) settings, which MedAgent-Pro's REFUGE2/MITEA/MIMIC/NEJM benchmarks did not stress-test.

**"A medical multimodal diagnostic framework integrating vision-language models and logic tree reasoning"** (arXiv:2512.21583, Dec 2025). Extends the same logic-tree idea explicitly into the multimodal VLM setting MedAgent-Pro operates in, rather than text-only differential diagnosis.

**MAM (Modular multi-agent framework for multimodal medical diagnosis via role-specialized collaboration)** — Zhou, Song, Shen. *Findings of ACL 2025* (peer-reviewed, pp. 25319–25333). Decomposes the single "VLM-as-everything" role in MedAgent-Pro into explicitly specialized collaborating roles, targeting the same qualitative-reasoning hallucination risk MedAgent-Pro's own limitations section names.

# 2.2 Evidence-grounding improvements (attacking hallucination in the qualitative/VLM steps)

**EviAgent** (arXiv:2603.13956, 2026) — Evidence-Driven Agent for Radiology Report Generation. Explicitly frames MedAgent-Pro (and MMedAgent) as "central planners that dispatch specialized tools" but argues the field still suffers a "weak or absent connection between diagnostic conclusions and supporting visual evidence" — i.e., MedAgent-Pro's evidence-gating function φ is VLM-implemented and therefore inherits VLM unreliability rather than solving it. EviAgent proposes tighter per-claim visual-evidence linkage specifically for report generation.

**Evidence-based diagnostic reasoning with multi-agent copilot for human pathology** (Chen, Weishaupt, Williamson et al., arXiv:2506.20964, 2025). Transplants the "evidence-based, step-verified reasoning" philosophy of MedAgent-Pro into computational pathology (whole-slide images), a modality MedAgent-Pro's own toolset (segmentation/grounding models tuned for radiology/ophthalmology) does not cover well.

**An Agentic System for Rare Disease Diagnosis with Traceable Reasoning ("DeepRare")** — Zhao, Wu, Fan, Xie et al. First posted arXiv:2506.20430 (June 2025); **published in *Nature*, Feb 2026** (DOI 10.1038/s41586-025-10097-9) — this is the most credentialed publication in this entire citation set. DeepRare pushes MedAgent-Pro's "RAG-grounded, evidence-linked" idea further: every intermediate reasoning step is linked to a **verifiable external medical evidence citation**, evaluated across 2,919 diseases and 9 datasets, with **95.4% clinician agreement** on reasoning-chain validity (vs. hallucinated/irrelevant citations occasionally observed). It integrates 40+ specialized tools (vs. MedAgent-Pro's smaller segmentation/coding toolset) and targets differential diagnosis of rare disease specifically — arguably the highest-stakes, most citation-sensitive niche where MedAgent-Pro-style tool orchestration could plausibly reach regulatory-grade rigor. A follow-up, **"End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning"** (Zheng et al., 2025), goes further by training the retrieval-reasoning loop end-to-end rather than orchestrating frozen tools — a genuine methodological step past MedAgent-Pro's fixed-toolbox design.

**LiteOdyssey** (arXiv:2606.16149, 2026) — positions itself as a **lightweight** alternative to DeepRare-class systems for rare-disease diagnosis, explicitly trading off some of the tool-orchestration overhead for interpretability at lower compute cost — relevant if your governance paper cares about deployability in resource-constrained settings, not just accuracy.

# 2.3 Efficiency / cost-aware triage (attacking MedAgent-Pro's "always run the full pipeline" design)

**ConfAgents** — Zhao, Zhu, Wang, Wang, Gao, Ma. arXiv:2508.04915 (Aug 2025; preprint). Explicit critique implied: MedAgent-Pro (and similar frameworks) invoke the full multi-step tool pipeline for every case regardless of difficulty, which is computationally and financially expensive at scale. ConfAgents adds a **conformal-prediction-based confidence triage**: easy cases are resolved by a single fast agent pass; only cases where conformal uncertainty is high get escalated to full multi-agent deliberation with dynamic knowledge retrieval. This is directly relevant to a *governance* paper because conformal prediction gives a **statistically calibrated, distribution-free uncertainty guarantee** — a much stronger auditability property than MedAgent-Pro's VLM-implemented, uncalibrated φ(·) reliability function.

# 2.4 Memory and self-evolution (attacking MedAgent-Pro's "static, hand-authored plan per disease" design)

**"Experience Makes Skillful": Self-Evolving Skill Memory for medical agents** (arXiv:2606.09365, 2026). Directly cites MedAgent-Pro (and MedCo, Agent Workflow Memory) as prior art whose diagnostic plans, once authored via RAG at the disease level, are static across all future patients of that disease. This paper's contribution is a **skill-memory mechanism** that lets the agent's diagnostic strategy improve over time from case experience, rather than being fixed once retrieved from the guideline knowledge base — a genuine gap MedAgent-Pro's own paper does not address (their plan generation is one-shot per disease, never updated post-deployment).

**HealthFlow** (Zhao et al., referenced alongside ConfAgents' arXiv listing, and independently at arXiv:2508.04955-adjacent ID space) — a "self-evolving AI agent" with meta-level policy evolution for broader healthcare-research tasks (not just diagnosis), introducing **EHRFlowBench** for evaluation. Relevant as a template for how self-evolution could be added to a MedAgent-Pro-style pipeline without losing the evidence-based gating.

**ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory** (arXiv:2603.26182, 2026) and **Joint Optimization of Reasoning and Dual-Memory for Self-Learning Diagnostic Agent** (arXiv:2604.07269, 2026, which also cites Tree-of-Reasoning and DeepRare) both add explicit short-term/long-term memory architectures to the orchestration loop — a design axis entirely absent from MedAgent-Pro.

# 2.5 Domain-specific extensions (porting MedAgent-Pro's paradigm to modalities it didn't cover)

- **SpineBench / SpineMed-450k** (arXiv:2510.03160) — cites MedAgent-Pro while building a spine-imaging-specific benchmark and corpus; targets a musculoskeletal-imaging gap in the original evaluation (which covered ophthalmology, cardiology, chest X-ray, and general NEJM cases but not spine/orthopedic imaging).
- **TopoAgent** (arXiv:2606.29763) — an agentic framework for automated topology learning in medical imaging; cites MedAgent-Pro as prior evidence-based agentic-workflow precedent while targeting a structural/topological image-analysis task outside MedAgent-Pro's segmentation-classification toolbox.
- **Med-VRAgent** (arXiv:2510.18424) — Medical Visual Reasoning-Enhanced Agents; addresses the same VLM-hallucination-in-radiology/pathology problem via visual prompting and plan-then-generate decoupling, situating itself in the same "agentic + tool-augmented" lineage.

# 2.6 Auditing and full-workflow benchmarking (attacking MedAgent-Pro's evaluation scope)

**MedOpenClaw and MedFlowBench: Auditing Medical Agents in Full-Study Workflows** (arXiv:2603.24649, 2026). Cites MedAgent-Pro and MMedAgent as tool-orchestration precedents, but argues existing evaluations (MedAgent-Pro included) test isolated diagnostic *decisions*, not **full clinical-study workflows** with multi-turn, multi-session, longitudinal state — a benchmarking-methodology gap directly relevant to your eventual governance framework, since regulators will care about full-workflow auditability, not single-shot accuracy.

# 2.7 Reinforcement-learning-trained multimodal orchestration

**MMedAgent-RL** (Xia, Wang, Peng et al., arXiv:2506.00555, 2025) — "Optimizing multi-agent collaboration for multimodal medical reasoning" via RL rather than MedAgent-Pro's hand-designed orchestration/planning rules. This represents a shift from *rule-based* orchestration (MedAgent-Pro's JSON-plan + operation-description matching, Eq. 2–3 in the paper) toward *learned* orchestration policies — a meaningful methodological fork worth discussing in your related-work section, since learned orchestration raises new governance questions (policy drift, non-determinism) that rule-based orchestration does not.

---

# 3. Papers that cite MedAgent-Pro mainly as comparative context (not building on it)

These use MedAgent-Pro as a **named example of the current state of the art** in evidence-based agentic diagnosis, generally within a broader taxonomy or survey, rather than extending its architecture:

- **MedCoG** (arXiv:2602.07905, 2026) — cites MedAgent-Pro within a broader lineage (MedPrompt → MedAgents → ReConcile → MDAgents → KAMAC → MedAgent-Pro) while proposing a meta-cognitive inference-density regulation mechanism for a different axis of the problem (compute efficiency of reasoning, not tool orchestration).
- **"From Conflict to Consensus" (MA-RAG)** (2025) — lists MedAgent-Pro as a related prior system while proposing multi-round agentic RAG with self-consistency-as-signal.
- **Evaluating Multi-Agent LLM Architectures for Rare Disease Diagnosis** (arXiv:2603.06856, 2026) — benchmarks MedAgent-Pro-style and other architectures against each other empirically for rare-disease tasks.
- **MultiDx** (arXiv:2604.24186, 2026) — cites MedAgent-Pro-adjacent work (ConfAgents) in framing multi-source knowledge integration for diagnostic reasoning.
- Google's **AMIE** system (Saab et al., *Nature Medicine* 2025, peer-reviewed) is repeatedly cited alongside MedAgent-Pro in the newer papers above as the other major reference point for "state-aware, phase-based reasoning that outperforms clinicians in evaluation" — worth reading directly since it is the most clinically validated system in this space and is **not** from the MedAgent-Pro lineage (different architecture, industry rather than academic origin, conversational rather than tool-orchestration paradigm).

---

# 4. Field-level surveys that situate MedAgent-Pro (use these to build your related-work section fast)

| Survey | Venue/status | What it's useful for |
|---|---|---|
| *A Survey of LLM-based Multi-agent Systems in Medicine* | OpenReview/TechRxiv/Preprints.org, Oct 2025 (ICLR 2026 submission — **withdrawn**, so treat as unreviewed) | Proposes a taxonomy along team composition, medical knowledge augmentation, agent interaction. Explicitly flags that "designing agent profiles that balance realism, adaptability... [and] transparent, verifiable, interpretable" updates is an open problem — directly your governance angle. |
| *A Survey of LLM-based Agents in Medicine: How far are we from Baymax?* | arXiv:2502.11211 (preprint) | Application-scenario taxonomy (diagnosis, triage, biomedical research, patient simulation); good for scoping your intro. |
| *Large Language Model Agents for Biomedicine: A Comprehensive Review...* | MDPI *Information*, Oct 2025 (peer-reviewed journal) | Broadest peer-reviewed survey found; architectures, methodologies, real-world use cases. |
| *Multi-agent systems for clinical decision support: A systematic review* | ScienceDirect (peer-reviewed), reviewed 42 studies 2016–2025 | Pre-dates the LLM-agent boom somewhat in scope but gives you a defensible PRISMA-style systematic-review methodology to imitate for your own survey. |
| *A Comprehensive Survey of Agentic AI in Healthcare* (Xu, Li, Chen et al.) | Authorea Preprints, 2025 | Broader than diagnosis — covers agentic AI across the healthcare stack (admin, research, clinical). |

---

# 5. Governance- and safety-specific literature (directly feeds your core research question)

This is thin compared to the technical/architectural literature above — which is itself an important finding for your gap analysis (see §6).

- **"Enhancing Clinical Decision-Making: Integrating Multi-Agent Systems with Ethical AI Governance"** (arXiv:2504.03699, 2025). The most directly on-topic paper found: explicitly pairs a MAS (referencing MDAgents) with an ethical-governance layer. Worth reading in full before you draft your own framework, both to build on it and to differentiate your contribution.
- **MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems** (arXiv:2505.20824, 2025). Addresses adversarial/safety risk specifically in *multi-agent* medical LLM systems (as opposed to single-model safety) — relevant since MedAgent-Pro-style architectures multiply the attack surface (RAG agent, tool agents, orchestrator).
- **Agentic AI Governance and Lifecycle Management in Healthcare** (arXiv:2601.15630, 2026). Proposes lifecycle governance (non-human identity, audit trails, Monte Carlo risk simulation) for agentic AI in healthcare organizations broadly — more operations/deployment-governance than model-architecture-governance, but a good structural template.
- **"Agentic AI in Healthcare: A Comprehensive Survey of Foundations, Taxonomy..."** (Banerjie, Zhu, Freeman et al., Nov 2025) — cited by the lifecycle-governance paper above; likely the single most relevant survey to track down in full for your governance section.

**Regulatory/standards context** (not academic papers, but load-bearing for a governance chapter):
- EU AI Act (Regulation (EU) 2024/1689) — Articles 14 (human oversight) and 15 (accuracy/robustness/cybersecurity) apply to autonomous agents in high-risk domains (medical diagnosis is high-risk under the Act's classification).
- Singapore IMDA *Model AI Governance Framework for Agentic AI* (Jan 2026) — first comprehensive agentic-AI-specific governance framework; requires verifiable digital identity and audit trail per acting agent.
- NIST AI Risk Management Framework (Govern/Map/Measure/Manage) + NIST AI Agent Standards Initiative (Feb 2026).
- US: no federal agentic-AI statute as of mid-2026; 250+ state-level health-AI bills introduced in 2025 (patchwork regulatory landscape — Colorado SB 24-205 is the first broad "high-risk AI" state statute, effective June 30, 2026).
- ISO/IEC 42001:2023 (AI management systems standard).
- HHS/FDA/CMS/CDC/NIST: FDA's SaMD (Software as a Medical Device) framework and the CDC's March 2026 "Considerations for Agentic Research in Public Health" guidance are the most directly medicine-specific agentic-AI regulatory documents to date.

*(I have not deeply verified every regulatory citation above against primary legal text — treat the summaries as pointers to go read primary sources, not as legal advice; flag this explicitly if you cite them.)*

---

# 6. Honest synthesis: what has actually been improved, and what hasn't

**What the field has clearly improved since MedAgent-Pro (March 2025):**
1. **Reasoning topology** — linear chains → trees/graphs with cross-validation (ToR and successors).
2. **Evidence traceability** — from "VLM says it's reliable" to per-claim external citation linkage with measured clinician agreement (DeepRare's 95.4%).
3. **Cost/efficiency** — from "always run the full pipeline" to calibrated, uncertainty-triggered escalation (ConfAgents' conformal triage).
4. **Adaptivity over time** — from static, one-shot disease plans to self-evolving skill/strategy memory.
5. **Domain coverage** — spine, pathology, topology-heavy imaging now have MedAgent-Pro-style pipelines; the original only covered ophthalmology/cardiology/chest-X-ray/general NEJM cases.
6. **Orchestration paradigm diversity** — rule-based (MedAgent-Pro) vs. RL-trained (MMedAgent-RL) vs. conformal-triaged (ConfAgents) orchestration are now all represented, giving you a genuine design-space taxonomy to build your framework on.
7. **Clinical/regulatory credibility** — DeepRare reaching *Nature* and AMIE reaching *Nature Medicine* shows the field can clear top-tier peer review, which matters if your eventual paper wants to be taken seriously by a clinical audience, not just an ML-conference audience.

**What has *not* been meaningfully improved — the real gaps for your novel contribution:**
1. **Governance is almost entirely decoupled from architecture.** Every technical paper above optimizes accuracy, cost, or traceability; only two or three papers even attempt to couple governance/ethics to the architecture itself, and none of the technical papers (ToR, DeepRare, ConfAgents, MedAgent-Pro itself) specify who is accountable when the RAG agent retrieves an outdated guideline, when a tool model's segmentation is silently wrong, or when the orchestrator's plan skips a step it shouldn't have. This is your opening.
2. **No standardized audit trail format across frameworks.** Each system logs its reasoning differently (JSON plans, evidence trees, skill memories, conformal sets) — there's no shared schema a regulator or hospital auditor could use across vendors. This is a concrete, buildable contribution (a governance-oriented interoperable audit schema) rather than a purely conceptual one.
3. **Multi-agent-specific safety is nascent.** MedSentry is essentially the only paper directly studying adversarial/failure risk *of the multi-agent interaction itself* (e.g., one compromised or malfunctioning tool agent poisoning the orchestrator's final decision) — this is distinct from single-model hallucination and under-studied.
4. **No paper evaluates liability/accountability allocation** when the pipeline includes third-party tools (MedSAM, Copilot-style coding modules, external RAG knowledge bases) — a governance question MedAgent-Pro's architecture (which explicitly plugs in external segmentation/grounding models) makes structurally unavoidable but which none of the citing papers address.
5. **Regulatory alignment is descriptive, not architectural.** The EU AI Act / NIST RMF / Singapore framework papers describe *what* governance should require (oversight, audit trails, risk tiering) but none of the *diagnostic-architecture* papers (MedAgent-Pro, ToR, DeepRare, ConfAgents) were designed with those specific legal requirements as design constraints from the start. Retrofitting governance onto an existing architecture vs. designing "governance-native" agentic diagnosis is a genuinely open, high-value research question.
6. **Longitudinal/full-workflow auditability remains benchmarked but not designed for.** MedFlowBench identifies the evaluation gap; no architecture paper yet treats "the audit trail is a first-class output, not a logging side-effect" as a design principle.

This gap set (# 1, # 2, # 3, # 5 especially) is where a "novel Agentic AI framework in Medical with complete Governance" paper can genuinely stake out new territory rather than re-deriving what MedAgent-Pro, ToR, or DeepRare already did.

---

# What's next

This document covers the **literature survey and gap analysis** — step 1 of your plan. Natural next steps, each of which I can do as a separate low-effort-first pass and then deepen:

1. **Deep-read pass** on the 4–5 highest-value papers for your specific angle: DeepRare (Nature), ConfAgents, the Ethical-AI-Governance MAS paper, and the Agentic-AI-Governance-Lifecycle paper — pull their exact methods/metrics rather than abstract-level summaries.
2. **A structured comparison table** (architecture × evidence-grounding mechanism × memory × governance features × evaluated modalities) across all systems named above — this becomes a real Table 1 for your paper.
3. **The roadmap** you asked for — turning gaps # 1–# 6 above into a phased research plan (a governance-native architecture proposal, an audit-schema contribution, a multi-agent safety evaluation, etc.), sequenced by effort and publishability.

Tell me which of these three you want next, or if you'd like me to just proceed straight into the comparison table.