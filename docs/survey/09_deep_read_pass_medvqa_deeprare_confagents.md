# Literature Survey — Part 2
# Deep-Read Pass: DeepRare, ConfAgents, "Ethical AI Governance MAS," and UALM

*Compiled July 25, 2026. Deliverable 2 — full-text extraction (not abstract-level) of methods, metrics, and honest limitations for the four papers you selected. One bibliographic caveat up front: arXiv:2508.04915's official abstract page currently displays a different paper ("HealthFlow") than the PDF served at that same ID, which contains "ConfAgents" — same author group (Zhao, Zhu, Wang, Wang, Gao, Ma), so this looks like an arXiv metadata/versioning mix-up rather than two different papers. I read the actual PDF content below (confirmed against GitHub and ResearchGate, which independently describe the same ConfAgents mechanism), so the methods/metrics here are reliable — just flag this if you cite the arXiv ID directly, and link to the GitHub (github.com/PKU-AICare/ConfAgents) as a backup reference.*

---

# 1. DeepRare — *An Agentic System for Rare Disease Diagnosis with Traceable Reasoning*
Zhao, Wu, Fan, Xie et al. arXiv:2506.20430 (June 2025) → **published in *Nature*, Feb 2026** (DOI 10.1038/s41586-025-10097-9)

# Exact architecture
Three-tier, **Model Context Protocol (MCP)-inspired** design:
1. **Central host** (default: locally-deployed **DeepSeek-V3**, 671B params, run on 16 Ascend 910B cards for data-security reasons) — holds a **memory bank** (initialized empty, incrementally updated), orchestrates all downstream steps.
2. **Agent servers** — six specialized servers, each wrapping a specific toolset: *phenotype extractor* (free-text → HPO terms, via two-stage LLM prompting + BioLORD embedding match, cosine similarity threshold 0.8), *disease normalizer* (maps to Orphanet/OMIM via BioLORD), *knowledge searcher* (general web: Bing/Google/DuckDuckGo; medical-domain: PubMed, Crossref, Orphanet, OMIM, HPO, Wikipedia, MedlinePlus), *case searcher* (two-stage retrieval: embedding-based top-50 → MedCPT-Cross-Encoder re-ranking against a 67,795-case bank), *phenotype analyzer* (wraps PhenoBrain + PubCaseFinder + zero-shot LLM), *genotype analyzer* (wraps Exomiser for VCF annotation/variant ranking).
3. **External data sources** — PubMed (34M+ papers), Orphanet (6,000+ diseases), OMIM (17,000+ genes), HPO (18,000+ terms, 156,000+ annotations), ClinVar, gnomAD, 1000 Genomes, TOPMed, UK10K, ESP.

**Workflow** (two stages, formalized as Algorithm 1 in the paper): (i) *Information collection* — parallel phenotype and genotype evidence gathering, tentative diagnosis; (ii) *Self-reflective diagnosis* — the host re-searches literature specifically about the *predicted* diseases to validate/refute them; if all candidates are ruled out, the system loops back and increases search depth (N += ΔN) rather than terminating. This self-reflection loop is the key structural difference from MedAgent-Pro's single-pass Continue/Terminate/Complete gate — DeepRare can **retry with more evidence** rather than simply halting.

**Post-processing:** a dedicated **reference-link verification** step (syntactic URL check + HTTP HEAD reachability test) strips hallucinated/dead citations before final output — a concrete, cheap mechanism your governance framework could adopt directly as a minimum-bar audit step.

# Exact metrics (not rounded from the abstract)
- 6,563 clinical cases, 8 datasets (+1 in-house), **3,134 diseases**, 14 specialties.
- HPO-only, average across benchmarks: **Recall@1 = 57.18%**, Recall@3 = 65.25%, vs. second-best (Claude-3.7-Sonnet-thinking) at 33.39% Recall@1.
- Multimodal (HPO + genetics), Xinhua Hospital cases: Recall@1 jumped from **39.9% (HPO only) → 69.1% (+genetics)**; Exomiser alone (same genetic tool DeepRare wraps) got 55.9% — i.e., the agentic orchestration adds ~13 points over the bioinformatics tool it's built on.
- **Physician comparison** (163 real outpatient cases, 5 physicians with 10+ years' experience, physicians allowed search engines but not AI tools): DeepRare Recall@1 = **64.4% vs. physicians' 54.6%** — first reported instance of a computational system beating specialist physicians at this task, per the authors.
- **Reasoning-chain validation**: 10 associate chief physicians, 180 cases, 3 reviewers/case → **95.4% mean reference accuracy**. Failure mode breakdown of 200 non-top-5 cases: Reasoning Weighing Error 41.0%, Phenotypic Mimic Diagnosis 38.5%, Etiologically Associated Diagnosis 15.5%, Reasoning Factual Error 2.5%, Evidence Linking Error 2.5%. **Honesty note the authors themselves make**: incorrect references break down into *hallucinated URLs* (plausible but nonexistent) and *irrelevant citations* (from a wrong diagnosis) — this is a genuinely candid failure taxonomy, not a marketing gloss.
- Ablation: agentic wrapping improved GPT-4o's raw Recall@1 from 26.11% → 54.67% (+28.56 points), DeepSeek-V3 from 26.99% → 56.94% (+29.95 points) — the architecture, not the base model, is doing most of the work, and the choice of central-host LLM barely mattered.

# What DeepRare does NOT do (stated limitations, directly from their §4)
- Does not yet cover all valuable data sources (acknowledged as ongoing, not fundamental).
- Knowledge search agent processes phenotypes in aggregate, not with fine-grained adaptive retrieval.
- Targets patients already flagged as possible rare-disease cases — **not a general screening tool** for non-specialist settings (a screening extension is only sketched in supplementary material).
- Patient-interaction module exists but is **unvalidated** — no dataset available yet.
- **No governance/accountability discussion anywhere in the paper.** Despite being the most clinically-validated system in this whole survey, DeepRare has zero discussion of who is liable for a wrong diagnosis, how the memory bank is secured, or how the system would satisfy EU AI Act Article 14/15 human-oversight requirements. This is a genuine, citable gap even in the single most credentialed paper on your list.

---

# 2. ConfAgents — *A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis*
Zhao, Zhu, Wang, Wang, Gao, Ma (Peking University, HKU, Edinburgh). arXiv:2508.04915 (Aug 2025, preprint — not yet peer-reviewed as of this writing)

# Exact mechanism
Four-stage pipeline, formalized with real equations:
1. **MainAgent** produces a probability distribution *p* over candidate diagnoses (from logits for white-box models, or sampling/logit-bias parsing for black-box ones).
2. **CP Judger**: computes a non-conformity score g(x,y) = 1 − p(y|x) for each candidate. Uses **split conformal prediction** — a held-out, *dynamically constructed per-query* calibration set (top-N semantically similar questions from the knowledge base, not a fixed global set), threshold τ = the (1−α)-quantile of calibration non-conformity scores (finite-sample corrected). Builds a prediction set C(x) = {y : g(x,y) ≤ τ}.
   - **|C(x)| = 1 → accept MainAgent's answer directly, no collaboration.**
   - **|C(x)| > 1 → trigger full multi-agent deliberation.**
   - This is a real, textbook-correct application of conformal prediction (Angelopoulos & Bates 2021 formalism), giving a **distribution-free, marginal coverage guarantee of at least 1−α** — this is the strongest formal reliability property of any mechanism in this entire survey, MedAgent-Pro's φ(·) included.
3. **AssistAgents**: dynamically recruited per relevant medical domain (cardiology, neurology, etc. — inferred from the candidate set), each runs an **iterative RAG loop** (question decomposition → retrieval from the Merck Manual → synthesis → refine-or-terminate) against a fixed external corpus.
4. **MainAgent refinement**: synthesizes all AssistAgent reports into a final diagnosis (not a vote — an explicit "Refine" reasoning step over the evidence).

# Exact metrics
- 4 benchmarks (MedQA, MMLU-medical, MedBullets, AfriMedQA), Test-Hard/Test-Normal splits of 100 each, following MedAgentsBench sampling.
- Headline: on MedQA with GPT-4o, ConfAgents hits **80.21% accuracy** (highest of all baselines) while being **2.76×–7.71× faster** than other multi-agent methods.
- Table 3 (absolute cost, MedQA/GPT-4o): ConfAgents 9.11s processing time vs. MDAgents' 25.15s (next-fastest) — **2.76× faster**; 0.33×10⁵ completion tokens vs. MDAgents' 0.66×10⁵ — **2.01× fewer**.
- Ablation (Table 5): removing the CP Judger (naive fixed threshold instead) causes processing time to balloon from 65.72s → 109.87s while accuracy *drops* (67.80% vs. 75.70%) — i.e., a poorly calibrated triage mechanism is worse than no triage at all on both axes simultaneously. This is a genuinely useful negative result for your gap analysis: **naive confidence thresholds are not a safe substitute for calibrated uncertainty**.
- Human evaluation: 14 MD/PhD evaluators, blind comparison, 16 questions — **95.54% average preference** for ConfAgents over ColaCare/MedAgent/MDAgents; 9 of 14 evaluators preferred it on all 16 cases.
- Sensitivity to α (miscoverage rate): the paper explicitly shows α as a tunable dial between safety and speed — smaller α = more cautious/more collaboration triggered, larger α = faster but more risk. α = 0.05 (95% target coverage) was their chosen operating point.

# What ConfAgents does NOT do (stated limitations, their own §6)
- CP Judger reliability **depends on a well-matched calibration set** — for genuinely out-of-distribution cases with no precedent in the knowledge base, confidence estimates degrade, exactly where you'd most want a safety net.
- Reasoning quality is bounded by the external corpus (Merck Manual only) — gaps or errors in that corpus propagate straight through.
- No debate/argumentation mechanism among AssistAgents — conflicting evidence isn't adjudicated, just synthesized.
- **The paper explicitly proposes, as future work, a human-in-the-loop escalation path where the CP Judger routes the most uncertain cases to a clinician** — this is the closest thing in this entire survey to a governance-native design decision (uncertainty-as-escalation-trigger rather than accuracy-as-only-metric), even though the current paper doesn't implement or evaluate it.
- Their own ethical statement is narrow: de-identified public data only, explicitly **not intended for direct clinical use** — a research-tool disclaimer, not a deployment governance plan.

---

# 3. "Enhancing Clinical Decision-Making: Integrating Multi-Agent Systems with Ethical AI Governance"
Chen, Albarqawi, Chen. arXiv:2504.03699 (April 2025, IEEE conference paper)

**This is the one where the deep read matters most, because the title oversells the content relative to the other three papers.**

# What it actually built
A 6-agent MAS (Lab Analysis, Vitals Analysis, Context Analysis, Integration, Prediction, Transparency, Validation — 7 named but "six discrete agents" stated in text) over the **eICU Collaborative Research Database** (200,000+ ICU stays), predicting **ICU mortality and length-of-stay (LOS)** from 150 sampled patients (76 mortality, 74 survived). Agents share a common memory; orchestration is a DAG (lab/vitals/context → integration → prediction → transparency → validation), executed via `asyncio` with GPT-4o as the underlying model for every agent, via a framework called "intelli."

# Exact metrics
- Mortality prediction accuracy: **MAS 59% (mean 58.6%, SD 1.1) vs. SAS 55.7% (SD 0.8)**, p = 0.0001, 8 runs.
- LOS mean error: **MAS 4.37 days vs. SAS 5.82 days** (~25% improvement), p < 0.0001. MSE 35.5 vs. 48.1; RMSE 5.95 vs. 6.94.
- **Transparency score: MAS 85.50 vs. SAS 86.21 — the single-agent system scored *higher* on transparency than the multi-agent one.** The authors report this plainly and don't hide it, which I want to give them credit for — but it directly undercuts the paper's own framing (MAS as the more "governable" architecture).

# What "Ethical AI Governance" actually means in this paper — the honest core finding
The "Transparency Agent" computes an **explainability/interpretability/traceability score by prompting an LLM agent to self-assess against a rubric** (feature-importance identification, reasoning clarity, alternative-scenario analysis, documentation of input sources) — modeled on a symbolic-regression transparency-evaluation method from a different paper (Cawiding et al. 2025) adapted to text output. There is:
- **No external audit standard invoked** (no NIST RMF, no EU AI Act mapping, no ISO 42001) — "governance" here means an internally-computed, LLM-graded transparency metric, not a compliance framework.
- **No accountability/liability model** — the paper never asks who owns a wrong prediction.
- **No description of how "ethical AI principles [are] applied at every stage of the pipeline"** beyond the transparency score — that phrase appears in their intro but isn't operationalized anywhere in the Methods section.
- The related-work section (§II.E) cites real governance concepts well (WHO AI ethics guidance, Healthcare AI Datasheets for bias documentation, GuardAgent for LLM agent safety) — but none of these are actually implemented in their system; they're cited as context, not built on.

**Bottom line for your project**: this paper is evidence that "governance" is being used as a *keyword* attached to a fairly standard multi-agent prediction pipeline, more than as an architectural commitment — and its own numbers (MAS transparency < SAS transparency) are a small but real data point that **more agents does not automatically mean more auditable**, which is directly useful ammunition for your gap-analysis section. This is exactly the kind of paper a rigorous survey needs to actually read in full rather than cite from the abstract, because the abstract alone would have overstated its governance contribution to you.

---

# 4. UALM — *Agentic AI Governance and Lifecycle Management in Healthcare*
Prakash, Lind, Sisodia (University of the Cumberlands). arXiv:2601.15630 (May 2026, preprint)

**This is the most substantial governance-specific paper in the whole survey — but it is a simulation study, not an evaluation of real deployed systems, and says so clearly.**

# Exact architecture: Unified Agent Lifecycle Management (UALM), 5 layers
1. **Identity & Persona Registry** — Non-Human Identity (NHI) certificates, Clinical Scope of Practice definitions, named liability owner per agent.
2. **Orchestration & Mediation** — policy-based access control, goal prioritization, authority mapping; explicit conflict-resolution rule (e.g., "a discharge-planning agent cannot override a medication-safety agent when clinical risk is unresolved").
3. **Context & Memory** — PHI segmentation for retrieval ("Vectorized PHI Sharding"), retention-bound longitudinal context ("Temporal Memory") — a direct, HIPAA-motivated answer to the "what data can an agent remember and for how long" question that none of the diagnostic-architecture papers above even ask.
4. **Guardrail & Compliance** — supervisor agents + "Governance-as-Code" policy engine with kill-switch triggers, blocking unauthorized actions in real time.
5. **Lifecycle & Decommissioning** — mandatory expiration dates per agent, automated NHI token revocation, decision-log retention.

# Exact evaluation method (this is a simulation, be precise about that)
**Monte Carlo simulation**, not a real-world trial: 3 synthetic organizations (150/400/800-bed community/regional/academic profiles), 4 governance conditions (No Governance / Registry-Only / NIST RMF-Lite / UALM-Full), 1,000 runs per organization×condition (12,000 total runs), 6 monthly timesteps, 7 stochastic event types (agent drift, orphan events, PHI near-misses, duplicate deployment, vendor deprecation, inter-agent conflict, prompt injection) injected at specified per-agent-per-month base probabilities, scaled by each agent's clinical-criticality and PHI-exposure scores. Statistics: Mann-Whitney U tests, Cohen's d, Holm-Bonferroni correction for 105 pairwise comparisons.

# Exact results (medium organization, month 6, vs. NIST RMF-Lite — their most important comparator)
| KPI | UALM | NIST RMF-Lite | Δ | Cohen's d |
|---|---|---|---|---|
| Agents with named owner | 89.4% | 76.5% | +17% | 1.42 |
| Credential revocation time | 3.1 days | 15.5 days | −80% | **6.06** |
| Tool calls logged | 86.5% | 54.9% | +57% | 2.95 |
| Orphan-agent ratio | 6.4% | 12.4% | −53% | 0.99 |
| PHI-minimization rate | 71.7% | 40.5% | +77% | 2.83 |
| Control drift rate | 14.6% | 27.3% | −47% | 1.34 |
| Incident rate | 0.12/agent/mo | 0.24/agent/mo | −56% | 1.45 |

All 105 pairwise comparisons significant at p < 0.001. **Only UALM-governed simulated organizations ever reached "Level 2 (Managed)" maturity — and only in 18–21% of runs**, never Registry-Only or NIST RMF-Lite. Sensitivity analysis: even at worst-case parameters (1.5× event rate, 2.0× agent growth), incident reduction only dropped from 60%→56% — the qualitative finding is robust to parameter choice.

# The single most important finding for your roadmap
**Stalled governance adoption erases ~80% of the benefit.** If the governance ramp-up curve plateaus at mid-effectiveness (~70–74%) instead of continuing to ~97% by month 6, incident rates stay nearly as high as under slow/no adoption (0.186 vs. 0.198 events/agent/month, vs. 0.111 under full ramp-up). This is a "deploy-and-forget governance doesn't work" result, which matters if your paper's roadmap includes any kind of staged/phased governance rollout recommendation — the simulation says phasing only works if organizations actually keep climbing the curve, not if they stop at "good enough."

Also notable: **centralization overhead scales as O(n^1.3)** with fleet size — i.e., UALM's own governance layer becomes proportionally more expensive as more agents are added, and a simulated 0.5%/month control-plane outage nearly **triples** (2.8×) the incident rate during the outage window. The authors report this as an honest architectural risk, not a footnote.

# What UALM does NOT do (their own stated limitations, §6)
- **Entirely synthetic scenario, not calibrated against any observed healthcare agent-fleet data** — event probabilities, adoption curves, and organization profiles are all "transparent assumptions," explicitly not empirical estimates. The paper is explicit that results are "model-based projections rather than observed performance."
- Does not model inter-organizational dynamics, vendor relationships, or variation in regulatory enforcement.
- KPI early-warning validity was **mixed across organization sizes** — moderate leading-indicator correlation for medium orgs (r≈0.59–0.60), weak for small (r≈0.42), and *negative* for large orgs (r≈−0.47 to −0.48, a "ceiling effect" the authors flag rather than hide).
- **Zero connection to any diagnostic-reasoning architecture** — UALM governs the *fleet* (which agents exist, who owns them, what they can touch), not the *reasoning* (how a diagnosis is derived, whether the evidence chain is valid). This is the layer MedAgent-Pro/DeepRare/ConfAgents operate at, and UALM has nothing to say about it. **This is precisely the gap your novel contribution can fill: nothing in this survey governs both the fleet-operations layer and the diagnostic-reasoning layer at once.**

---

# Cross-cutting honest synthesis from this deep-read pass

1. **The strongest formal reliability property in this whole survey (conformal prediction's distribution-free coverage guarantee, in ConfAgents) exists in a paper with zero governance framing.** The strongest governance architecture (UALM's five layers) exists in a paper with zero diagnostic-reasoning content. The most clinically validated diagnostic system (DeepRare, published in *Nature*) has zero governance discussion at all. **No single paper you asked me to deep-read connects statistically rigorous uncertainty quantification, diagnostic evidence-grounding, and fleet-level lifecycle governance in one architecture.** That three-way gap is more specific and more defensible than the general "governance is thin" observation from Deliverable 1.
2. **One paper's own numbers quietly undercut its governance framing** (MAS transparency < SAS transparency in the eICU paper) — worth citing directly as evidence that adding agents is not a free lunch for auditability, contra an assumption that seems to run through some of the technical literature.
3. **UALM's "stalled adoption erases 80% of benefit" and "O(n^1.3) centralization overhead" findings** are the most concrete, quantified reasons in this entire survey to worry about a naively-designed governance-native architecture — they should directly inform any design choices in your roadmap about centralized vs. federated audit/control mechanisms.
4. Bibliographic hygiene note to carry into your paper: verify the ConfAgents arXiv ID against the GitHub repo before citing, given the mix-up documented above.

---

# What's next
The three options from before are now: (1) proceed to the phased roadmap, now informed by these four deep reads rather than abstracts; (2) go deep-read on 2–3 more papers (Tree-of-Reasoning and MedSentry would be the next-highest-value, since they're the strongest "reasoning structure" and "multi-agent safety" entries respectively and I've only read them at abstract level); or (3) something else you'd rather prioritize. Given the three-way gap identified in point 1 above, I'd lean toward the roadmap next — but it's your call.