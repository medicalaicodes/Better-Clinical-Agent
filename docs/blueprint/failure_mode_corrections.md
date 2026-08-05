# HERALD v1.1 — Verification Errata and Blueprint Corrections

**Date:** 27 July 2026
**Applies to:** `internship_presentation.md` (gap taxonomy) and `HERALD_blueprint.md` (v1)
**Method:** live source verification of the claims the blueprint's principles depend on.
**Scope caveat:** this is a *targeted* audit of load-bearing claims, not a full re-verification of
all 27 papers. Claims not listed below were not checked and should not be assumed correct.

---

# Part A — Verification of the internship presentation

# A.1 Claims that FAIL verification

# E1 — Cerebra does NOT halt on missing MRI. The claim is inverted.

**Presentation, Gap 11:** *"Cerebra halts completely if MRI is missing. It lacks the graceful
degradation required in real clinics."* Also Slide 4c: *"If the patient didn't get an MRI, the
Cerebra pipeline halts completely."*

**What the source says:** Cerebra's own project materials state it *"utilizes an advanced
multimodal fusion architecture that gracefully handles missing data types without failing."*

**Your own other document already says the opposite.** `ERC_survey.md` §1.2 lists Cerebra's
mechanism as *"Gracefully handles missing MRI"* and §1.5 repeats it under Missing-Modality
Robustness. The presentation and the ERC survey directly contradict each other on the same system.

**Severity: high.** Gap 11 ("Missing Modality Brittleness") rests almost entirely on this example,
and HERALD principle **P3** cites it by name as the failure being designed against.

**Additional problem with Gap 11:** MedAgent-Pro also handles missingness. Its ICLR-2026 version
describes a memory-initialisation step that *"select[s] executable steps from P based on data
availability, filtering out those requiring unavailable inputs,"* and its case study explicitly
*"skips those requiring unavailable data such as OCT scans."* So two of the systems the gap cites
do handle missing modalities.

**Correction.** Gap 11 should be restated as a *representation* gap, not an *absence* gap:

> Existing systems handle missingness by **skipping** the affected step (MedAgent-Pro) or by
> **fusing over what remains** (Cerebra). Neither *represents* the consequence: which hypotheses
> were left unresolved, whether the missing item is obtainable, and whether the output's confidence
> should be discounted because of it. Skipping is silent; fusing is silent. Nothing discloses.

That is a smaller claim and a true one. It also remains a sufficient basis for P3 and for research
extension R3, which should be renamed from "graceful degradation" (solved) to
**"missingness disclosure and its effect on calibration"** (not solved).

---

# E2 — The "90% → 42.1%" headline is not in the MedOpenClaw paper.

**Presentation, Slide 8 ("Most Critical Finding"):** *"Agents scoring 90%+ on static 2D benchmarks
collapse to 42.1% 3D coordinate precision when asked to navigate real clinical imaging software"*,
attributed to MedOpenClaw + MedFlowBench.

**What the source says.** Full text of arXiv:2603.24649 (v1 Mar 2026, v2 May 2026; TUM / Imperial /
Oxford / CMU / NVIDIA / NUS) was read. There is **no 42.1% figure**, and **no comparison against a
90% static-benchmark score** anywhere in the paper. No cross-benchmark comparison of that shape is
made or attempted.

**What the paper actually reports.** Two scores per task: `Task` (answer only) and `Strict`
(answer + deterministically-checked evidence: key-slice offsets, RAS coordinates, lesion-state
fields).

| Setting | Best Task | Best Strict | Note |
|---|---|---|---|
| Brain MRI (UCSF-PDGM) | 0.73 | 0.73 | Gemini 3.1 Pro; Loc 0.96. **No gap at all here.** |
| Longitudinal MRI (LUMIERE) | 0.37 | **0.18** | Large gap |
| Lung PET/CT (NSCLC) | 0.55 | **0.14** | Large gap |
| Model average (best model) | 0.53 | 0.32 | |

The real finding is: **answer-only scoring overestimates workflow competence; the gap appears when
evidence must also be correct, and only on complex tasks.**

**And one result partly inverts the presentation's narrative.** In Track C, a naive 2D
slice-montage baseline sometimes *beat* interactive viewer use — GPT-5.5 scored 0.84 Task / 0.79
Strict on Brain MRI from static montages versus 0.71 / 0.66 in the live viewer. Static 2D input is
not uniformly the inflated condition.

**Severity: critical.** This is the presentation's headline slide and it is repeated in HERALD §1
and §8. It cannot be presented to a supervisor as written.

**Correction (defensible, and a better claim):**

> On MedFlowBench, requiring that an answer be supported by checkable evidence drops the best
> model's average score from 0.53 to 0.32, and on the two hardest task families from 0.37→0.18 and
> 0.55→0.14. Correct answers routinely rest on wrong or absent evidence. The authors also report
> that only 3 of 10 repeated runs of the same registration workflow completed a valid operation
> sequence.

That is stronger for HERALD's purposes than the fabricated version, because it is *specifically*
an argument for evidence-linked outputs — which is what P1 and the entailment gate exist to do.

**A relevant detail the presentation omits.** MedGemma 27B IT scored 0.19 Task / **0.00 Strict**;
MedGemma 1.5 4B IT scored 0.26 / 0.04 / **0.00 Loc**. The proposed MVP backbone has essentially
zero spatial-grounding ability in this evaluation. That does not invalidate the MVP — HERALD gets
localisation from tools, not from the backbone — but it must be stated, and it rules out any design
where the backbone is asked to localise.

---

# A.2 Claims that PASS verification

| Claim | Status |
|---|---|
| MedAgent-Pro: adding the weak Disc-Haemorrhage indicator drops F1 by **16.5 points** |  Confirmed. Table 3, comparison of the last two ablation rows (−5.1 and −16.5). |
| MedAgent-Pro: F1 falls to **14.3%** without vCDR |  Confirmed, and the paper's explanation is that the LLM decider becomes conservative when a critical indicator is absent. |
| MedOpenClaw / MedFlowBench exists as described (3D Slicer, full studies, auditable runtime) |  Confirmed — only the *numbers* attributed to it were wrong. |
| MedRAX: 63.1% ChestAgentBench, 8 CXR tools, ICML 2025 |  Confirmed. |
| Tree-of-Reasoning (Peng et al.), ACM MM 2025 |  Confirmed. arXiv:2508.03038; proceedings pp. 1744–1753. |

# A.3 Claims not verified in this pass

Not checked, and therefore not safe to present: Meissa (57.2% MedQA, 22×), MDAgents (72.9 vs 75.1),
MAM (70.2→64.0), CoMMa (+0.5% AUC, 0.595), MemSeg-Agent (77.30 vs 30.74), MedSAM3 (Dice 0.14;
cell→nuclei 0.00), IBISAgent (28.70 s), ToolSelect (72 vs 96), Camyla (0.932 vs 0.918), MedXIAOHE
(87.53 → 25.7), MedAgentSim (26–29%), AgentDS (10–15 h).

**Recommended triage before the presentation is shown to anyone:** verify the five numbers you
actually intend to say out loud. Everything else can be softened to a qualitative statement.

# A.4 One further overstatement

**Gap 10** states *"No paper has deployed in a real hospital"* and that clinician validation is
*"universally absent."* AMIE was published in *Nature* with a 20-clinician comparative study;
DeepRare (in your own `ComparisionTable.md`) is a 2026 *Nature* paper reporting 95.4% clinician-rated
reasoning-chain agreement; Cerebra reports validation across multiple hospital systems. Restate as:
*"no system in this set has regulatory clearance or reported prospective in-workflow deployment"* —
which is true and still makes the point.

# A.5 The pattern worth internalising

The presentation is **strong on qualitative failure modes and weak on quantitative provenance.**
The eleven gaps are, with the Gap-11 exception, real and well-observed. But several of the specific
numbers are cross-system comparisons the source papers never made, and at least one is not in its
source at all. In a research group this is the failure mode that costs credibility fastest, because
the one number a reviewer checks is always the headline one.

**Process fix:** for every quantitative claim, record `paper → table/figure → exact quantity →
comparison condition`. If any of the four is missing, the claim becomes qualitative.

---

# Part B — Corrections to the HERALD blueprint

# B.1 Corrections forced by Part A

| Blueprint location | Change |
|---|---|
| §1, §8.1 (Axis 1), §12 | Delete the "90% → 42%" framing everywhere. Replace with the Task-vs-Strict gap. |
| §3-P3 | Remove Cerebra as the halt example. Re-motivate P3 on *non-disclosure*: MedAgent-Pro skips silently, Cerebra fuses silently, neither reports what was lost. |
| §9-R3 | Rename to "Missingness disclosure and calibration under incomplete evidence." Narrow the novelty: graceful degradation exists; *disclosing it and measuring its effect on calibration* does not. |
| §7.1 | Add: the MVP backbone scores ~0 on spatial localisation in MedFlowBench. All localisation must come from tools. Never ask the backbone for coordinates. |
| §9-R1 | Cite arXiv:2507.10571 (Roumeliotis et al., IEEE Access 2026) — trust-aware orchestration modulating agent trust via ECE/OCR/CG. Verified real. R1's novelty narrows to: **per-case, case-descriptor-conditioned** competence (not global ECE), **medical** domain, and **joint use for routing and weighting** with measured transfer between the two. |

# B.2 Design defects found on re-reading (independent of Part A)

**D1 — The hypothesis space conflates findings with explanations. (Serious.)**
§5.9 defines a closed set of "~18 CXR findings + OTHER" and §5.12 renormalises a posterior over it,
while §5.11 correctly states that effusion + consolidation is not a conflict. Those cannot both
hold. Chest findings co-occur routinely; renormalising over them asserts exclusivity and will make
evidence for one finding mechanically suppress another. It also breaks the abstention trigger —
a flat posterior is the *correct* output for a patient with three simultaneous findings.

*Fix (two levels, no new mathematics required):*
- **Findings level** — independent per-finding probabilities, multi-label, **no renormalisation**.
  Metrics: per-label AUROC, mAP.
- **Explanation level** — a small mutually-exclusive set (~8–12: pneumonia, pulmonary oedema,
  effusion, pneumothorax, COPD/hyperinflation, mass/nodule, normal, OTHER) with a hand-written
  table of `P(finding | explanation)`. Metrics: Top-1/Top-3, Capture@K.
Only the explanation level is a simplex, so only it gets renormalised, ranked, and abstained on.

**D2 — The reliability weight `f` is unspecified.**
§5.12 writes `logit(h) += w(packet) × llr(packet, h)` with `w = f(reliability)` and never defines
`f`. As written it is a tuned knob that silently controls the posterior, the abstention threshold,
and every guarantee downstream. Either derive it or fit it — but specify it before writing code,
and print it in the trace.

**D3 — `agreement` inside the packet breaks immutability and double-counts corroboration.**
§5.8 makes cross-source agreement one of three inputs to a packet's reliability, while §5.3
requires packets to be immutable. Packet 1's agreement score depends on Packet 2 existing, so its
reliability changes retroactively. Worse: when three classifiers concur, corroboration raises each
packet's reliability *and* contributes three summed terms — confidence grows faster than the
evidence warrants.

*Fix:* reliability at packet creation uses **competence + integrity only** (both computable from
that packet alone). Agreement moves to the fusion layer, where corroboration is counted once.

**D4 — The TorchXRayVision "independent specialists" premise is weaker than §6.2 claims.**
Two of the weights (`mimic_ch`, `mimic_nb`) are trained on the *same MIMIC-CXR images*, differing
only in the NLP labeller; `chex` shares the CheXpert labeller with them; all are DenseNet-121 at
224². Realistically there are ~3 independent sources (NIH-family, CheXpert/MIMIC-family, PadChest),
not 5–6. Separately, the weights do **not** all support the same 18 labels — each supports the
subset present in its training corpus, so inter-tool agreement is only defined on the intersection.

*This is load-bearing for the whole modality choice.* **Enumerate the supported-label intersection
empirically on day one of Phase 0** — load each weight, read its pathology list, build the matrix.
If the intersection with ≥3 genuinely distinct sources is under about six labels, the multi-source
premise is thinner than assumed and the scope needs renegotiating then, not in Phase 3.

**D5 — `Capture@K` does not type-check on CXR findings.**
Capture@K was defined on NEJM CPC differentials, where each diagnosis has a *diagnostic test of
record*. A CXR *finding* has no test of record. The metric only becomes well-defined at the
explanation level of D1, and only with an explicit ~20-row action-mapping table
(explanation → next action). That table is worth writing anyway: it is exactly what `REQUEST(item)`
needs in order to name something obtainable.

**D6 — "Replay determinism" is not achievable as a Phase-1 gate.**
vLLM under continuous batching is not bitwise deterministic; identical prompts can produce
different tokens depending on batch composition. The gate as written will fail for reasons that have
nothing to do with HERALD.

*Fix:* content-address an LLM response cache (key = model + weight hash + prompt hash + sampling
params + seed); replay reads from cache. Then measure **regeneration stability** separately —
resample k=5, report agreement on TaskFrame and Disposition. That is a more interesting number than
determinism, and it is a real robustness metric rather than an infrastructure assertion.

**D7 — Hardware claim is wrong.** §7.1 offers "MedGemma-27B-text as a text-only upgrade if VRAM
allows." A 27B model in bf16 is roughly 54 GB. It does not co-reside with the tool layer on a 24 GB
card at any useful quantisation. Remove it from the v1 stack; it is a separate run on separate
hardware or nothing.

**D8 — `REQUEST` fires before the risk threshold, which can flatter the abstention results.**
In §4.3-L8 the decision cascade reaches `REQUEST` (rule 3) before the conformal/selective-risk check
(rule 4). Hard cases can therefore be routed into `REQUEST` and never enter the accepted set,
making selective risk look better than it is. *Fix:* count `REQUEST` as abstention in every
risk–coverage plot, and always report a `REQUEST`-disabled variant alongside.

---

# Part C — De-scoped build plan (HERALD-Core)

The v1 blueprint specifies 16 modules, 8 evaluation axes × 3 settings, and 7 research extensions.
That is a good *design document* and a bad *project plan*. For comparison: MedRAX reached ICML 2025
with a ReAct loop over 8 tools plus one benchmark. CXReasonAgent is an LLM plus 12 deterministic
tools plus one dialogue benchmark. Each is one idea and one evaluation artifact.

**Cut to this. Everything else moves to v2 or to a research extension.**

| Keep in v1 | Why it survives the cut |
|---|---|
| Typed Evidence Ledger + `MissingEvidence` | The one decision everything else depends on; cheap now, a rewrite later |
| Reliability Estimator (competence + integrity only) | This *is* the contribution |
| Two-level hypothesis state (findings / explanations) | Required for the abstention and DDx claims to be well-posed |
| Simple reliability-weighted fusion, printable | Auditability; and D2 must be resolved anyway |
| Four-way typed Disposition | The clinically meaningful output; cheap |
| OTel spans + Case Bundle + LLM response cache | Cheap now, impossible later |
| 5 no-op governance hooks | One afternoon, not one week |

| Defer | To |
|---|---|
| Acquisition Planner / EIG | v2 — use a fixed tool order plus the anti-under-calling coverage rule in v1 |
| Conformal risk control | v2 — report empirical risk–coverage curves in v1; add the guarantee once the estimator works |
| Entailment gate | Research extension R5 — measure unsupported-claim rate in v1 without gating |
| Conflict Detector | Minimal version only (direct contradiction on the same concept); the taxonomy is v2 |
| Governance policy demo | v2 / teammate |
| Evaluation axes 2, 5, 7 | v2 — v1 reports axes 1 (clinical), 3 (reliability), 4 (abstention), 6 (corruption) |

**Revised gate structure.** Three go/no-go checkpoints, each of which can legitimately end the
project early with a reportable result:

1. **P0 gate (day 1–2):** the TorchXRayVision supported-label intersection is ≥ 6 labels across ≥ 3
   genuinely distinct sources. *If not,* the multi-source premise fails and the modality or the
   claim changes before anything is built.
2. **P3 gate:** between-slice competence variance is large enough to route on. *If not,* routing has
   no headroom — report it as a negative result and pivot the contribution to reliability weighting
   alone.
3. **P4 gate:** reliability AUROC for predicting packet correctness beats temperature-scaled softmax.
   *If not,* that is the negative result, and it is publishable.

Start `NEGATIVE_RESULTS.md` on day one. On current evidence there is a real chance the strongest
output of this project is a well-measured demonstration that reliability-aware orchestration does
not pay for itself in this setting — and that would be more useful to the field than another system
paper.

---

# Part D — On the external audit document

The uploaded `HERALD_audit_and_roadmap.md` identifies D1–D8 above independently, and its diagnosis
of the fusion layer is correct and worth quoting to yourself:

> implemented as written, HERALD would be most overconfident precisely when its specialists agree
> because they were trained on the same data — the failure it exists to prevent, occurring inside
> the module built to prevent it.

Two cautions about it, though.

**It audits the mathematics and accepts the evidence base.** It states that the load-bearing
citations "were verified live and all resolve." Resolving is not supporting: MedOpenClaw resolves,
and does not contain the claim made about it. The Cerebra inversion and the 42.1% figure are more
serious than any of the mathematical defects, because a wrong motivating premise invalidates a
design principle, whereas a fusion equation is a week's work.

**It adds more than it removes.** It criticises scope while introducing hierarchical shrinkage,
Learn-then-Test, Jeffrey conditioning, redundancy factors, and a restructured shift-arm design.
Adopt its *diagnoses* (D1–D8) and treat its *prescriptions* as v2 material. The instinct that v1 is
overbuilt applies to the audit as well.
