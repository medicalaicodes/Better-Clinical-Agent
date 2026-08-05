# Why This Metric is Hard

The difficulty is a three-way entanglement:

1. **What counts as "conflict"?** Specialist agents can disagree in degree (one says 60% risk, another says 40%) or in kind (one says "aggressive tumor," another says "indolent").
2. **What counts as "preservation"?** The final output must *communicate* conflict — but how? Lower confidence? An explicit flag? A split report?
3. **What is the ground truth?** Here is the core epistemological problem: *when is it correct to preserve conflict vs. resolve it?*

On a case where pathology says high-risk and genomics says low-risk, and the patient happens to die — was preserving conflict the right call? Was resolving it? You can't evaluate "correctness of conflict reporting" using only outcome labels. You need a different kind of ground truth.

---

# The Key Insight: Two Separate Problems

A conflict preservation metric is actually **two metrics** that must be distinguished:

| | **Conflict Detection** | **Conflict Communication** |
|---|---|---|
| **Question** | Does the system *notice* that specialists disagree? | Does the system *report* this disagreement rather than suppressing it? |
| **Ground truth needed** | Injected/known conflicts, or inter-specialist disagreement magnitude | The system's own output — no external label needed |
| **What failure looks like** | Coordinator ignores a disagreeing specialist | Coordinator reports a single confident conclusion despite noticing conflict |

Most people conflate these. They're not the same. A system could detect conflict perfectly but still suppress it (false consensus). The metric must test both.

---

# Concrete Metric Ideas

# Idea 1: Injected Conflict Recovery (ICR) — Most Rigorous

**How it works:**

You synthetically flip one specialist's output to conflict with the others, then measure whether the coordinator flags it.

```
Setup:
  - Ground-truth case: path says HIGH risk, genomics says HIGH risk → coordinator says HIGH risk.
  - Inject: flip pathology to LOW risk (keeping it clinically plausible).
  - Expected behavior: coordinator should lower confidence AND flag disagreement.

Measure:
  ICR-Recall  = (conflicts flagged by coordinator) / (total injected conflicts)
  ICR-Precision = (flagged conflicts that were injected) / (all conflict flags raised)
```

**Why it works:** You have perfect ground truth — you know *exactly* when conflict exists because you created it. This is analogous to how MedAgent-Pro's disc-haemorrhage failure was discovered: a known-bad tool was inserted and its effect was measured.

**Implementation detail:** The injected output must be *plausible*, not random noise. A pathology agent saying "0% cancer" when the slide shows clear cancer is detectable as an outlier; a pathology agent saying "intermediate grade" when genomics says "high-grade" is a genuine clinical ambiguity. The injection must sample from the realistic output distribution of the specialist.

**Limitation:** This only evaluates conflict *detection*, not whether the system was *right* to preserve it in cases without injected conflicts.

---

# Idea 2: Confidence–Disagreement Correlation (CDC) — Most Elegant

**Core idea:** If specialists genuinely disagree, the coordinator *should* be less confident. A system that forces false consensus will be confident regardless of specialist disagreement. Measure this as a correlation.

```
For each test case i:
  D_i = disagreement magnitude among specialists
      = e.g., std(specialist confidence scores)
      = or Jensen-Shannon divergence between specialist output distributions
      = or proportion of specialists voting differently from the majority

  C_i = coordinator's reported confidence (its own probability for the predicted class)

CDC = Spearman_correlation(D_i, 1 - C_i)
```

**Interpretation:**
- CDC close to 1.0 → confidence drops when specialists disagree (correct behavior).
- CDC close to 0 → coordinator is equally confident regardless of specialist disagreement (false consensus behavior).
- CDC negative → coordinator becomes *more* confident when agents disagree (Diversity Collapse failure mode, exactly what the NeurIPS debate paper documented).

**Why this is powerful:** It does not require any ground-truth conflict labels. It measures a *behavioral property* of the system purely from inputs and outputs. You can compute it on any test set.

**Limitation:** It conflates "conflict detection" and "conflict communication" — a system could lower confidence without explicitly flagging the conflict in its output.

---

# Idea 3: Label-Uncertainty Alignment (LUA) — Handles the Epistemological Problem

This directly addresses the philosophical problem: *when should conflict be preserved?*

**The insight:** You cannot use outcome labels to judge conflict flagging, but you *can* use **label uncertainty** — cases where the ground truth itself is uncertain. These exist in both datasets:

- In **MIMIC**: cases at the boundary of the outcome definition (e.g., patients who died 2 days after the 48-hour window, or length-of-stay exactly at 7 days).
- In **TCGA survival**: censored patients with short follow-up — the "true" outcome is unknown.
- Directly measurable: cases where an ensemble of supervised models shows high *predictive variance*.

```
Operationalization:
  1. Label each test case with its "ground truth uncertainty" score U_i:
     - Use a calibrated ensemble's entropy over its predictions, or
     - Use survival data: censoring indicator + time-to-event
     - Use expert disagreement if annotation data exists
  
  2. For each case, flag whether the coordinator preserved conflict: P_i ∈ {0,1}
  
  3. LUA = Spearman_correlation(U_i, P_i)
```

**Interpretation:** High LUA means the system flags conflict on cases that are genuinely ambiguous (high label uncertainty) and does NOT flag conflict on cases that are clear. This is the normatively correct behavior.

**Why this is novel:** No existing multi-agent benchmark evaluates conflict flagging against label uncertainty. It requires a small amount of additional annotation effort but gives you a *normatively grounded* metric, not just a behavioral one.

---

# Idea 4: Conflict Report Fidelity via LLM Judge (CRF) — Most Practical

If the system produces free-text output alongside its prediction, you can use an LLM judge (or a TBFact-style entailment check) to evaluate whether conflict is mentioned.

```
For each case where inter-specialist disagreement exceeds a threshold:
  Prompt a judge model:
  "The specialist agents produced the following outputs: [outputs].
   Does the following coordinator report acknowledge the disagreement or
   report uncertainty stemming from conflicting specialist opinions?
   Report: [coordinator output]
   Answer: Yes / No / Partial"

CRF = proportion of "Yes" or "Partial" responses on high-disagreement cases
```

**Why it's useful:** It evaluates the *communication* half of conflict preservation. A system that numerically down-weights a specialist but gives a confident final report is still suppressing conflict from the user's perspective.

**Limitation:** LLM-as-judge has known biases (verbose outputs score higher, confident outputs score lower regardless of content). Should be validated against human annotations on a calibration set.

---

# Idea 5: Abstention Calibration Curve (ACC) — Connects to Selective Safety

Rather than measuring conflict preservation directly, measure its *downstream effect*: when a system preserves conflict, it should either lower confidence (measured by CDC) or abstain/escalate. This gives a risk-coverage trade-off curve:

```
For a system with abstention capability:
  - Sort test cases by coordinator's conflict flag strength (a scalar).
  - At each threshold θ: the system "abstains" on cases with conflict strength > θ.
  - Plot accuracy and calibration on non-abstained cases vs. θ.

A well-calibrated system: abstaining on high-conflict cases improves both accuracy
and calibration on the remaining cases. 

Metric: Area Under the Risk-Coverage Curve (AURCC)
```

**Why this matters:** It is the only metric that evaluates whether conflict preservation is *useful* rather than just syntactically present. A system that flags every case as conflicted has perfect ICR-Recall but terrible coverage — the AURCC penalizes this.

---

# Putting It Together: A Battery, Not a Single Number

No single metric captures all of this. The proposal would define a **Conflict Preservation Battery (CPB)**:

| Metric | What It Tests | Requires |
|---|---|---|
| ICR (Injected Conflict Recovery) | Detection: does the system notice injected conflict? | Synthetic perturbation harness |
| CDC (Confidence–Disagreement Correlation) | Communication: does confidence fall under disagreement? | Only specialist + coordinator outputs |
| LUA (Label-Uncertainty Alignment) | Normativity: does flagging align with genuine ambiguity? | Ensemble uncertainty or survival metadata |
| CRF (Conflict Report Fidelity) | Text communication: does the report mention conflict? | LLM judge + human calibration set |
| AURCC (Abstention Calibration Curve) | Utility: does conflict-flagging improve downstream decisions? | Abstention capability in the system |

The **publishable novelty** is not any single one of these — it is establishing that this battery is *necessary* and that systems performing well on standard AUROC can fail all five. That is the claim that would make this a benchmark contribution.

---

# The One Concrete Paper Claim This Enables

If you run the naive multi-agent baseline and the ERC on the same test set, you can show something like:

> *"Naive majority-voting achieves AUROC 0.748 (comparable to ERC at 0.751) but CDC = −0.12 (confidence increases with disagreement) and ICR-Recall = 0.23 (misses 77% of injected conflicts). ERC achieves CDC = 0.67 and ICR-Recall = 0.81. Standard AUROC evaluation declares the systems equivalent; the CPB reveals a fundamental behavioral failure in the naive system."*

That is a result that would be cited. It demonstrates the benchmark is necessary, not merely useful.