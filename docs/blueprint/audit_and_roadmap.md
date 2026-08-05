# HERALD — Technical Audit and Chronological Implementation Roadmap

**Reviewer role:** clinical AI systems architect / deep-research audit
**Subject:** `HERALD_blueprint.md` (v1) + `HERALD_Research.md` (evidence dossier)
**Date:** 27 July 2026
**Verdict in one line:** the architecture is *directionally correct and unusually well-motivated*, but it contains **four mathematical defects that will produce confidently wrong outputs if implemented as written**, **one dependency inversion in the build order**, and **three feasibility assumptions that are false on the stated hardware and data**. All are fixable. None require abandoning the design.

---

# 0. Executive audit summary

| ID | Finding | Severity | Where | Fix cost |
|---|---|---|---|---|
| **M1** | Hypothesis space treated as mutually exclusive; CXR findings are multi-label and co-occurring. Renormalising over them is a category error and directly contradicts §5.11. | **Critical** | §5.9, §5.12, §4.5 | 1 table + rewrite of fusion (2 days) |
| **M2** | `w = f(reliability)` multiplying an LLR is a type error with no generative justification; `f` is unspecified and hand-tuned. | **Critical** | §5.12, §3-P2 | Replace with derived noisy-channel LR (1 day) |
| **M3** | `agreement` is a packet-level reliability signal **and** an input to fusion → corroboration counted twice; also contradicts the append-only immutable-packet rule. | **Critical** | §5.3, §5.8, §5.12 | Split `r_pre` / ledger-level agreement (1 day) |
| **M4** | Naive log-odds summation over TorchXRayVision classifiers with overlapping training corpora; `mimic_ch` and `mimic_nb` are the *same images*. | **Critical** | §5.12 | Per-label fitted combination weights (2 days) |
| **M5** | "Reliability-discounted EIG" presented as a novel coupling; it is the *implied* consequence of M2 done correctly. Claim integrity issue. | High | §5.10, §9-R4 | Relocate novelty claim (0 days, prose) |
| **M6** | Conformal Risk Control applied to a non-monotone risk; guarantee reported over the wrong set; calibration not selection-conditioned. | High | §5.13, §4.3-L8 | Switch to Learn-then-Test + accounting fields (3 days) |
| **M7** | Competence slices are statistically underpowered — ~1,944 (tool × label × slice) cells, most with `n_pos < 10`. "Declared-but-unmeasured competence" is the blueprint's own named fatal failure, and the design produces it. | High | §5.4, Phase 3 | Hierarchical shrinkage + `MEASURED` flag (3 days) |
| **M8** | `Capture@K` and Top-1/3/10 DDx are transplanted from CPC differential diagnosis; neither type-checks on a CXR finding set. | High | §5.9, §8-Axis 1 | Action-mapping table + metric swap (2 days) |
| **CF1–CF5** | Five control-flow defects, incl. replay determinism being **unachievable** under vLLM continuous batching. | Med–High | §4.1, §4.3, §5.3 | See §3 |
| **F1** | `MedGemma-27B-text` does **not** fit 24 GB bf16 (≈54 GB). vLLM default `gpu_memory_utilization` starves the tool layer. | High | §7.1, §7.3 | Config discipline (0 days) |
| **F2** | TorchXRayVision weights do **not** all support the same 18 labels. The multi-source agreement premise holds only on the label intersection — likely 6–10 labels, not 18. **This is load-bearing for the entire thesis.** | **Critical** | §6.2-(1), §6.3 | Empirical enumeration in P0 (1 day, must be first) |
| **F3** | CXR-LT 2026 Task 2 OOD evaluation is **n = 305 internal + 79 external**, not 500. It is *label* shift on PadChest, not *domain* shift. | High | §6.3, §8 | Restructure shift arms (1 day) |
| **F4** | VinDr-CXR is PA-only → the `REQUEST(lateral_view)` and missing-modality experiments **cannot run on it**. | Med | §6.3, §8-Axis 5 | Reassign datasets per experiment |
| **F5** | Full evaluation grid ≈ 250 GPU-hours without caching. Deterministic tool/LLM response cache is an **architectural requirement**, not an optimisation. | High | §8 | Build into P2 |
| **F6** | Reproducing MedRAX locally is not realistic (hosted reasoning core + large VLM tools). MedCTA (107 tasks, 5 tools, no segmentation) is not a usable evaluation target. | Med | §6.2-(5), §8 | Demote to citation-only |
| **F7** | Timeline underestimated ~30%. Realistic 22–24 weeks vs. stated ~17.5. | Med | §13 | De-scoped critical path provided |
| **C1** | Close prior art absent from related work (`arXiv:2507.10571`, trust-calibrated orchestration with ECE-based agent trust for visual classification). Threatens R1's novelty framing. | High | §9-R1 | Address explicitly |

**Good news, stated plainly, because the rest of this document is critical:** the five most load-bearing citations were verified live and all resolve. More importantly, <cite index="26-1">Robust-TO's combination step is a coarse three-tier (high/medium/low) synthesis with a confidence-cost GRPO reward</cite> — a heuristic. A HERALD fusion layer built on the corrected M2/M4 mathematics has a **bounded-evidence guarantee** (Proposition 1, §1.2) that Robust-TO does not. That makes R1 a *methodological* contribution, not merely a domain transfer. This is a stronger position than the blueprint claims for itself.

---

# PART I — TECHNICAL AND PRACTICAL AUDIT

# 1. Mathematical validity

# 1.1 M1 — The hypothesis space is not a simplex

§5.12 specifies a posterior over the hypothesis set, "renormalised over observed evidence." §5.9 defines that set as "the ~18 CXR findings + `OTHER`." §5.11 then states that "effusion *and* consolidation is not a conflict."

These are inconsistent. Renormalisation to a simplex asserts exclusivity; the conflict rule asserts co-occurrence. Chest findings co-occur constantly (effusion + atelectasis + consolidation is the modal pneumonia presentation). Three consequences:

1. The abstention trigger "flat posterior mass across clinically divergent hypotheses" is **meaningless** when the truth is genuinely multi-label — a flat posterior is the *correct* answer for a patient with three findings.
2. Evidence for finding A mechanically suppresses finding B through renormalisation, producing spurious negative correlation the tools never reported.
3. `Capture@K` and Top-K ranking are defined on exclusive hypotheses.

**Correction — two-level state.** Separate *findings* (multi-label, what tools measure) from *explanations* (exclusive, what a differential ranks).

- **Level F.** Independent marginals $q_j = p(F_j = 1 \mid E)$ for $j = 1 \dots J$ findings. Updated by log-odds accumulation. **No renormalisation.**
- **Level X.** An exclusive set $\mathcal{X} = \{x_1 \dots x_K\} \cup \{\texttt{OTHER}\}$ of clinical explanations, $K \approx 8\text{–}12$, with a hand-authored $J \times K$ table $P(F_j = 1 \mid x_k)$.

Level X is updated from Level F by Pearl's virtual-evidence (Jeffrey conditioning) rule, which is the correct way to pass *soft* marginals into a downstream posterior:

$$p(x_k \mid E) \;\propto\; \pi_k \prod_{j=1}^{J} \Big[ q_j\, P(F_j{=}1 \mid x_k) \;+\; (1-q_j)\, P(F_j{=}0 \mid x_k) \Big]$$

This preserves everything the blueprint wants — a proper simplex for the flat-posterior abstention shape, a well-posed EIG target, `NOT_EXAMINED` vs `EXCLUDED` separation — and removes the contradiction. Cost: one $J{\times}K$ table (~180 cells) authored once, alongside the discrimination table §5.9 already requires.

> **Consequence for §5.9/§5.12/§8-Axis 1:** finding-level metrics become multi-label (mAP, R@k, per-label AUROC); Top-K and Capture@K live only at the explanation level.

---

# 1.2 M2 — Reliability is not a weight; it is a channel parameter

§5.12 specifies `logit(h) += w(packet) × llr(packet, h)` with `w = f(reliability)`, and §5.13 leaves `f` unspecified. This is a type error: multiplying a log-likelihood-ratio by a scalar is not the Bayesian consequence of any noise model, so `f` becomes a tuned hyperparameter that silently controls every downstream guarantee.

**Correction — derive the shrinkage from a noisy-channel model.**

Let $F \in \{0,1\}$ be the latent finding with prevalence $\pi$. Let the tool occupy a state $Z \in \{\texttt{ok}, \texttt{fail}\}$ with $P(Z = \texttt{ok}) = r$, where $r$ is the reliability. If `ok`, the tool emits $S \sim p_{\text{ok}}(\cdot \mid F)$. If `fail`, it emits $S \sim p_{\text{fail}}(\cdot)$, and the honest choice for a failed tool is the **marginal**, $p_{\text{fail}} = \pi\, p_{\text{ok}}(\cdot \mid 1) + (1-\pi)\, p_{\text{ok}}(\cdot \mid 0)$ — a broken instrument carries no information about $F$.

Write $L = p_{\text{ok}}(s\mid 1)/p_{\text{ok}}(s\mid 0)$, the working-mode likelihood ratio, obtained from the per-tool per-slice calibration map (§1.6):

$$L \;=\; \frac{p_{\text{cal}}(F{=}1\mid s)}{1 - p_{\text{cal}}(F{=}1\mid s)} \cdot \frac{1-\pi}{\pi}$$

Then the **effective** likelihood ratio actually used in fusion is

$$\boxed{\;\Lambda(s;r,\pi) \;=\; \frac{r\,L \;+\; (1-r)\big(\pi L + 1 - \pi\big)}{r \;+\; (1-r)\big(\pi L + 1 - \pi\big)}\;}$$

and the evidence contribution is $\Delta = \log \Lambda$.

**Sanity:** $r \to 1 \Rightarrow \Lambda \to L$ (full strength). $r \to 0 \Rightarrow \Lambda \to 1$ (no contribution). $\Lambda$ is monotone in $r$. No free parameter.

> **Proposition 1 (bounded evidence).** For all $s$,
> $$\Big(1 + \tfrac{r}{(1-r)(1-\pi)}\Big)^{-1} \;\le\; \Lambda(s;r,\pi) \;\le\; 1 + \tfrac{r}{(1-r)\pi}$$
> Hence a single packet's log-evidence is bounded by
> $$|\Delta| \;\le\; \max\Big\{\log\!\big(1 + \tfrac{r}{(1-r)\pi}\big),\; \log\!\big(1 + \tfrac{r}{(1-r)(1-\pi)}\big)\Big\}$$

*Proof sketch:* $\Lambda$ is a Möbius transform of $L$, monotone increasing on $L \in [0,\infty)$; evaluate the limits $L \to \infty$ and $L \to 0$. ∎

**Why this matters operationally.** Proposition 1 *derives* the ad-hoc safeguard the blueprint bolts on in §5.9 ("mitigated by flooring exclusion at a reliability threshold"). No packet with $r < 1$ can drive a marginal to 0 or 1. Reliability acquires a reportable physical meaning — an **information cap in nats** — which is a far better headline than a correlation coefficient. Worked example at $\pi = 0.10$:

| $r$ | max $|\Delta|$ (nats, positive direction) |
|---|---|
| 0.50 | 2.40 |
| 0.75 | 3.43 |
| 0.90 | 4.51 |
| 0.99 | 6.90 |

---

# 1.3 M3 — Agreement is double-counted, and it breaks the immutable ledger

§5.8 makes `agreement` one of three inputs to the packet-level score $r$. §5.3 requires packets to be immutable. §5.12 then sums reliability-weighted evidence across those same packets.

Two defects:

**(a) Temporal inconsistency.** Packet 1's `agreement` depends on Packet 2's existence. When Packet 2 arrives, Packet 1's $r$ changes — so $r$ is not a property of a packet, and either immutability or `supersedes`-chain explosion must give.

**(b) Double counting — the serious one.** When three classifiers concur, corroboration raises *each* packet's $r$ **and** contributes three summed terms. Confidence grows super-linearly in the number of agreeing sources. Under M4 (correlated sources), those sources are not independent, so the system becomes **most overconfident exactly when its tools share training data** — reproducing the "manufactured overconfidence" failure the blueprint was written to prevent, inside the module designed to prevent it.

**Correction — split the score.**

```python
class ReliabilityScore(BaseModel):
    r_pre: float            # IMMUTABLE. = g(competence_slice, integrity, ood_score)
                            # Computed at packet creation. The ONLY r used in Λ (M2).
    competence: float       # posterior mean from the hierarchical fit (M7)
    competence_sd: float    # posterior SD — widens r_pre downward when unmeasured
    integrity: float        # preconditions + TTA/perturbation stability
    ood_score: float
    method: str; calibration_ref: str
```

Agreement is **removed from the packet** entirely. Corroboration enters the system exactly once, through the fitted combination weights of §1.4. If a display-level or conflict-severity agreement number is still wanted, it is a **ledger-level field keyed by ledger state hash**, never a packet field, and never an input to fusion.

---

# 1.4 M4 — Correlated evidence and effective independence

The MVP's specialist pool is not independent:

- `mimic_ch` and `mimic_nb` are **the same MIMIC-CXR images**, differing only in NLP labeller (CheXpert vs NegBio). Treating them as two votes is close to a factor-of-two error.
- `chex` and `mimic_*` share the CheXpert labeller → correlated *label noise*, not just correlated features.
- All are DenseNet-121 at 224², so architectural error modes are shared.

Effective independent sources are realistically **3** (NIH-family, CheXpert/MIMIC-family, PadChest), not 5–6.

**Correction — fitted, printable combination.** Per label $j$:

$$\operatorname{logit} q_j \;=\; b_j \;+\; \sum_{t \in \mathcal{T}_j} w_{jt}\, \tilde{\Delta}_{jt}, \qquad \tilde{\Delta}_{jt} = \log \Lambda(s_{jt}; r^{\text{pre}}_{jt}, \pi_j)$$

with $w_{jt} \ge 0$, L2-regularised, fitted on the profiling split; $b_j$ initialised at $\operatorname{logit} \pi_j$. The fitted $w$ absorb the correlation structure directly.

This does **not** violate P1/auditability: $|\mathcal{T}_j|$ coefficients per label is a printable table, and the per-packet contribution $w_{jt}\tilde{\Delta}_{jt}$ is exactly attributable. The additivity assertion the blueprint asks for holds *by construction*.

**Report alongside it:** the pairwise error-correlation matrix $\rho_{tt'}$ from profiling, and the empirical redundancy factor $R_j = |\mathcal{T}_j| \big/ \big(\sum_t w_{jt}\big)$. If $R_j \gg 1$ you have quantified diversity collapse in your own tool pool — a publishable observation in its own right, and directly relevant to §9-R2.

---

# 1.5 M5 — The reliability discount in EIG is not a free parameter (claim-integrity issue)

§5.10 and §9-R4 present "expected information gain **discounted by** $r_t$" as "the coupling that makes this interesting" and "the novel coupling ... no existing planner models that."

Under the corrected M2 likelihood, the discount is **implied, not added**. EIG must be computed under the *same* channel used for fusion:

$$\mathrm{EIG}(t,j) \;=\; I(X; S_{tj}) \;=\; H(X) - \mathbb{E}_{S_{tj}}\big[H(X \mid S_{tj})\big]$$

with $S$ distributed under the mixture of §1.2. As $r \to \tfrac12$ the channel capacity → 0 and EIG → 0 **automatically**. Multiplying a reliability-free EIG by $r$ is a first-order approximation to this, and it will *misrank* tools whose $r$ varies across labels — which is precisely the case you care about.

**This is a claim-integrity problem, not just a maths problem.** A reviewer who knows information theory will point out that "reliability-discounted EIG" is the textbook answer. Relocate the novelty in R1/R4 to what is actually new:

1. **Per-case estimation of $r$** — an OOD-attenuated, hierarchically-shrunk competence posterior conditioned on a case descriptor, before the call.
2. **Joint use of one estimator for two purposes** (selection *and* weighting) with a measured transfer between them.
3. **The bounded-evidence guarantee** (Proposition 1), which Robust-TO's three-tier synthesis does not have.

Those are defensible. "We multiply EIG by reliability" is not.

**v1 implementation:** discretise $S$ into 3–5 bins per (tool, label) at profiling quantiles; store $P(\text{bin} \mid F)$ per slice; marginalise $F$ over the Level-X posterior via the M1 table. Fully precomputable; table size $|T|\times|J|\times|\text{slices}|\times|\text{bins}|\times 2$.

---

# 1.6 M6 — The conformal layer, corrected

Four defects in §5.13 / §4.3-L8.

**(a) Wrong instrument.** Conformal Risk Control (Angelopoulos et al., ICLR 2024) requires a **monotone, bounded** risk in the threshold. Selective risk $R(\tau) = P(\text{err} \mid \text{accept at } \tau)$ is *not* guaranteed monotone in finite samples — accepting more cases usually raises it, but not monotonically. **Learn-then-Test** (Angelopoulos et al., *Ann. Appl. Stat.* 2025) is the correct instrument: test $H_0^\tau : R(\tau) > \alpha$ over a finite grid using a valid p-value for a bounded loss (Hoeffding–Bentkus), with fixed-sequence or Bonferroni correction over the grid. Cite CRC only for the monotone special case.

**(b) Freeze the loss before calibration.** Suggested and declarable in P0:

$$\ell(\text{case}) = \mathbb{1}\big[\text{top-1 explanation incorrect}\big] \;\vee\; \mathbb{1}\big[\exists\, \text{CRITICAL } F_j \text{ true but } q_j < \tau_{\text{report}}\big] \;\in \{0,1\}$$

Bounded in $[0,1]$, so HB p-values apply directly.

**(c) Selection-conditioned calibration.** Rules 1–3 (`TECHNICAL`, `CONFLICT`, `REQUEST`) fire **before** the threshold. The guarantee is therefore conditional on reaching rule 4, and calibration **must be run on the identically-filtered subset**. Expect 30–45% of calibration cases to be filtered out. Budget $n_{\text{cal}}^{\text{raw}} \ge 2{,}000$ to retain $\ge 1{,}200$ effective. The blueprint's 1,000-case calibration set is insufficient at $\alpha = 0.05$ after filtering.

**(d) Accounting.** Rule 5 (`ANSWER_WITH_CAVEAT`) runs *after* rule 4, so caveated answers are **inside** the accepted set. The guarantee covers `ANSWER ∪ ANSWER_WITH_CAVEAT`. `REQUEST` and `ESCALATE` are abstentions for coverage accounting. Reporting a risk guarantee for `ANSWER` alone is invalid. Add to the schema:

```python
class Disposition(BaseModel):
    ...
    in_accepted_set: bool           # True for ANSWER and ANSWER_WITH_CAVEAT only
    ltt_grid_ref: str               # which threshold grid / correction was used
    n_cal_effective: int            # post-filter calibration count
    exchangeability_asserted: bool  # False on every shift arm — forces honest reporting
```

On the OOD arm, exchangeability is violated **by construction**. Report *empirical* selective risk with an exact (Clopper–Pearson) interval and never the word "guarantee."

---

# 1.7 M7 — Competence slices are statistically underpowered

§5.4 slices competence by `{view, source_domain, age_band, sex, ood_bucket}`. Even the reduced MVP slicing (source × view × OOD = 3 × 2 × 3 = 18 cells) gives

$$18 \text{ slices} \times 6 \text{ tools} \times 18 \text{ labels} \;=\; 1{,}944 \text{ cells}$$

A per-cell AUROC needs roughly $n_{\text{pos}} \ge 30$ for a ±0.10 interval. Pneumothorax runs 1–2% prevalence; pneumomediastinum well under 0.5%. Most cells will have $n_{\text{pos}} \in [0, 5]$ — **precisely for the CRITICAL findings where routing decisions matter most.** The blueprint names "declared-but-unmeasured competence" as *fatal*, and the design as specified produces it at scale.

**Correction — hierarchical partial pooling + an honesty flag.**

Empirical-Bayes shrinkage, closed form, no MCMC needed for v1:

$$\hat{p}_{\text{cell}} \;=\; \frac{n_{\text{cell}}\,\hat{p}_{\text{raw}} \;+\; \kappa\,\bar{p}_{(t,j)}}{n_{\text{cell}} + \kappa}, \qquad \kappa = \frac{\bar{p}(1-\bar{p})}{\operatorname{Var}_{\text{between-slice}}} - 1$$

Store **posterior mean and posterior SD**. The SD feeds $r_{\text{pre}}$ downward: an unmeasured slice yields low, not default-high, reliability.

**Hard rule for the capability card:**

```python
class CompetenceSlice(BaseModel):
    descriptor: CaseDescriptor
    n: int; n_pos: int
    measured: bool          # False iff n_pos < 10 — then auroc/ece are None, not defaults
    auroc_mean: float | None; auroc_sd: float | None
    ece_mean: float | None
    prior_pooled_from: str  # which parent cell supplied the shrinkage target
```

If `measured == False`, the router must treat the tool as **maximum-uncertainty** in that slice, never as competent. Add a CI assertion that no card ships a non-`None` metric with `measured=False`.

---

# 1.8 M8 — Metric transplant: `Capture@K` does not type-check on CXR

`Capture@K` originates in CPC differential diagnosis and asks whether the top-K differential is sufficient to make a clinician order the **diagnostic test of record**. A CXR *finding* has no DToR. As written, the metric is undefined and a reviewer will say so immediately.

**Correction.** Author an explicit **action-mapping table** (~20 rows), one row per Level-X explanation and per CRITICAL finding:

| Explanation / finding | Next action of record |
|---|---|
| Pneumothorax | Urgent clinical review + expiratory/lateral film |
| Suspected PE | CTPA |
| Pleural effusion, indeterminate | Lateral view / decubitus / thoracic US |
| Consolidation vs. effusion ambiguity | Lateral view |
| Free subdiaphragmatic air | Urgent surgical review + erect film |
| Normal | None |

Then `Capture@K` = P(the top-K explanations license the correct next action). This is well-defined, clinically meaningful, and — usefully — it is exactly the table the `REQUEST(item)` disposition needs anyway, so it is not extra work. Get it reviewed by a clinician if at all possible; it is the highest-leverage 20 rows in the project.

**Also swap Axis-1 primaries:** finding level → mAP, R@k, per-label AUROC (multi-label). Explanation level → Top-1/Top-3, `Capture@K`, can't-miss recall.

---

# 2. Control-flow soundness

| ID | Defect | Correction |
|---|---|---|
| **CF1** | Rule 3 (`REQUEST`) fires **before** the conformal threshold. The system can route hard cases into `REQUEST` and leave an artificially easy accepted set, inflating apparent selective-risk performance. | Cap the `REQUEST` rate; count `REQUEST` as abstention in every risk–coverage plot; **mandatorily report a `REQUEST`-disabled variant** alongside. |
| **CF2** | Termination on "no tool has positive EIG" — a table-estimated EIG is never exactly zero. | Explicit cost-normalised threshold $\varepsilon$, *calibrated* on the dev set against the "would one more call have changed the disposition?" replay, not guessed. Log which of the four termination conditions fired, always. |
| **CF3** | §4.5's worked example emits `REQUEST(lateral_view)` **with `ESCALATE(CONFLICT)` as fallback**, but §5.13's `Disposition` is a single enum with no fallback field. | Add `fallback: DispositionKind \| None` + `fallback_reason`, or move fallback resolution to the harness. Pick one and freeze it in P1. |
| **CF4** | A conflict found at round $k$ invalidates a packet from round $k-1$ whose $r$ was computed without it. | Resolved automatically by the M3 fix ($r_{\text{pre}}$ immutable, agreement moved to fusion). Without M3, this is a live inconsistency. |
| **CF5** | **Phase-1 gate "empty bundle replays deterministically" is unachievable as stated.** vLLM under continuous batching is not bitwise deterministic — numerics vary with batch composition, so identical prompts can yield different tokens across runs. | Content-address an **LLM response cache**: key = `SHA256(model_id ‖ weight_hash ‖ prompt_hash ‖ sampling_params ‖ seed)`. The Case Bundle stores the mapping; replay reads from cache. Then measure **regeneration stability** (resample $k{=}5$, agreement on `TaskFrame` and `Disposition`) as a *separate, reportable* metric — which is more interesting than determinism anyway. |

Everything else in §4.1 is sound. The bounded loop with four logged termination conditions is a correct response to MAST failure mode 1.5, and the ordered first-match-wins disposition cascade is the right shape.

---

# 3. Practical feasibility

# 3.1 F1 — VRAM budget, corrected

Measured-order estimates for a 24 GB card (RTX 4090 / A5000 class), bf16 unless noted:

| Component | VRAM |
|---|---|
| MedGemma-4B-it (multimodal) weights | ≈ 8.6 GB |
| vLLM KV cache, 8k ctx × 4 concurrent seq | ≈ 1.5 GB |
| vLLM overhead / CUDA graphs | ≈ 2.5 GB |
| TorchXRayVision DenseNet-121 × 5, batch 8 @ 224² | ≈ 1.2 GB |
| Lung/heart segmentation (PSPNet-class) | ≈ 0.6 GB |
| MedSigLIP-400M fp16 | ≈ 0.9 GB |
| DeBERTa-v3-large NLI fp16 (entailment gate) | ≈ 0.9 GB |
| Fragmentation headroom | ≈ 2.0 GB |
| **Total** | **≈ 18.2 GB** |

Fits 24 GB with ~5 GB margin **only if** `gpu_memory_utilization` is set to ≈ 0.55. vLLM's default of 0.90 preallocates ~21.6 GB and starves the entire tool layer — this is the single most common way this build fails on day one.

**16 GB card:** feasible only with a 4-bit LLM (≈ 3.5 GB) and the NLI model loaded on demand rather than resident. Report the quantisation quality delta; do not hide it.

**`MedGemma-27B-text` does not fit 24 GB in bf16** (≈ 54 GB). At 4-bit it is ≈ 15–16 GB, leaving nothing for tools. §7.1's "27B-text as a text-only upgrade if VRAM allows" is misleading on the stated hardware. Correct to: *27B is a serialised second pass on a separate run, or a different machine, and is out of scope for v1.*

# 3.2 F2 — The multi-source premise must be verified before anything else is built

§6.2 reason (1) — "the only domain where you get a heterogeneous pool of specialists for free" — is the load-bearing justification for the entire modality choice, the Tool Router, and the Reliability Estimator.

But TorchXRayVision's dataset-specific weights **do not all support the same 18 pathologies**. Each weight supports the subset present in its training corpus; unsupported outputs are untrained. Therefore:

- **Inter-tool agreement exists only on the label intersection.**
- The "genuinely different per-domain competence" story exists only on that intersection.
- `rsna` supports essentially pneumonia/lung-opacity only — it is not a general specialist.
- `mimic_ch` / `mimic_nb` are one source, not two (§1.4).

**Expected intersection with ≥3 genuinely distinct sources: 6–10 labels, not 18.**

> **This is the highest-priority item in the entire project and it belongs in P0-Day-1.** Enumerate it empirically — load every weight, read its supported-pathology mask, build the intersection matrix — *before* the hypothesis space is frozen, *before* datasets are downloaded, *before* a line of pipeline code is written. If the intersection is smaller than ~6, the multi-source premise is weaker than the blueprint assumes and the scope must be renegotiated then, not in Phase 3.

# 3.3 F3 — The OOD substrate is much smaller than assumed

Verified from the challenge overview (`arXiv:2602.22092v2`; also appears as `2604.15555`):

<cite index="44-1">Task 1 training draws 142,928 CXRs from PadChest with report-derived labels; development is 1,620 radiologist-annotated PadChest-GR images and test is 1,400. Task 2 uses 200 development, 305 internal test, and 79 external NIH test images, all radiologist-annotated.</cite>

Two corrections follow:

1. **n = 305 internal + 79 external**, not the 500-case OOD set §6.3 budgets. At $n = 79$, a selective-risk estimate has a Clopper–Pearson half-width of roughly ±0.11 at $p = 0.5$. Subgroup slicing on this arm is statistically meaningless — do not attempt it.
2. **It is label shift, not domain shift.** <cite index="44-1">Task 2 assesses recognition of 6 "unseen" classes strictly excluded from training, with labels derived from PadChest-GR</cite> — same institution, same acquisition. The blueprint conflates open-set shift with cross-centre shift.

**Restructure the shift arms:**

| Arm | Construction | Size | What it tests |
|---|---|---|---|
| **S2a — domain shift (primary)** | Profile on NIH + CheXpert slices, evaluate on PadChest. Self-constructed via TXRV dataset wrappers. | 1,000+ | Competence-slice transfer; calibration-map breakdown; conformal validity collapse |
| **S2b — open-set** | CXR-LT 2026 Task 2 | 305 + 79 | `OTHER/OUT_OF_SCOPE` hypothesis winning; R6 under-coverage |
| **S2c — corruption** | Controlled injection, self-constructed | any | Propagation / detection / recovery |

S2a becomes primary. S2b is reported with exact intervals and no slicing.

**Bonus finding:** <cite index="37-1">PadChest-GR is a radiologist-annotated grounded subset reserved for development and evaluation</cite>. It gives **sentence↔box grounded report** ground truth on PadChest. That is precisely the substrate RadFact-style evaluation requires and it is absent from the blueprint's dataset list. It is a better fit for R5 (entailment-gated generation) than anything currently listed. **Add it.**

# 3.4 F4 — Dataset/experiment assignment is wrong in places

**VinDr-CXR is PA-only.** The `REQUEST(lateral_view)` disposition, the missing-modality ablation (§8-Axis 5), and request-specificity replay **cannot run on VinDr**. Corrected assignment:

| Experiment | Correct dataset | Why |
|---|---|---|
| Localisation / grounding metrics | VinDr-CXR (boxes) | Only open source of radiologist boxes |
| Lateral ablation, `REQUEST` replay | **PadChest or CheXpert** | Both contain laterals; VinDr does not |
| Grounded-report entailment (R5) | **PadChest-GR** | Sentence↔region ground truth |
| Competence profiling, domain shift | TXRV wrappers over NIH / CheXpert / PadChest | Source-labelled |
| Open-set / unseen class | CXR-LT 2026 Task 2 | Only 6-unseen-class split with published SOTA |
| Clean-label reliability headline | VinDr test (5-rad consensus), PadChest-GR, CXR-LT dev | Report-derived labels are too noisy for the headline number |

# 3.5 F5 — The evaluation grid does not fit without a cache

Per-case cost on the stated hardware:

- Tool layer (5 classifiers + segmentation + geometry + MedSigLIP + 5× TTA): **≈ 0.4 s**
- LLM layer (parse ≈ 1 s; router × ≤4 rounds ≈ 4 s; prose ≈ 2.5 s; local NLI gate ≈ 0.2 s): **≈ 8–12 s**

So ≈ **10 s/case** → 1,000 cases ≈ **2.8 GPU-hours per pass**. The blueprint specifies 8 axes × 3 settings × ~4 baselines ≈ **90 experiment cells** → **≈ 250 GPU-hours ≈ 11 days continuous**, with zero iteration budget. That is not survivable.

**The fix is architectural, and it belongs in P2.** A content-addressed deterministic cache:

```
key = SHA256(tool_id ‖ tool_version ‖ input_hash ‖ seed ‖ canonical_params)
```

with a parallel LLM response cache (CF5). With it:

- Every ablation that changes **fusion, reliability, conflict, disposition, or generation logic** becomes a **CPU-only replay** at ~50 ms/case → 1,000 cases in **≈ 1 minute**.
- Only cells that change **tool inputs** (image corruption, modality ablation) or **the LLM** (backbone swap, paraphrase) require GPU.

That reduces ~90 GPU passes to roughly **12–15**. This is the difference between a project that produces results and one that produces a demo.

# 3.6 F6 — Two baselines are not what the blueprint thinks they are

**MedRAX.** Its default reasoning core is a hosted frontier model, and its 8 tools include large specialist VLMs. Reproducing it locally on one 24 GB card, alongside HERALD, is not realistic in the time available. **Demote to citation-only.** The honest local baseline — and, conveniently, the *correct* ablation — is what §8 already lists as baseline (c): **ReAct over HERALD's own tool set with the reliability layer disabled.** Same tools, same data, same backbone; the only difference is the mechanism under test. That is a much cleaner scientific comparison than a cross-system accuracy race. Treat ChestAgentBench as optional external comparability, not a gate.

**MedCTA.** Verified: <cite index="18-1">107 clinician-verified tasks over 5 deployed tools; the best agentic outcome accuracy is 31.54% and no model achieves non-zero strict trajectory success</cite>, and <cite index="21-1">its tool library covers perception, OCR, retrieval, and calculation, with segmentation noted as a future extension</cite>. Two implications: (i) it is **not a usable evaluation target for HERALD** — the tool taxonomy does not overlap; (ii) at $n = 107$ its controller-gap estimate carries wide intervals, so cite it as *motivation* ("routing is empirically brittle") and never as a precise numerical target.

**AgentProp-Bench's 0.62.** Verified: <cite index="11-1">a parameter-level injection propagates to a wrong final answer with human-calibrated probability approximately 0.62 (range 0.46–0.73)</cite>, across four general domains and nine production LLMs. Note the mechanism: **parameter-level injection into agent-constructed tool arguments**, not corruption of tool *outputs* in an imaging pipeline. These are different failure channels.

> **Therefore the Phase-5 gate "propagation rate materially below the ≈0.62 reference" is invalid as written.** Different corruption mechanism, different task family, different modality. The valid gate is the **internal ablation delta**: HERALD vs. HERALD-with-reliability-disabled, same injections, same cases, paired bootstrap. Report 0.62 as context, never as the bar you cleared.

# 3.7 F7 — Timeline

| Phase (this roadmap) | Blueprint | Realistic |
|---|---|---|
| P0 Spec freeze + feasibility probes | 0.5 wk | **1 wk** |
| P1 Contracts, state, audit spine, hooks | 1 wk | **1.5 wk** |
| P2 Tool layer + cache + ingestion | 1.5 wk | **2 wk** |
| P3 Profiling + calibration + competence posteriors | 2 wk | **2.5 wk** |
| P4 Packets + fusion + hypothesis/explanation | (folded) | **2 wk** |
| P5 Reliability + conflict | 2 wk | **2 wk** |
| P6 Acquisition planner + router | (in Ph3) | **2 wk** |
| P7 Disposition (LTT) + generation + gate | (in Ph4) | **2 wk** |
| P8 Stress harness + full evaluation | 2 wk | **3 wk** |
| P9a Governance attach | 1 wk | **1 wk** |
| P9b Research extension | 4+ wk | **4–6 wk** |
| **Total** | **≈ 17.5 wk** | **≈ 22–24 wk** |

A ~30% underestimate. §4 of Part II gives a **de-scoped 12-week critical path** that still yields a defensible result.

---

# 4. Failure-mode coverage

# 4.1 Well covered

| Documented failure | HERALD mechanism | Adequate? |
|---|---|---|
| Cerebra halt on missing modality | `MissingEvidence` as first-class type; mask-and-renormalise | **Yes** — but see caveat below |
| CARE coordinator over-editing | Generator sees ledger only, never images; entailment gate | **Yes**, conditional on measured gate precision |
| Meissa silent failure / no abstention | Typed 4-way disposition + LTT threshold | **Yes**, after M6 |
| MAST specification failures (~42%) | Pydantic contracts, registry, tool-removal CI | **Yes** |
| MedCTA under-calling | Forced-coverage rule before termination | **Yes** |
| Untraceable decisions | OTel spine + Case Bundle | **Yes**, after CF5 |
| MedAgent-Pro −16.5 F1 from one weak tool | Bounded evidence (Proposition 1) + $r_{\text{pre}}$ | **Yes, and stronger than before** — the bound is provable, not empirical |

> **Caveat on halt rate.** With `MissingEvidence` in the schema and `ESCALATE` always reachable, halt rate = 0 is **true by construction**. It is an *invariant to assert in CI*, not a result to report. Reporting "halt rate 0 vs. Cerebra" is a weak claim. The real quantity is **information loss under missingness** — the joint accuracy/calibration degradation curve — which §8-Axis 5 already specifies. Lead with that.

# 4.2 Uncovered or under-covered

| # | Gap | Severity | Recommended treatment |
|---|---|---|---|
| **G1** | **Correlated-source overconfidence** — the system's own diversity collapse. As specified (M3+M4) HERALD *creates* it. | **Critical** | M3 + M4 fixes; report $\rho_{tt'}$ and redundancy factor $R_j$ as first-class results |
| **G2** | **Ground-truth label noise.** PadChest labels are report-derived; NIH labels are text-mined. Reliability AUROC measured against noisy labels is biased downward, non-uniformly across labels. | High | Headline reliability number computed **only** on clean-label subsets (VinDr 5-rad test, PadChest-GR, CXR-LT radiologist dev). Report noisy-label numbers separately and label them as such. |
| **G3** | **Calibration-map transfer failure.** The per-tool per-slice Platt/isotonic maps break under shift *before* the reliability estimator does. The blueprint never measures this layer separately. | High | Add a standalone metric: ECE of $p_{\text{cal}}$ on ID vs. S2a. This will likely be one of the cleanest findings in the project. |
| **G4** | **Estimator overfits the injected corruption taxonomy.** If you fit and evaluate on the same 6 corruption types, clean-vs-corrupt AUROC is optimistic. | High | **Hold out 2 corruption types entirely** from any fitting. Declare the split in P0 and never touch it. |
| **G5** | **Entailment gate causes information loss and incoherence.** Measured but unmitigated. | Medium | Gate at **claim level**, then regenerate once with failed claims excluded, rather than deleting sentences post hoc. Report drop rate *and* a coherence rating. |
| **G6** | **Indication-text anchoring is an adversarial prior.** A wrong indication seeds a wrong Level-X prior with no correction path. | Medium | Promote "indication contradicts image" from a stress note to a first-class family with its own metric: **prior-override rate** = P(posterior correctly overrides a false indication prior). |
| **G7** | **Out of scope — state it explicitly.** Adversarial images, prompt injection via report text, DICOM parser attacks, multi-tenant isolation. | — | One paragraph in the limitations section. Silence here reads as oversight. |

# 4.3 One structural observation

The blueprint's central thesis — *"the interesting engineering in a clinical agent is not the reasoning, it is the epistemics"* — is correct and is the strongest thing in the document. But as specified, the epistemics module contains the sharpest mathematical error (M3's double-counting). **The system would fail in the exact way it was designed to prevent, in the exact module designed to prevent it.** That is not an argument against the design; it is an argument for building the fusion mathematics *first, on paper, with a proof* — which is why P0 of the roadmap below is a specification phase with a written derivation as its exit artifact, not a literature phase.

---

# PART II — CHRONOLOGICAL IMPLEMENTATION ROADMAP

# Reading instructions

Ten phases, **P0 → P9**. Each phase consumes the exit artifacts of its predecessors and nothing else. Each has:

- **Objective & scope** — what exists at the end that did not exist at the start
- **Entry state** — exact artifacts required to begin
- **Exit artifacts** — exact files, schemas, and numbers produced
- **Validity & alignment gate** — binary or numeric criteria; *do not proceed if unmet*
- **Task checklist** — granular technical steps
- **Prompting seam** — a self-contained block to paste into a fresh Claude session to request production code

# One structural change from the blueprint's roadmap

The blueprint puts **routing (Phase 3) before fusion (Phase 4)**. This is a dependency inversion: the router's expected-information-gain objective is defined *over the hypothesis posterior*, and the posterior-update machinery does not exist until fusion is built. Building EIG against a posterior that does not yet exist means building it against a placeholder and rewriting it later.

**Corrected order:** contracts (P1) → tools + cache (P2) → profiling + calibration (P3) → **fusion + hypothesis layer (P4)** → **reliability (P5)** → **routing/planning (P6)** → disposition + generation (P7) → stress (P8) → governance + research (P9).

Routing is the *last* reasoning component built because it depends on everything else. This also means that if the schedule slips, you lose the router — and a fixed-order pipeline with correct epistemics is a far better outcome than a router over broken epistemics.

---

# P0 — Specification freeze and feasibility probes
**Duration: 1 week. Nothing else may start until this closes.**

# Objective & scope
Convert the blueprint from prose into (a) a frozen mathematical specification, (b) three empirically-verified feasibility facts, and (c) frozen vocabularies. No pipeline code. The output is documents and small probe scripts.

This phase exists because three of the blueprint's premises are unverified and **load-bearing**: the TorchXRayVision label intersection (F2), the VRAM budget (F1), and the OOD set size (F3). If any fails, the scope changes now — not in week 9.

# Entry state
- `HERALD_blueprint.md`, `HERALD_Research.md`
- A machine with the target GPU
- Nothing else

# Exit artifacts

| Artifact | Content |
|---|---|
| `docs/MATH_SPEC.md` | Full derivation of §1.1–§1.6 of this audit: two-level state, noisy-channel Λ with Proposition 1 and its proof, fitted combination weights, EIG under the same channel, LTT loss function and accounting. **Every equation the code will implement, written out before any code exists.** |
| `probes/txrv_label_matrix.csv` | Empirical support matrix: rows = TXRV weights, cols = pathologies, cells ∈ {supported, unsupported}. Plus the derived intersection set and per-label source count. |
| `probes/vram_report.md` | Measured resident VRAM for every model in the stack, loaded together, with the `gpu_memory_utilization` value that makes it fit. |
| `docs/vocabulary.yaml` | Frozen: finding set $J$ (from the intersection, **not** from the blueprint's 18), explanation set $\mathcal{X}$, UMLS CUIs, CRITICAL flags. |
| `docs/tables/finding_explanation.csv` | The $J \times K$ table $P(F_j{=}1 \mid x_k)$ (M1) |
| `docs/tables/action_map.csv` | Explanation/finding → next action of record (M8) |
| `docs/tables/discrimination.csv` | Hypothesis pair → discriminating observation (~40 rows, from §5.9) |
| `docs/capability_map.md` | failure → module → metric → target (blueprint's Phase-0 deliverable, retained) |
| `docs/datasets.md` | Licence, access status, size, download command, **and which experiments each dataset is valid for** (per §3.4) |
| `docs/EVAL_PROTOCOL.md` | Frozen: the LTT loss ℓ, the α grid, the corruption taxonomy **with 2 types held out**, the clean-label subsets, the split manifest with hashes |
| `docs/CITATION_AUDIT.md` | Every arXiv ID in the blueprint, resolved or flagged (see Part III) |

# Validity & alignment gate

| Check | Threshold | If failed |
|---|---|---|
| TXRV label intersection with ≥3 **distinct** sources (`mimic_ch`/`mimic_nb` count as one) | $\ge 6$ labels | If < 6, the multi-source premise is weak → escalate to supervisor and consider adding a non-TXRV specialist (e.g. a CheXpert-trained public checkpoint) before proceeding |
| Full stack co-resident VRAM | fits with ≥ 3 GB headroom | Drop to 4-bit LLM; record the quality cost as a planned experiment |
| Every equation in `MATH_SPEC.md` has a worked numerical example | 100% | Do not proceed — untested maths becomes untestable code |
| Every dataset in `datasets.md` has a *verified* download path | 100% | Reassign the dependent experiment |
| Every arXiv ID resolved or explicitly flagged unresolvable | 100% | Flag in the eventual write-up |

# Task checklist

1. **[Day 1 — highest priority] TXRV probe.** `pip install torchxrayvision`; load each of `densenet121-res224-{nih,chex,mimic_ch,mimic_nb,pc,rsna,all}`; read each model's supported-pathology mask; write the matrix; compute the intersection under the "distinct source" rule. **Report the number to your supervisor the same day.**
2. **VRAM probe.** Load MedGemma-4B under vLLM at `gpu_memory_utilization` ∈ {0.5, 0.55, 0.6}; then load all TXRV models + segmentation + MedSigLIP + NLI in the same process group; record `nvidia-smi` peak.
3. **Write `MATH_SPEC.md`.** Derive Λ; prove Proposition 1; write the Jeffrey-conditioning update; define the combination-weight fit; define EIG under the mixture; define ℓ and the LTT procedure with the selection-conditioning argument. Include a numerical worked example for each.
4. **Freeze vocabularies** from the intersection, not from the blueprint's aspiration.
5. **Author the three tables** (finding↔explanation, action map, discrimination). These are the highest-value hand-written artifacts in the project. Seek clinician review.
6. **Freeze the corruption taxonomy** and designate 2 held-out types (G4). Write them into `EVAL_PROTOCOL.md` and do not look at them again until P8.
7. **Freeze the split manifest**: dev / calibration (≥ 2,000 raw, per M6c) / test / S2a domain-shift / S2b open-set. Hash every file list.
8. **Citation audit** (Part III gives the starting register).
9. Submit the PhysioNet credentialing application. Treat as optional.

# Prompting seam

```
CONTEXT: I am building HERALD, a reliability-aware clinical agent for chest radiography.
I have a frozen mathematical specification (attached: MATH_SPEC.md) and need P0 probe scripts.

TASK: Generate production Python for three probes.

1. probes/txrv_label_matrix.py
   - Load every torchxrayvision densenet121-res224-* weight
   - Extract each model's supported pathology list programmatically (do not hardcode)
   - Emit a CSV support matrix and a JSON summary containing:
       intersection_all, intersection_min3_distinct_sources
     where mimic_ch and mimic_nb are collapsed to a single source "mimic"
   - Print a clear PASS/FAIL against a threshold of 6 labels

2. probes/vram_probe.py
   - Sequentially load: vLLM(MedGemma-4B, gpu_memory_utilization param),
     5x TXRV classifiers, a segmentation model, MedSigLIP, a DeBERTa NLI model
   - Record torch.cuda.max_memory_allocated() and nvidia-smi after each load
   - Sweep gpu_memory_utilization over [0.45,0.5,0.55,0.6,0.65]
   - Emit a markdown table and the largest value that leaves >=3GB free

3. probes/split_manifest.py
   - Given dataset roots, build dev/calibration/test/shift splits with a fixed seed
   - Enforce: patient-level disjointness (never image-level), stratification by label
     prevalence, calibration split >= 2000 raw
   - Emit JSON with per-file SHA256 and a manifest hash

REQUIREMENTS: type hints, argparse CLIs, no network at runtime beyond model download,
deterministic seeds, structured logging, pytest for the split-disjointness invariants.
```

---

# P1 — Contracts, state, audit spine, governance seams
**Duration: 1.5 weeks. No clinical intelligence whatsoever.**

# Objective & scope
Every Pydantic schema, the LangGraph state machine with pass-through nodes, the OTel spine, the deterministic cache interface, and the five no-op governance hooks. An end-to-end run over one image produces a valid, complete, **empty** Case Bundle.

This is the week five of the ten blueprint principles are either implemented or permanently lost (P1, P6, P9, P10, and — via the cache interface — the feasibility of P8).

# Entry state
All P0 exit artifacts, gate passed.

# Exit artifacts
- `herald/schemas/*.py` — all models from §5 **with the audit corrections**: `ReliabilityScore.r_pre` (M3), `Disposition.{in_accepted_set, ltt_grid_ref, n_cal_effective, exchangeability_asserted, fallback}` (M6d, CF3), `CompetenceSlice.{measured, auroc_sd, prior_pooled_from}` (M7), two-level `FindingLedger` + `ExplanationLedger` (M1)
- `herald/graph/` — LangGraph graph, all nodes pass-through, checkpointer enabled
- `herald/audit/` — OTel GenAI spans, content in **span events**; Case Bundle writer
- `herald/cache/` — content-addressed cache **interface** (`get/put/key`), filesystem backend
- `herald/governance/hooks.py` — five frozen-signature no-ops
- `tests/` — `test_schemas.py`, `test_hook_coverage.py`, `test_replay_from_cache.py`, `test_ledger_immutability.py`
- `docker-compose.yml` — app, vLLM, Langfuse, MLflow

# Validity & alignment gate

| Check | Threshold |
|---|---|
| Empty Case Bundle validates against schema and round-trips | 100% |
| Replay **from cache** reproduces the bundle byte-identically (CF5 — *not* regeneration) | 100% |
| Every hook returning `allow=False` halts the pipeline with the correct disposition and a complete trace | 5/5 hooks |
| CI assertion: every `@irreversible`-decorated function is preceded by a hook call | passes |
| Ledger immutability: no in-place packet mutation reachable | property test passes |
| No PHI-class field in an indexed OTel attribute | scan passes |

# Task checklist
1. Schemas first, with `schema_version` and `trace_id` on every model. Generate JSON Schema and commit it — schema drift becomes a reviewable diff.
2. `CaseState` as a LangGraph `TypedDict` with append-only reducers; `state_hash` after every write.
3. Packets immutable; corrections create a new packet with `supersedes`. Enforce with a property-based test (Hypothesis) that attempts mutation through every public path.
4. OTel: `invoke_agent` root, `chat` per LLM call, `execute_tool` per tool. **Message and image content into span events, never attributes.** Pin the semconv version explicitly.
5. Cache interface: `key(tool_id, version, input_hash, seed, params) -> str`, `get/put`, plus an LLM-response variant keyed on `(model_id, weight_hash, prompt_hash, sampling_params, seed)`.
6. Five hooks with frozen signatures; one integration test per hook proving deny is honoured end to end.
7. `@irreversible(hook=...)` decorator + registry + the CI assertion.
8. Case Bundle writer: input manifest with per-file hashes, model/weight versions, tool versions, full ledgers, disposition, resource accounting, **cache keys used**.

# Prompting seam

```
CONTEXT: HERALD, a reliability-aware clinical agent for CXR. P0 is complete: I have a frozen
math spec, frozen vocabularies, and a verified tool-label intersection. I am now building the
contract layer with zero clinical logic.

TASK: Generate the complete P1 skeleton.

SCHEMAS (Pydantic v2, herald/schemas/):
  evidence.py    EvidencePacket, MissingEvidence, ReliabilityScore
                 ReliabilityScore has r_pre (immutable, = f(competence, integrity, ood)),
                 competence_sd, and NO agreement field
  hypothesis.py  Finding, FindingLedger (multi-label marginals, NO renormalisation),
                 Explanation, ExplanationLedger (exclusive simplex)
  case.py        CaseState, AvailabilityMap, TaskFrame, BudgetState
  tools.py       CapabilityCard, CompetenceSlice (with measured: bool, auroc_sd,
                 prior_pooled_from), ToolResult, AcquisitionAction
  disposition.py Disposition (kind, reason_code, in_accepted_set, ltt_grid_ref,
                 n_cal_effective, exchangeability_asserted, fallback), Conflict,
                 GovernanceDecision[T]
  All models carry schema_version and trace_id.

GRAPH (herald/graph/): LangGraph StateGraph, nodes L1..L9 as pass-throughs, SqliteSaver
  checkpointer, typed reducers enforcing append-only semantics on all ledgers.

AUDIT (herald/audit/): OpenTelemetry GenAI conventions. invoke_agent root span, chat and
  execute_tool children. ALL message/image content in span EVENTS, never attributes.
  CaseBundle JSONL writer with input manifest hashes and resource accounting.

CACHE (herald/cache/): content-addressed store, filesystem backend.
  tool_key(tool_id, version, input_hash, seed, params) and
  llm_key(model_id, weight_hash, prompt_hash, sampling_params, seed).
  Bundles record every cache key used so replay is exact.

GOVERNANCE (herald/governance/): five hooks pre_ingest, pre_tool_call, pre_fusion,
  pre_response, pre_persist. Signature frozen:
    def hook(payload: T, ctx: GovernanceContext) -> GovernanceDecision[T]
  All return allow=True in v1. Plus @irreversible(hook=...) decorator and registry.

TESTS: schema round-trip; ledger immutability via Hypothesis property tests; deny-honoured
  integration test per hook; replay-from-cache byte equality; CI assertion that every
  @irreversible function is preceded by its hook.

REQUIREMENTS: full type hints, docstrings citing the layer (L0..L11), no clinical logic,
pytest, Hydra config, docker-compose for app/vLLM/Langfuse/MLflow.
```

---

# P2 — Tool layer, deterministic cache, ingestion
**Duration: 2 weeks.**

# Objective & scope
Real tools behind the uniform wrapper contract, real ingestion with quality checks and the `AvailabilityMap`, and a **populated** cache. Fixed tool order, naive scoring, no routing, no reliability. The goal is to close the loop and produce a first number.

# Entry state
P1 schemas and cache interface; P0 vocabulary and split manifest; datasets downloaded.

# Exit artifacts
- `herald/tools/` — wrappers for: TXRV classifiers × N (N from the P0 intersection), lung/heart segmentation, the deterministic geometric suite (CTR, rotation index, inspiration level, costophrenic angles, tracheal deviation), MedSigLIP k-NN retriever, guideline retriever
- `herald/ingest/` — DICOM/PNG handling, view detection, four quality checks, `AvailabilityMap` construction with `absent_expected`
- `cache/` — populated for the full dev + calibration + test splits
- `app/` — Gradio single-case explorer (image, packet table, ledgers, trace link)
- `results/P2_baseline.json` — first accuracy numbers, S1 clean

# Validity & alignment gate

| Check | Threshold |
|---|---|
| Determinism: identical input → identical `ToolResult`, 100 repeats | 100% |
| Cache hit rate on a re-run of the dev split | ≥ 99% |
| Wall-clock for a cached replay of 1,000 cases | ≤ 3 min (validates F5) |
| Precondition-violation catch rate on deliberately malformed inputs | 100%, zero `UNKNOWN` error classes |
| View-classification accuracy vs. dataset metadata | ≥ 0.95 |
| Gap-detection recall on synthetically ablated studies | ≥ 0.98 |
| Single best classifier macro-AUROC on dev | within 0.03 of the published TXRV figure — *if not, the preprocessing is wrong* |

> The last row catches the single most common silent failure in this kind of build: a preprocessing mismatch between wrapper and model, which costs ~10 accuracy points invisibly.

# Task checklist
1. Abstract `ToolWrapper`: precondition assertion → fixed seed → timeout with a **typed** error → mandatory `native_uncertainty` capture → optional TTA replicates. **Wrappers never interpret, summarise, or threshold.**
2. Error taxonomy: `PRECONDITION | TIMEOUT | RUNTIME | EMPTY_OUTPUT | SCHEMA_VIOLATION`. No `UNKNOWN` permitted.
3. TXRV wrappers restricted to each model's **supported** labels (F2). Unsupported labels emit nothing — never a default.
4. Geometric suite on the segmentation mask, in pure NumPy/OpenCV, fully deterministic. Tag `tool_class=VERIFICATION`.
5. Ingestion: DICOM header parse, view detection (MedSigLIP zero-shot), four CXReasonAgent-style quality checks, `AvailabilityMap` with `absent_expected` from a per-task template.
6. Cache warm-up job over all splits, with a resumable driver.
7. Gradio explorer. Build it now — it is the demo, and having it early changes the character of every supervisor meeting.
8. Fixed-order pipeline → first S1 numbers.

# Prompting seam

```
CONTEXT: HERALD P2. P1 contract layer complete (schemas, LangGraph graph, OTel, cache
interface, hooks). P0 determined the supported label intersection: <PASTE FROM
probes/txrv_label_matrix.json>.

TASK: Generate the tool and ingestion layer.

WRAPPERS (herald/tools/): abstract ToolWrapper enforcing
  - precondition assertion from the CapabilityCard before invocation
  - fixed seed; torch.use_deterministic_algorithms(True); CUBLAS_WORKSPACE_CONFIG=:4096:8
  - timeout with a TYPED error, never a silent empty result
  - error taxonomy PRECONDITION|TIMEOUT|RUNTIME|EMPTY_OUTPUT|SCHEMA_VIOLATION, no UNKNOWN
  - mandatory native_uncertainty capture (logits, mask-area variance, retrieval score)
  - optional TTA replicates with seeded augmentation
  - cache read-through using herald.cache.tool_key
  Wrappers must NEVER interpret, summarise, or threshold.

Concrete wrappers:
  xrv_classifier.py   one per weight, RESTRICTED to that weight's supported labels
  segmentation.py     lung + heart masks
  geometry.py         cardiothoracic ratio, rotation index, inspiration level,
                      costophrenic angles, tracheal deviation. Pure NumPy/OpenCV,
                      deterministic, tool_class=VERIFICATION
  retrieval.py        MedSigLIP kNN over a labelled index
  guidelines.py       local corpus retriever

INGESTION (herald/ingest/): DICOM/PNG loader, view detection, four quality checks
  (inspiration, rotation, projection, inclusion), AvailabilityMap with absent_expected
  from a per-task template.

CACHE DRIVER (scripts/warm_cache.py): resumable, parallel, progress-reporting.

APP (app/gradio_explorer.py): image + overlays, packet table, ledgers, disposition, trace link.

TESTS: 100-repeat determinism; malformed-input precondition catch; cache hit rate;
  preprocessing regression test asserting single-model macro-AUROC within 0.03 of published.

REQUIREMENTS: type hints, Hydra config, structured logging, no interpretation in wrappers.
```

---

# P3 — Profiling, calibration maps, competence posteriors
**Duration: 2.5 weeks. This is the phase everything downstream depends on.**

# Objective & scope
Produce the three statistical artifacts that make routing and reliability possible: per-tool per-slice **calibration maps**, hierarchically-shrunk **competence posteriors**, and the **evidence-bin likelihood tables** for EIG.

This is also, as the blueprint correctly notes, the training data for research extension R1. Build it carefully.

# Entry state
P2 populated cache; P0 split manifest and vocabulary.

# Exit artifacts

| Artifact | Content |
|---|---|
| `profiling/calibration_maps.pkl` | Per (tool, label, slice): isotonic or Platt map $s \mapsto p_{\text{cal}}(F{=}1\mid s)$ |
| `profiling/competence_posteriors.parquet` | Per (tool, label, slice): shrunk mean, SD, $n$, $n_{\text{pos}}$, `measured` flag, `prior_pooled_from` |
| `profiling/error_correlation.npz` | Pairwise error-correlation matrix $\rho_{tt'}$ per label |
| `profiling/evidence_bins.parquet` | $P(\text{bin} \mid F)$ per (tool, label, slice, bin) — the EIG likelihood table |
| `registry/cards/*.yaml` | Capability cards with **measured** competence, never declared |
| `results/P3_profiling_report.md` | Slice coverage, `measured` fraction, correlation structure, redundancy factors |

# Validity & alignment gate

| Check | Threshold | Why |
|---|---|---|
| Calibration ECE per (tool, label) on ID held-out | ≤ 0.05 | If the map is not calibrated, nothing downstream is |
| Fraction of (tool, label, slice) cells with `measured=True` | reported, and **≥ 0.5 for non-rare labels** | Quantifies M7 honestly |
| CI assertion: no card ships a non-`None` metric with `measured=False` | passes | Prevents the blueprint's own named fatal failure |
| Mean pairwise error correlation $\bar\rho$ across TXRV tools | **reported, whatever it is** | If $\bar\rho > 0.7$ the multi-source premise is empirically weak — say so |
| Competence posteriors have non-degenerate between-slice variance | $\operatorname{Var} > 0$ | If tools do *not* differ across slices, routing has nothing to exploit and the thesis needs re-framing **now** |

> The last check is the project's real go/no-go. If competence does not vary meaningfully by slice, the Tool Router cannot beat "always call the best tool," and you should pivot the primary claim toward R3 (disclose-and-degrade) or R5 (entailment gating), both of which survive that outcome.

# Task checklist
1. Define `CaseDescriptor`: `{view, source_domain, ood_bucket, age_band, sex}`. OOD bucket from MedSigLIP Mahalanobis distance to each tool's training-source embedding centroid (sample 2,000 images per source).
2. Run all tools over dev + calibration splits **from cache** — this should be minutes, not hours. If it is not, the P2 cache is broken.
3. Fit per (tool, label, slice) isotonic regression (or Platt where $n$ is small); hold out a nested split for ECE.
4. Compute raw per-cell competence; then empirical-Bayes shrinkage toward the (tool, label) parent with $\kappa$ estimated from between-slice variance. Store mean **and SD**.
5. Set `measured = (n_pos >= 10)`. Null out metrics where `False`.
6. Pairwise error correlation $\rho_{tt'}$ per label, on the calibration split.
7. Discretise scores into 3–5 quantile bins; tabulate $P(\text{bin} \mid F)$ per slice.
8. Emit capability cards. Add a `card_drift` job that re-profiles and diffs.
9. **Validate the OOD score does what you think.** Check it is not simply tracking acquisition artifacts: condition on label and verify the OOD distribution shifts across *sources* more than across *pathology presence*. If it does not, the OOD term in $r_{\text{pre}}$ is measuring the wrong thing.

# Prompting seam

```
CONTEXT: HERALD P3. P2 complete with a populated deterministic cache over all splits.
I need the offline profiling job that produces calibration maps, competence posteriors,
error correlations, and EIG likelihood tables.

TASK: Generate profiling/ as a runnable pipeline.

1. descriptors.py
   CaseDescriptor = {view, source_domain, ood_bucket, age_band, sex}
   OOD bucket via MedSigLIP embedding Mahalanobis distance to per-source training centroids
   (sample 2000 images per source; fit shrunk covariance).

2. calibrate.py
   Per (tool, label, slice): fit isotonic regression, fall back to Platt when n<200.
   Nested held-out ECE. Persist to calibration_maps.pkl with a fit manifest.

3. competence.py
   Raw per-cell AUROC/F1/ECE, then empirical-Bayes shrinkage toward the (tool,label) parent:
       p_hat = (n*p_raw + kappa*p_parent) / (n + kappa)
       kappa = p_parent*(1-p_parent)/Var_between_slice - 1
   Store posterior mean AND SD. Set measured = (n_pos >= 10); null metrics when False.

4. correlation.py
   Pairwise error-correlation matrix rho[t,t'] per label on the calibration split.
   Collapse mimic_ch and mimic_nb into one source for the reporting view.
   Emit the empirical redundancy factor per label.

5. evidence_bins.py
   Quantile-discretise each tool score into 3-5 bins; tabulate P(bin | F) per slice.
   This is the EIG likelihood table consumed in P6.

6. emit_cards.py
   Write registry/cards/*.yaml. Assert: no non-None metric with measured=False.

7. ood_validation.py
   Verify the OOD score separates SOURCES more than it separates PATHOLOGY PRESENCE.
   Emit both AUROCs and fail loudly if pathology dominates.

8. report.py -> results/P3_profiling_report.md
   Slice coverage, measured fraction, mean pairwise rho, between-slice competence variance
   with a go/no-go verdict on whether routing has exploitable signal.

REQUIREMENTS: reads ONLY from cache (no GPU inference), parquet outputs, deterministic,
resumable, pytest for the shrinkage estimator against a synthetic ground truth.
```

---

# P4 — Evidence packets, fusion, two-level hypothesis state
**Duration: 2 weeks. This is where the audit's mathematical corrections land.**

# Objective & scope
Implement `MATH_SPEC.md`: packet construction, the noisy-channel likelihood, the fitted combination, the finding→explanation Jeffrey update, and the contribution trace. Reliability is stubbed at $r_{\text{pre}} = $ competence only; P5 completes it.

# Entry state
P3 calibration maps, competence posteriors, correlation matrix; P0 finding↔explanation table.

# Exit artifacts
- `herald/reliability/packet_builder.py` — deterministic adapters per `output_kind`; mandatory localisation where the tool supports it; `MissingEvidence` emission
- `herald/fusion/channel.py` — $\Lambda(s; r, \pi)$ with the Proposition-1 bound asserted at runtime
- `herald/fusion/combine.py` — per-label fitted weights $w_{jt}$, non-negative, L2
- `herald/fusion/explanations.py` — Jeffrey conditioning into the Level-X simplex
- `herald/fusion/trace.py` — per-packet contribution, additivity-asserted
- `profiling/combination_weights.parquet`
- `results/P4_fusion_report.md`

# Validity & alignment gate

| Check | Threshold |
|---|---|
| Additivity: $b_j + \sum_t w_{jt}\tilde\Delta_{jt} = \operatorname{logit} q_j$ exactly | assertion, every case |
| Proposition-1 bound never violated | assertion, every packet |
| Fusion beats best-single-tool macro-AUROC on dev | **> 0**, and report the delta with a paired bootstrap CI |
| Fusion beats *unweighted* log-odds summation on ECE | **> 0** — this is the M4 fix earning its keep |
| Redundancy factor $R_j$ reported per label | 100% coverage |
| Level-X posterior sums to 1; Level-F marginals never renormalised | assertion |
| `NOT_EXAMINED` never rendered as `EXCLUDED` anywhere | test passes |

# Task checklist
1. Packet adapters: LABEL, SCORE, MASK, MEASUREMENT, RETRIEVAL. **No thresholding in the builder** — thresholding is a decision and belongs in P7.
2. Localisation mandatory when the tool can produce it; `localisation_coverage` metric emitted.
3. `MissingEvidence` emitted by ingestion (absent modality) and by the planner (declined call), with `would_have_resolved` populated from the discrimination table.
4. Implement $\Lambda$ exactly as derived. Assert the Proposition-1 bound at runtime — a violation means a calibration map is broken.
5. Fit $w_{jt}$ on the profiling split: non-negative least squares in logit space, or L2-regularised logistic with `positive=True`. Persist with the fit manifest.
6. Jeffrey conditioning into Level X using the P0 table.
7. Contribution trace with an exact additivity assertion. If it does not sum, you have a bug — the blueprint is right that this one will bite you.
8. Mask-and-renormalise over *observed* evidence; record the mask in the trace.
9. Ablations for the report: fusion vs. best-single, vs. unweighted sum, vs. mean-probability.

# Prompting seam

```
CONTEXT: HERALD P4. P3 produced calibration maps, competence posteriors (with SD and a
measured flag), an error-correlation matrix, and evidence-bin tables. MATH_SPEC.md is
attached and is authoritative. Implement it exactly; do not substitute alternatives.

TASK: Generate the packet builder and the fusion layer.

1. herald/reliability/packet_builder.py
   Deterministic adapters per output_kind (LABEL, SCORE, MASK, MEASUREMENT, RETRIEVAL).
   NO thresholding anywhere in this module. Localisation mandatory when the tool supports it.
   MissingEvidence emission with would_have_resolved from the discrimination table.

2. herald/fusion/channel.py
   Implement Lambda(s; r, pi) = (r*L + (1-r)*(pi*L + 1 - pi)) / (r + (1-r)*(pi*L + 1 - pi))
   where L is derived from the calibrated posterior and the prevalence prior.
   Assert Proposition 1 at runtime:
       1/(1 + r/((1-r)*(1-pi))) <= Lambda <= 1 + r/((1-r)*pi)
   Raise a typed CalibrationError on violation.

3. herald/fusion/combine.py
   Per-label combination: logit q_j = b_j + sum_t w_jt * Delta_tilde_jt
   Fit w with non-negativity and L2 on the profiling split; b_j initialised at logit(prior).
   Persist weights + fit manifest. Provide a printable summary table.

4. herald/fusion/explanations.py
   Jeffrey / virtual-evidence conditioning:
     p(x_k | E) ∝ prior_k * prod_j [ q_j*P(F_j=1|x_k) + (1-q_j)*P(F_j=0|x_k) ]
   Level F is multi-label and MUST NOT be renormalised. Level X is a simplex.

5. herald/fusion/trace.py
   Per-packet contribution records; assert exact additivity; record the missingness mask
   and the renormalisation applied.

6. eval/axes/axis1_fusion.py
   Ablations: fused vs best-single-tool vs unweighted-sum vs mean-probability.
   Paired bootstrap CIs over cases. Emit redundancy factor R_j per label.

TESTS: additivity assertion; Proposition-1 bound under adversarial synthetic inputs;
Level-F never renormalised; NOT_EXAMINED never serialises as EXCLUDED; golden-file test
of the Jeffrey update against a hand-computed 3x3 example.
```

---

# P5 — Reliability estimator and conflict detector
**Duration: 2 weeks. The differentiator.**

# Objective & scope
Complete $r_{\text{pre}}$ from its three *packet-local* signals (competence, integrity, OOD — **not** agreement, per M3), and build conflict detection as a separate ledger-level concern.

# Entry state
P4 fusion; P3 competence posteriors with SD.

# Exit artifacts
- `herald/reliability/estimator.py` — $r_{\text{pre}} = g(\text{competence}, \text{competence\_sd}, \text{integrity}, \text{ood})$, logistic combiner fitted on packet-correctness labels, temperature-scaled
- `herald/reliability/integrity.py` — TTA spread for image models; RadFlag-style resampling for text outputs
- `herald/reliability/conflict.py` — three rule types; severity from $\min(r_i, r_j)$; ontology-aware parent/child checks
- `results/P5_reliability_report.md` — Axis-3 results

# Validity & alignment gate

| Check | Threshold | Notes |
|---|---|---|
| AUROC of $r_{\text{pre}}$ for packet correctness, **on clean-label subsets only** | **> raw softmax and > temperature-scaled softmax** | If it does not beat temperature scaling, the module has not earned its compute (§11-R9) — report that |
| ECE of $r_{\text{pre}}$ | ≤ 0.05 ID | |
| Clean-vs-corrupted separation AUROC (on the **non-held-out** corruption types) | ≥ 0.75 | Held-out types are not touched until P8 |
| Per-signal ablation completed | 3/3 | competence / integrity / OOD each removed in turn |
| Conflict detection precision/recall vs. injected contradictions | recall ≥ 0.9, false-conflict rate on clean cases ≤ 0.05 | |
| Confidence-Dichotomy check: `EVIDENCE`-class tools require larger downward correction than `VERIFICATION`-class | **reported, sign either way** | This is a testable prediction from ACL 2026; confirming *or* refuting it is a result |

# Task checklist
1. Define **packet correctness** explicitly per output kind, in `EVAL_PROTOCOL.md`: LABEL/SCORE → agreement with the clean-label ground truth at the calibrated operating point; MEASUREMENT → within a per-concept tolerance band; RETRIEVAL → **no ground truth, therefore excluded from the headline metric and reported separately.** State the coverage fraction honestly.
2. Integrity: seeded TTA (5 replicates) spread; precondition and quality-check status; RadFlag-style resampling for any text-producing tool.
3. OOD: Mahalanobis distance from P3, per tool source.
4. Combiner: logistic regression over `{competence_mean, competence_sd, integrity, ood, raw_confidence}` fitted on the calibration split, then temperature-scaled. Simple, inspectable, reportable — resist the urge to use a gradient-boosted model in v1.
5. `competence_sd` enters with the expected negative sign: unmeasured slices reduce reliability. Assert the fitted sign; if it comes out positive, something is wrong upstream.
6. Conflict rules: `DIRECT_CONTRADICTION` (same concept, opposite assertion), `MAGNITUDE_DISAGREEMENT` (numeric outside tolerance), `CROSS_MODAL` (image vs. indication text). Severity from $\min(r_i, r_j)$ — two *high*-reliability packets in contradiction is the serious case.
7. Ontology-aware conflict: parent asserted absent while child asserted present, using the UMLS relations from the PadChest mapping.
8. Axis-3 evaluation with all baselines: raw softmax, temperature-scaled softmax, ensemble variance alone, constant $r$.

# Prompting seam

```
CONTEXT: HERALD P5. P4 fusion is live with the noisy-channel likelihood. P3 supplied
competence posteriors with SD and a measured flag. CRITICAL DESIGN CONSTRAINT: agreement
between tools is NOT a packet-level reliability signal — it is handled once, in the P4
combination weights. Do not reintroduce it here.

TASK: Generate the reliability estimator and conflict detector.

1. herald/reliability/integrity.py
   Seeded TTA (5 replicates) score spread for image models.
   RadFlag-style resample-and-count support for any text-producing tool.
   Precondition and quality-check status folded in.

2. herald/reliability/estimator.py
   r_pre = sigma(logistic over [competence_mean, competence_sd, integrity, ood, raw_conf])
   Fit on the calibration split against packet-correctness labels. Temperature-scale after.
   ASSERT the fitted coefficient on competence_sd is negative; raise if not.
   r_pre is IMMUTABLE once written to a packet.

3. herald/reliability/conflict.py
   Rules: DIRECT_CONTRADICTION, MAGNITUDE_DISAGREEMENT (per-concept tolerance table),
   CROSS_MODAL. Severity = f(min(r_i, r_j)) so that two high-reliability contradictions
   escalate rather than resolve. Ontology-aware parent/child check via UMLS relations.

4. eval/axes/axis3_reliability.py
   Headline: AUROC of r_pre for packet correctness, computed ONLY on clean-label subsets
   (VinDr 5-rad test, PadChest-GR, CXR-LT radiologist dev). Report coverage fraction.
   Baselines: raw softmax, temperature-scaled softmax, ensemble variance, constant r.
   Per-signal ablation (competence / integrity / ood each removed).
   Per-tool-class breakdown testing the Confidence Dichotomy prediction.
   Clean-vs-corrupted separation using ONLY the non-held-out corruption types.

TESTS: r_pre immutability; competence_sd sign assertion; conflict precision/recall against
synthetically injected contradictions; false-conflict rate on clean co-occurring findings
(effusion + consolidation must NOT fire).
```

---

# P6 — Acquisition planner and tool router
**Duration: 2 weeks.**

# Objective & scope
Turn the fixed pipeline into an agent: EIG-driven acquisition planning over the Level-X posterior, and the three-stage routing funnel.

# Entry state
P5 reliability; P4 explanation posterior; P3 evidence-bin tables.

# Exit artifacts
- `herald/routing/planner.py` — EIG under the noisy channel (M5); `PlanDecision`
- `herald/routing/router.py` — hard precondition filter → score → LLM proposal with logged override
- `results/P6_routing_report.md` — Axis-2 results including the controller gap

# Validity & alignment gate

| Check | Threshold |
|---|---|
| Under-calling rate (eligible + informative tool never invoked) | ≤ baseline, and **reported** |
| Controller gap = oracle-routed − actual-routed | **reported with a CI**, no target |
| EIG ↔ realised information gain correlation | Spearman ρ **> 0**; if ≤ 0 the bin tables are wrong |
| Calls-to-disposition vs. call-everything | fewer calls at ≤ 0.02 accuracy loss |
| Decoy-tool test: a plausible-description, poor-competence tool is not selected | selection rate ≤ chance |
| Best-tool-removal test: system degrades, never fails | halt rate 0 |
| Distribution-shift routing test: previously-best tool is no longer best → router follows | **reported** — this is the test that separates real routing from a memorised preference |
| LLM-override win rate | **reported honestly, including if negative** |

# Task checklist
1. EIG over Level X using the P3 bin tables and the same channel as fusion. Do **not** add a separate reliability multiplier (M5).
2. Round cap of 4; cost-normalised stop threshold $\varepsilon$ calibrated on dev against exhaustive replay ("would one more call have changed the disposition?").
3. Three-stage funnel: hard precondition filter (an ineligible tool is never an option) → score by `competence_prior × EIG / normalised_cost` → LLM proposal with a **logged, separately evaluated** override.
4. Forced-coverage rule: any Level-X hypothesis with zero evidence and an eligible tool forces a call before termination.
5. Log which of the four termination conditions fired, every time (CF2).
6. Oracle router for the upper bound: exhaustive search over tool orders on the dev split, cached, so it is cheap.
7. Axis-2 stress tests: decoy tool, best-tool removal, distribution shift.

# Prompting seam

```
CONTEXT: HERALD P6. Fusion (P4) and reliability (P5) are live. P3 supplied evidence-bin
tables P(bin | F) per slice. CRITICAL: expected information gain must be computed under the
SAME noisy-channel likelihood used in fusion. Do NOT multiply an r-free EIG by reliability —
the discount is implied by the channel and multiplying it again is wrong.

TASK: Generate the acquisition planner and tool router.

1. herald/routing/planner.py
   EIG(t,j) = H(X) - E_S[H(X | S)] with S distributed under the mixture channel.
   Marginalise F over the Level-X posterior via the finding-explanation table.
   PlanDecision{action: ACQUIRE|STOP_SUFFICIENT|STOP_EXHAUSTED|REQUEST_HUMAN,
                target, eig, justification}
   Hard round cap 4. Cost-normalised stop threshold epsilon read from config.

2. herald/routing/router.py
   Stage 1: hard precondition filter from CapabilityCard. Ineligible tools are never options.
   Stage 2: score = competence_prior(descriptor) * EIG / normalised_cost.
            Tools with measured=False in this slice get maximum-uncertainty treatment.
   Stage 3: LLM proposal; overrides allowed but must carry a rationale and are logged with
            selection_mode=LLM_OVERRIDE for separate evaluation.
   Forced-coverage rule: a hypothesis with zero evidence and an eligible tool forces a call.
   Log which of the four termination conditions fired, always.

3. eval/oracle_router.py
   Exhaustive tool-order search on the dev split, cached, to give the upper bound.

4. eval/axes/axis2_routing.py
   Tool-selection accuracy vs oracle; controller gap with CI; under-calling rate;
   redundant-call rate; argument validity; calls-to-disposition; LLM-override win rate.
   Stress: decoy tool with a plausible description and poor measured competence;
   best-tool removal; distribution shift where the previously-best tool is no longer best.
   EIG vs realised-information-gain Spearman correlation.

TESTS: EIG -> 0 as r -> 0.5 (channel-capacity limit); ineligible tools never appear in the
candidate set; forced-coverage fires; termination condition always logged; oracle >= actual.
```

---

# P7 — Disposition (Learn-then-Test) and grounded generation
**Duration: 2 weeks.**

# Objective & scope
The abstention engine with a *correct* distribution-free risk guarantee, and the entailment-gated renderer.

# Entry state
P6 routing; P0 frozen loss ℓ and α grid; calibration split ≥ 2,000 raw.

# Exit artifacts
- `herald/disposition/ltt.py` — Learn-then-Test threshold selection with HB p-values and fixed-sequence correction
- `herald/disposition/rules.py` — the ordered cascade with correct accepted-set accounting
- `herald/generate/` — deterministic structured section, LLM prose, claim-level entailment gate with regeneration
- `results/P7_disposition_report.md` — risk–coverage curves, AURC

# Validity & alignment gate

| Check | Threshold |
|---|---|
| Empirical error rate among the accepted set vs. target α, **ID** | within the LTT guarantee at every α ∈ {0.05, 0.10, 0.20} |
| $n_{\text{cal}}^{\text{effective}}$ after rule-1–3 filtering | ≥ 1,200 — **if not, you cannot calibrate at α = 0.05; say so rather than reporting an invalid number** |
| Accepted set correctly = `ANSWER ∪ ANSWER_WITH_CAVEAT` | assertion |
| `REQUEST`/`ESCALATE` counted as abstentions in every coverage number | assertion |
| `REQUEST`-disabled variant reported alongside (CF1) | present |
| Over-abstention rate vs. the strongest baseline | **reported as a headline, not a footnote** |
| Unsupported-claim rate post-gate | ≈ 0 |
| Entailment-gate precision/recall vs. injected unsupported sentences | reported with κ against 100–200 human labels |
| Evidence coverage (% of clinical claims with a valid packet citation) | ≥ 0.95 |

# Task checklist
1. Implement LTT: grid over τ, Hoeffding–Bentkus p-value for the bounded loss, fixed-sequence correction, return the largest admissible acceptance region. Emit `ltt_grid_ref` and `n_cal_effective`.
2. **Filter the calibration set through rules 1–3 before calibrating** (M6c). Log the filtered fraction.
3. Ordered cascade exactly as §4.3-L8, with `in_accepted_set` set correctly and `fallback` resolved (CF3).
4. Structured output generated **deterministically** from the ledger by templating. Only prose is LLM-generated.
5. Generator prompt contains **the ledger only** — never the images, never the original free-text query.
6. Claim-level entailment gate: extract claims, check each against its cited packets with the local NLI model, **then regenerate once** with failed claims excluded (G5) rather than deleting sentences. Report drop rate and coherence.
7. Validate the gate against 100–200 human-labelled items; report κ. Never report a judge you have not validated.
8. Risk–coverage curves, AURC, selective risk at coverage ∈ {90%, 80%, 70%}, ID and shift separately.
9. Request specificity: replay each `REQUEST` case with the item supplied; measure how often the disposition changes. If it rarely does, the request was theatre — report that.

# Prompting seam

```
CONTEXT: HERALD P7. Routing (P6) is live. The loss function and alpha grid were frozen in
P0 (EVAL_PROTOCOL.md, attached).

CRITICAL STATISTICAL REQUIREMENTS — these are not optional:
 (a) Use Learn-then-Test, NOT Conformal Risk Control. Selective risk is not guaranteed
     monotone in the threshold, so CRC's assumption fails.
 (b) The calibration set must be filtered through disposition rules 1-3 before calibration,
     because those rules fire before the threshold. Log n_cal_effective.
 (c) The accepted set is ANSWER union ANSWER_WITH_CAVEAT. REQUEST and ESCALATE are
     abstentions for coverage accounting.
 (d) On any shift arm, exchangeability is violated by construction. Report empirical
     selective risk with Clopper-Pearson intervals and never the word "guarantee".

TASK: Generate the disposition engine and the grounded generator.

1. herald/disposition/ltt.py
   Grid over tau; Hoeffding-Bentkus p-value for the bounded loss; fixed-sequence correction;
   return the admissible acceptance region. Emit ltt_grid_ref, n_cal_effective,
   exchangeability_asserted.

2. herald/disposition/rules.py
   Ordered cascade, first match wins, all logged:
     1 integrity failure -> ESCALATE(TECHNICAL)
     2 unresolved high-severity conflict -> ESCALATE(CONFLICT)
     3 top hypotheses separable by a named obtainable observation -> REQUEST(item)
     4 LTT threshold not met -> ESCALATE(UNCERTAIN)
     5 threshold met, coverage incomplete -> ANSWER_WITH_CAVEAT
     6 otherwise -> ANSWER
   Set in_accepted_set correctly. Resolve the fallback field.
   Provide a REQUEST-disabled variant switch for the CF1 control.

3. herald/generate/structured.py   deterministic templating from the ledger
4. herald/generate/prose.py        LLM prose; prompt contains the LEDGER ONLY —
                                   never images, never the raw user query
5. herald/generate/gate.py         claim-level entailment against cited packets using a
                                   local NLI model; on failure REGENERATE ONCE with the
                                   failed claims excluded rather than deleting sentences.
                                   Report drop rate and a coherence score.

6. eval/axes/axis4_abstention.py
   Risk-coverage curve, AURC, selective risk at coverage 90/80/70, empirical error among
   accepted vs target alpha (ID and shift reported separately), over-abstention rate vs the
   strongest baseline, disposition-type distribution, request specificity via counterfactual
   replay, escalation precision.
   Baselines: no abstention, softmax threshold, entropy threshold, random at matched
   coverage, oracle abstention.

7. eval/judges/validate.py
   Harness for labelling 100-200 items and computing Cohen's kappa for every judged metric.
   judge_kappa must be non-null in the results table.

TESTS: LTT coverage on synthetic data with known risk; calibration filtering applied;
accepted-set accounting; gate precision/recall against injected unsupported sentences;
generator receives no image tensor (assert at the call boundary).
```

---

# P8 — Stress harness and full evaluation
**Duration: 3 weeks. This is where the project stops being a system and becomes a result.**

# Objective & scope
The three-setting protocol, all eight axes, complete results table with bootstrap CIs and judge κ.

# Entry state
P7 complete; frozen test set untouched; the 2 held-out corruption types still untouched.

# Exit artifacts
- `eval/stress/` — corruption injection, modality ablation, image degradation, paraphrase, tool removal, input ablation
- `results/RESULTS.parquet` — one row per experiment in the §8.3 schema
- `results/P8_report.md` — all curves and tables
- `NEGATIVE_RESULTS.md` — populated

# Validity & alignment gate

| Check | Threshold |
|---|---|
| **Internal propagation ablation**: HERALD vs. reliability-layer-disabled, same injections, paired bootstrap | delta reported with CI. *This replaces the invalid ≈0.62 comparison (F6)* |
| Detection rate and recovery rate reported **separately** | present |
| Held-out corruption types evaluated **once**, at the end | present; report the generalisation gap vs. the fitted types |
| Halt rate | 0 (asserted invariant, not a headline) |
| Degradation curve: accuracy **and** calibration on the same axes | present |
| Input-ablation control (image removed) run on the clean benchmark and **published** | present |
| Every judged metric carries `judge_kappa` | 100%, null-forbidden |
| Every number traceable to a `trace_manifest` hash | 100% |
| Reliability-layer overhead as % of wall-clock | reported |
| Cost per avoided error at each α | reported |

# Task checklist
1. Corruption injection at the **wrapper boundary** (which is why the P2 contract matters): label flip, confidence inflation, systematic bias, empty output, timeout, schema violation, stale result. Two types held out until now.
2. Modality ablation on PadChest/CheXpert (F4): lateral only, indication only, no text, no priors, image only, 1-of-N tool outputs dropped. Also *degrade* rather than remove (an under-inspired film).
3. Image corruption: Gaussian noise, JPEG artifacts, rotation, non-CXR image, lateral where frontal expected.
4. Paraphrase family: 3 paraphrases per query; measure frame agreement and disposition flip rate.
5. Input-ablation control on the clean multiple-choice benchmark. If accuracy holds with the image removed, the benchmark is shortcut-solvable — publish that.
6. G6 family: contradictory indication text; metric = prior-override rate.
7. S3 realistic arm: consecutive unfiltered cases at real prevalence from a held-out source. **Realistic ≠ real** — say so explicitly.
8. Judge validation: 100–200 human-labelled items, κ reported per judged metric.
9. Bootstrap CIs over cases for everything.
10. **Run the frozen test set once.**

# Prompting seam

```
CONTEXT: HERALD P8. The full system is built and the frozen test set is untouched. Two
corruption types were held out in P0 and have not been used for any fitting.

TASK: Generate the stress harness and evaluation driver.

1. eval/stress/corruption.py
   Injection at the ToolWrapper boundary: label_flip, confidence_inflation, systematic_bias,
   empty_output, timeout, schema_violation, stale_result. Single-tool and pairwise.
   Two types are flagged HELD_OUT and only enabled by an explicit run flag.

2. eval/stress/ablation.py
   Modality ablation (lateral / indication / priors / image-only / k-of-N tools dropped)
   AND degradation (under-inspired film) rather than only removal.
   MUST run on PadChest or CheXpert — VinDr-CXR is PA-only and has no laterals.

3. eval/stress/perturbation.py
   Image corruption (Gaussian, JPEG, rotation, non-CXR, wrong view);
   query paraphrase (3 per query, frame agreement + disposition flip rate);
   contradictory-indication family with a prior-override-rate metric.

4. eval/stress/input_ablation.py
   The Illusion-of-Readiness control: run the clean multiple-choice benchmark with the
   image removed. Publish the result whatever it shows.

5. eval/driver.py
   Executes the full grid across settings S1/S2/S3 and all eight axes.
   MUST use the deterministic cache: any cell that changes only fusion/reliability/conflict/
   disposition/generation logic replays from cache on CPU. Only cells that change tool inputs
   or the LLM may touch the GPU. Report the GPU/CPU cell split at the end.

6. eval/report/table.py
   Emit RESULTS.parquet with exactly:
     system | backbone | setting | axis | metric | value | ci_low | ci_high | n |
     judge_kappa | calib_ref | trace_manifest
   Bootstrap CIs over cases. judge_kappa null-forbidden for any judged metric.

7. eval/axes/axis6_propagation.py
   PRIMARY COMPARISON is internal: HERALD vs HERALD-with-reliability-disabled on identical
   injections, paired bootstrap. Report detection rate and recovery rate SEPARATELY.
   Cite the external AgentProp-Bench 0.62 figure as context only — it measures parameter-level
   injection in general-domain agents, which is a different failure channel.

TESTS: injection reaches the wrapper and nothing downstream bypasses it; held-out corruption
types are inaccessible without the explicit flag; the frozen test set is read-only and
access is logged; every emitted row validates against the results schema.
```

---

# P9 — Governance attachment and research extension
**Duration: 1 week + 4–6 weeks.**

# P9a — Governance (1 week)

**Objective.** Prove attachability. Do not build policy.

**Exit artifacts:** completeness audit against the §10.4 checklist with a score; one demonstration policy (one RBAC rule, one redaction rule, one review-routing rule); an OTel Collector redaction processor with a synthetic-identifier leak test; `docs/governance_handover.md` including the modality-aware requirements of §10.3.

**Gate:** deny honoured end-to-end on 5/5 hooks; completeness score ≥ target; synthetic PHI planted in inputs never appears in an indexed span attribute.

# P9b — Research extension (4–6 weeks)

**Choose on evidence, not on aspiration.** The decision memo is written *from* P8's numbers:

| If P8 showed… | Pursue |
|---|---|
| Between-slice competence variance is large **and** the controller gap is large | **R1** — calibrated tool-trust priors. Primary claim. The P3 profiling artifact is already its training set. |
| Competence variance is small (P3 gate flagged it) | **R3** — disclose-and-degrade. Survives the null routing result intact. |
| Reliability layer failed to beat temperature scaling (P5 gate) | **R9 as a result** — a well-measured negative on a widely-assumed-good idea, plus **R5** as the shippable minor paper |
| Conflict-triggered abstention dominated confidence-triggered on the risk–coverage curve | **R2** — highest novelty, hardest guarantee |

**Mandatory framing correction for R1 (M5, C1).** The novelty is *not* "reliability-discounted EIG" — that is the textbook Bayesian answer under the corrected channel. State the contribution as:

1. Per-case, pre-call tool-trust estimation via OOD-attenuated hierarchically-shrunk competence posteriors;
2. One estimator serving both selection and evidence weighting, with the transfer between the two uses measured;
3. A bounded-evidence guarantee (Proposition 1) that the closest prior system does not have.

And **explicitly position against `arXiv:2507.10571`** (trust-calibrated orchestration with ECE-based agent trust scoring and re-evaluation loops for visual classification), which is closer prior art than anything the blueprint cites and is currently absent from its related work. A reviewer will find it.

---

# 4. De-scoped 12-week critical path

If the semester is shorter than 22 weeks — and it will be — this is what to cut and what is untouchable.

| Week | Phase | Cut from full plan |
|---|---|---|
| 1 | P0 | Nothing. **P0 is never cut.** |
| 2–3 | P1 | Nothing. **P1 is never cut** — five principles are cheap now and impossible later. |
| 4–5 | P2 | Drop the guideline retriever and the similar-case retriever. Keep classifiers, segmentation, geometry, cache. |
| 6–7 | P3 | Slice on **source × view only** (6 cells, not 18). Keeps competence measurable. |
| 8–9 | P4 + P5 | Reliability from competence + integrity only; drop the OOD term to a reported-but-unused field. |
| 10 | P7 (skip P6) | **Fixed tool order, no router.** LTT disposition + entailment gate. |
| 11–12 | P8 (reduced) | Axes 3, 4, 5, 6 only. Settings S1 + S2 only. Two corruption types + modality ablation. |

**What this still yields:** a working evidence-ledger agent with a calibrated reliability estimator, a distribution-free selective-risk guarantee, a graceful-degradation curve, and a propagation ablation — i.e. R3 and R5 are both fully supported, and R1 has its training data sitting in `profiling/` for a follow-on.

**What it sacrifices:** the router, and therefore R1 as a *primary* claim.

**What must never be cut:** P0 (the maths spec and the three feasibility probes), P1 (contracts, audit spine, hooks), and the deterministic cache in P2. Those three are the ones that are cheap now and structurally impossible to retrofit.

---

# 5. Standing invariants for every future coding session

Paste this block at the top of any implementation prompt:

```
HERALD INVARIANTS — these override any conflicting instruction:

1.  Level F (findings) is multi-label and is NEVER renormalised.
    Level X (explanations) is exclusive and IS a simplex.
2.  Evidence enters fusion ONLY through the noisy-channel Lambda(s; r, pi).
    There is no hand-tuned weight function f(reliability).
3.  Proposition 1's bound is asserted at runtime on every packet.
4.  Agreement between tools is NOT a packet field and NOT a reliability signal.
    Corroboration is handled once, in the fitted combination weights.
5.  r_pre is immutable once written. Packets are append-only; corrections use supersedes.
6.  Wrappers transport; they never interpret, summarise, or threshold.
    Thresholding is a decision and lives in the disposition layer.
7.  Any competence metric with measured=False is None, and the router treats that slice
    as maximum uncertainty.
8.  Selective risk uses Learn-then-Test, calibrated on the rule-1-to-3-filtered subset.
    The accepted set is ANSWER union ANSWER_WITH_CAVEAT.
9.  The generator receives the ledger only — never images, never the raw user query.
10. Every tool and LLM call is content-addressed and cached. Replay reads from cache;
    regeneration stability is measured separately.
11. Every irreversible action is preceded by a governance hook, asserted in CI.
12. No judged metric is reported without a validated judge_kappa.
13. Halt rate 0 is an asserted invariant, not a result.
14. NOT_EXAMINED must never serialise, render, or be summarised as EXCLUDED.
```

---

# PART III — REFERENCE AND LITERATURE MANIFEST

# Verification tiers

The blueprint's dossier leans heavily on 2026 preprints. Five of the most load-bearing were resolved live during this audit; the rest were not. Tiers below are **epistemic status**, not quality judgements.

| Tier | Meaning | Action before citing in a write-up |
|---|---|---|
| **A** | Verified live in this session, or long-established pre-2025 work | Cite freely |
| **B** | Established work I can corroborate from training, published before mid-2025 | Spot-check the numeric claims you quote |
| **C** | 2026 preprint, **not verified** | **Resolve the arXiv ID and read the abstract before any claim rests on it** |

> **The register below is the P0 citation-audit worklist.** Nine Tier-C entries carry numbers that appear in the blueprint's argument structure. If any fails to resolve, the corresponding claim must be removed, not softened.

---

# 1. Required papers and research literature

# 1.1 The four papers the architecture actually depends on — read these first, in this order

| # | Reference | Tier | Why it is on the critical path |
|---|---|---|---|
| 1 | **Angelopoulos, Bates, Candès, Jordan, Lei — "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control"**, *Ann. Appl. Stat.* 2025; 19(2):1641–1662 (arXiv:2110.01052) | A/B | **The** instrument for §5.13, replacing the blueprint's CRC. You need §3 (HB p-values) and the multiple-testing correction. Non-negotiable reading. |
| 2 | **Angelopoulos, Bates, Fisch, Lei, Schuster — "Conformal Risk Control"**, ICLR 2024 (arXiv:2208.02814) | A/B | Read to understand *why* it does not apply here (monotonicity), so you can defend the choice. |
| 3 | **Xuan, Zeng, Qi, Xiao, Wang, Yokoya — "The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents"**, ACL 2026 Long; arXiv:2601.07264 | **A — verified** | The `tool_class: EVIDENCE\|VERIFICATION` field is load-bearing and comes from here. Their finding is a *testable prediction* for your P5 gate. |
| 4 | **He, Choi, Yoon — "Confidence-Aware Tool Orchestration for Robust Video Understanding" (Robust-TO)**, arXiv:2606.26904 | **A — verified** | The closest architectural precedent. Read it to position HERALD's probabilistic fusion against their three-tier heuristic synthesis — that contrast *is* your differentiator. |

# 1.2 Verified in this session

| Reference | ID | What you may safely claim |
|---|---|---|
| Xuan et al., *The Confidence Dichotomy*, ACL 2026 | 2601.07264 | Evidence tools induce overconfidence; verification tools with deterministic feedback mitigate it; RL joint accuracy+calibration objective |
| He, Choi, Yoon, *Robust-TO* | 2606.26904 | Blind Trust Problem; unified evidence interface with calibrated reliability; three-tier synthesis; 15–30 pp drops under corruption; 56.4% clean / 54.3% corrupted |
| Gurram, *AgentProp-Bench* | 2604.16706 | κ=0.049 substring vs κ=0.432 three-LLM ensemble; propagation ≈0.62 (0.46–0.73); rejection/recovery independent (ρ=0.126, p=0.747). **Note: parameter-level injection, general domains — not tool-output corruption in imaging.** |
| Ashraf, Jeong, Thoker, Ghanem, *MedCTA* | 2606.11702 | 107 tasks, 5 tools, 18 models; best agentic outcome accuracy 31.54%; zero strict trajectory success; large controller gap under gold routing; failures in under-calling and weak local grounding. **Tools = perception/OCR/retrieval/calculation, no segmentation.** |
| Dong et al., *CXR-LT 2026 Challenge Overview* | 2602.22092 (also appears as 2604.15555) | 145k+ from PadChest + NIH; Task 1 = 30 known classes, Task 2 = 6 unseen; winning mAP 0.5854 / 0.4315. **Task 2: dev 200, internal test 305, external NIH test 79.** |

# 1.3 Benchmark validity and stress-test methodology

| Reference | Tier | Use |
|---|---|---|
| Gu et al., *The Illusion of Readiness in Health AI*, arXiv:2509.18234 | B | The stress-test suite (input ablation, distractor perturbation, visual substitution) is directly reusable. The input-ablation control in P8 comes from here. |
| Hager et al., *Nature Medicine* 2024;30(9):2613–2622 | B | LLM failure on sequential clinical decision-making; sensitivity to information order and quantity |
| Jin et al., *Hidden flaws behind expert-level accuracy of multimodal GPT-4V*, npj Digit. Med. 7:190 (2024) | B | Correct answers with wrong rationales — the motivation for the entailment gate |
| Bedi et al., *MedHELM*, arXiv:2505.23802 / *Nature Medicine* 2026 | B/C | LLM-jury ICC 0.47 vs clinician–clinician 0.43. Verify the *Nature Medicine* version. Apache-2.0, Python 3.10. |
| Bedi et al., *JAMA* 2025 — systematic review of LLM health-application evaluation | B | Framing for the limitations section |
| *How to benchmark medical AI agents*, PLOS Medicine 2026, pmed.1005170 | C | Verify. Used for the clinician-supporting scope claim (§2.3). |

# 1.4 Selective prediction, abstention, calibration

| Reference | Tier | Use |
|---|---|---|
| Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World* (2005) | A | Foundational; split conformal |
| Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, arXiv:2107.07511 | A | The practical primer; read before LTT |
| Angelopoulos et al., *Learn Then Test*, AoAS 2025 | A/B | **Primary instrument** |
| Angelopoulos et al., *Conformal Risk Control*, ICLR 2024 | A/B | Read for the contrast |
| Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017 | A | Risk–coverage curves, AURC — the metrics of Axis 4 |
| Abbasi-Yadkori et al., *Mitigating LLM Hallucinations via Conformal Abstention*, NeurIPS 2024 WS | B | Prior art for R2 |
| Mohri & Hashimoto, *Conformal Factuality*, ICML 2024 | B | Back-off chain — directly relevant to `ANSWER_WITH_CAVEAT` |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | A | Temperature scaling — your P5 baseline |
| Kull et al., *Beyond temperature scaling: Beta calibration*, NeurIPS 2019 | A | Alternative for the per-tool maps in P3 |
| *CARE: A Conformal Safety Layer for Medical Summarization*, arXiv:2606.08969 | C | Verify. Direct precedent; must be positioned against in R2. |
| *Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty*, arXiv:2601.12471 | C | Verify. R2 prior art. |
| *Entropy Alone is Insufficient for Safe Selective Prediction*, arXiv:2603.21172 | C | Verify. The blueprint's §2.4(a) argument rests on it. |
| *Conformal selective prediction with cost-aware deferral … clinical triage under distribution shift*, Sci. Rep. 2026, s41598-026-40637-w | C | Verify. Best methodological template for the shift arm. |
| *Cascaded Language Models for Cost-Effective Human-AI Decision-Making*, arXiv:2506.11887 | C | Verify. R2 prior art. |

# 1.5 Tool reliability, routing, error cascades

| Reference | Tier | Use |
|---|---|---|
| Xuan et al., *Confidence Dichotomy*, 2601.07264 | **A** | See §1.1 |
| He et al., *Robust-TO*, 2606.26904 | **A** | See §1.1 |
| Gurram, *AgentProp-Bench*, 2604.16706 | **A** | Injection taxonomy for P8; the judge-validation requirement |
| Ashraf et al., *MedCTA*, 2606.11702 | **A** | Routing motivation only — not an evaluation target |
| Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST), arXiv:2503.13657, NeurIPS 2025 | B | 14 failure modes; the argument against a debate architecture |
| **Anonymous/Agentic AI with Orchestrator-Agent Trust**, arXiv:2507.10571 | **A — discovered in this audit** | **Absent from the blueprint. Trust-calibrated orchestration with ECE/OCR/CCC-based agent trust scoring and re-evaluation loops. Closer prior art to R1 than anything currently cited. Address explicitly.** |
| *When to Trust Tools? Adaptive Tool Trust Calibration*, arXiv:2604.08281 | C | Verify. Cited for the "ATTC-style pre-call trust signals" in §5.5-Advanced. |
| *Silent Failure in LLM Agent Systems*, arXiv:2606.08162 | C | Verify. Source of "Behaviour Routing Deficiency." |
| Schick et al., *Toolformer*, NeurIPS 2023 | A | Background |
| Yao et al., *ReAct*, ICLR 2023 | A | The baseline architecture you are arguing against |

# 1.6 Fusion, missing modality, graceful degradation

| Reference | Tier | Use |
|---|---|---|
| Pearl, *Probabilistic Reasoning in Intelligent Systems* (1988), §2.3 virtual evidence | A | **The Jeffrey-conditioning update in M1 comes from here. Read this before implementing P4.** |
| Jacobs et al., *Adaptive Mixtures of Local Experts* (1991) | A | Background for gated fusion |
| Ma et al., *SMIL: Multimodal Learning with Severely Missing Modality*, AAAI 2021 | B | Missing-modality baselines |
| *Transparent Early ICU Mortality Prediction … Per-Case Modality Attribution*, arXiv:2511.15847 | B/C | The mask-and-renormalise pattern; AUPRC −0.09/−0.11 with calibration retained. **Verify the numbers — they are quoted three times in the blueprint.** |
| *Meta-Modal Agent: Sequential Evidence Routing for Missing-Modality…*, arXiv:2605.25007 | C | Verify. The "reconstruction as liability" argument for R3. |
| *Handling and Interpreting Missing Modalities in Patient Clinical Trajectories*, arXiv:2604.18753 | C | Verify. |

# 1.7 Differential diagnosis and information-gain acquisition

| Reference | Tier | Use |
|---|---|---|
| Lindley, *On a Measure of the Information Provided by an Experiment*, Ann. Math. Stat. 1956 | A | **The EIG objective. Foundational for P6; read before implementing.** |
| Chaloner & Verdinelli, *Bayesian Experimental Design: A Review*, Statist. Sci. 1995 | A | The standard reference for sequential experiment design |
| MacKay, *Information Theory, Inference, and Learning Algorithms*, ch. 28 | A | Practical EIG computation |
| McDuff, Tu et al., *Towards Accurate Differential Diagnosis with LLMs* (AMIE), arXiv:2312.00164; *Nature* 2025 | B | Top-10 59.1% vs 33.6%; the "goal-directed acquisition of information under uncertainty" framing |
| Kanjee et al., *JAMA* 2023 — GPT-4 on NEJM CPCs | B | 39% exact, 64% in-list |
| *Toward Accurate and Actionable Differential Diagnosis with Lean LLM Orchestration*, medRxiv 2025.10.28.25338335 | C | Verify. **`Capture@K` originates here — and per M8 it does not transfer to CXR without the action-mapping table.** |
| *Impact of lab results on LLM-generated differential diagnoses*, npj Digit. Med. 2025, s41746-025-01556-8 | B/C | Up to +30% from labs — the empirical case for evidence acquisition |
| *HeartAgent*, arXiv:2603.10764 | C | Verify. The "scaffolding > backbone" claim supporting P7-backbone-agnosticism. |

# 1.8 Clinical agents, CXR systems, report metrics

| Reference | Tier | Use |
|---|---|---|
| Fallahpour et al., *MedRAX* + ChestAgentBench, arXiv:2502.02673, ICML 2025 | B | Citation-only baseline (per F6). Read the tool set for design ideas. |
| Lee, Yoon & Choi, *CXReasonAgent*, arXiv:2602.23276 | C | Verify. The CheXStruct-based deterministic-evidence design is the model for your geometric suite and quality checks. |
| *RadFabric*, arXiv:2506.14142 | C | Verify. MCP as a CXR tool bus (v2 relevance only). |
| Bannur et al., *MAIRA-2 / RadFact*, arXiv:2406.04449 | B | **The closest instrument to evidence fidelity. Read before P7's gate.** |
| Ostmeier et al., *GREEN*, arXiv:2405.03595 | B | LLM-judge report metric; note the documented length bias |
| Smit et al., *CheXbert* (2020); Jain et al., *RadGraph* (2021); Yu et al., *RadCliQ* (2023) | A/B | Report metric lineage |
| *RadFlag: Black-Box Hallucination Detection for Medical VLMs*, arXiv:2411.00299 | B | **Training-free, cheap — the integrity signal for text-producing tools in P5** |
| Jiang et al., *MedAgentBench*, arXiv:2501.14654; NEJM AI 2025 | B | v2 EHR scope only |
| Sellergren et al., *MedGemma Technical Report*, arXiv:2507.05201 | B | Backbone; MedSigLIP-400M |

# 1.9 Governance and audit

| Reference | Tier | Use |
|---|---|---|
| EU AI Act, Reg. (EU) 2024/1689 — Arts. 9–12, 14, 19, 26, 50; Annexes III, IV | A | Art. 12 logging, Art. 14 oversight — the P1/P9a requirements |
| Digital Omnibus on AI — timeline deferrals | C | **Verify the current state; this moved repeatedly through 2026 and any date you quote will be checked.** |
| OpenTelemetry GenAI Semantic Conventions (semconv v1.41) | B | `invoke_agent` / `chat` / `execute_tool`; content in span **events**. Pin the version — Development stability. |
| ISO/IEC 42001; IEC 62304 | A | Context for the limitations section |
| *Decision Trace Schema for Governance Evidence*, arXiv:2604.09296; *Runtime Governance for AI Agents*, arXiv:2603.16586 | C | Verify. R7 material for the teammate. |

---

# 2. Datasets and toolkits

# 2.1 Datasets — with the corrected experiment assignment (per §3.4)

| Dataset | Access | Size | **Valid for** | **Not valid for** |
|---|---|---|---|---|
| **VinDr-CXR** (Nguyen et al., *Sci Data* 2022) | PhysioNet, open-ish registration | 18,000 (15k train / 3k test); radiologist boxes; 3 annotators train, 5 consensus test | Localisation and grounding metrics; clean-label reliability headline (test split) | **Lateral ablation, `REQUEST(lateral)`, missing-modality — it is PA-only** |
| **PadChest** (Bustos et al.) | Open download | 160,868 imgs / 67,625 patients; 174 findings; UMLS-mapped; ~27% physician-validated; Spanish reports; **multiple projections incl. lateral** | Ontology bootstrap; lateral ablation; `REQUEST` replay; domain-shift target | Headline reliability (report-derived labels are noisy — G2) |
| **PadChest-GR** | Open (via PadChest/CXR-LT) | Radiologist-annotated grounded subset | **Grounded report entailment (R5); clean-label reliability; CXR-LT dev/test** | — |
| **NIH ChestX-ray14** | Open (NIH Box) | 112,120 imgs / 30,805 patients; 14 text-mined labels | Scale; profiling source; CXR-LT external test | Any clean-label metric |
| **CheXpert / CheXpert Plus** | Stanford AIMI registration | 224,316 CXRs / 65,240 patients; **frontal + lateral** | Lateral ablation; profiling source; reference reports | — |
| **CXR-LT 2026** | Challenge release | Train 142,928; T1 dev 1,620 / test 1,400; **T2 dev 200 / internal test 305 / external NIH 79** | **Open-set arm only (S2b)**, reported with exact CIs | Domain shift (same institution); subgroup slicing (n too small) |
| **CheXmask** | Open | Anatomical masks across ChestX-ray8/CheXpert/MIMIC/PadChest/VinDr | Segmentation ground truth for the geometric suite | — |
| **RSNA Pneumonia**, **SIIM-ACR Pneumothorax** | Kaggle | — | Additional profiling sources | — |
| **MIMIC-CXR / MIMIC-IV** | **PhysioNet credentialed + CITI + DUA** | — | Upgrade path only | **Never on the critical path (R10). DUA prohibits third-party LLM APIs — this is why P8 local-first is a legal constraint, not a preference.** |
| **Synthea**, **SMART Health IT sandbox** | Fully open, no DUA | Synthetic FHIR | v2 EHR arm only | v1 |
| **SLAKE, VQA-RAD** | Open | — | Sanity checks | Any reliability/missingness claim (§6.2) |

# 2.2 Models and weights

| Component | Choice | Notes |
|---|---|---|
| Reasoning backbone | **MedGemma-4B multimodal** (default), **Qwen2.5-VL-7B-Instruct** (swap arm) | bf16 on 24 GB with `gpu_memory_utilization ≈ 0.55`. **MedGemma-27B-text does not co-reside on 24 GB — see F1.** |
| Encoder / OOD / retrieval | **MedSigLIP-400M** | View detection, Mahalanobis OOD score, k-NN retrieval |
| Evidence tools | **TorchXRayVision** `densenet121-res224-{nih,chex,mimic_ch,mimic_nb,pc,rsna,all}` | **Verify the supported-label intersection in P0 (F2). `mimic_ch` + `mimic_nb` = one source.** |
| Segmentation | TorchXRayVision PSPNet / CheXmask-derived | Feeds the geometric suite |
| Verification tools | Custom NumPy/OpenCV geometric suite | CTR, rotation, inspiration, costophrenic angles, tracheal deviation. Deterministic → `tool_class=VERIFICATION` |
| Entailment | **DeBERTa-v3-large-MNLI** class local NLI | Validate against 100–200 human labels before reporting; clinical negation is its known weak point |
| Judge | 3-model local LLM jury | κ mandatory (AgentProp-Bench: substring judging is chance-level) |

# 2.3 Software stack

| Concern | Choice | Audit note |
|---|---|---|
| Serving | vLLM (OpenAI-compatible) | **Not bitwise deterministic under continuous batching → cache LLM responses (CF5)** |
| Orchestration | LangGraph | Typed state, checkpointing, native HITL interrupts |
| Schemas | Pydantic v2 | The enforcement mechanism for P6 |
| Imaging | `torchxrayvision`, MONAI, OpenCV | All pip-installable, no training |
| Statistics | `scipy`, `statsmodels`, `scikit-learn` (isotonic, logistic with `positive=True`) | Empirical-Bayes shrinkage is closed-form; no MCMC needed in v1 |
| Storage | Parquet + DuckDB + JSONL bundles + content-addressed cache | DuckDB reads trace JSONL directly — evaluation queries become SQL |
| Observability | OpenTelemetry SDK + self-hosted Langfuse or Grafana Tempo | Pin semconv version |
| Experiment tracking | MLflow (self-hosted) | Data-residency discipline |
| Eval harness | Custom `herald-eval`; `crfm-helm` for MedHELM comparability | Python 3.10 for HELM |
| Config | Hydra | Backbone/tool/threshold sweeps without code edits |
| CI | GitHub Actions | Four tests keep the architecture honest: schema validation, tool-removal, replay-from-cache, hook-coverage assertion |

# 2.4 Reproducibility settings that are easy to miss

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # required for torch determinism on CUDA
```
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# vLLM: gpu_memory_utilization=0.55  (default 0.90 starves the tool layer — F1)
```
Splits must be **patient-level disjoint**, never image-level — PadChest and NIH both contain multiple studies per patient, and image-level splitting will inflate every number in the project.

---

# Closing assessment

The blueprint's central bet — that the interesting engineering in a clinical agent is epistemic rather than inferential — is correct, well-evidenced, and better-motivated than most of the systems it surveys. Its failure-mode-to-module correspondence is genuinely the right way to structure a research architecture.

But it currently specifies a fusion layer that double-counts corroboration among tools that share training data, weights evidence by an undefined function, renormalises a multi-label space as if exclusive, and applies a conformal method whose assumption its own risk function violates. Implemented as written, HERALD would be **most overconfident precisely when its specialists agree because they were trained on the same data** — which is the failure it exists to prevent, occurring inside the module built to prevent it.

Every one of those is a two-day fix if made before code exists, and a rewrite if made after. That is the entire argument for P0 being a specification phase with a written derivation as its exit artifact.

Two further things worth internalising:

**The measurable claim is smaller than the blueprint implies, and that is fine.** The multi-source premise likely supports 6–10 labels, not 18. The OOD arm is 305 + 79 images, not 500. The reliability headline is computable only on clean-label subsets. State each of these in the write-up before a reviewer does; a smaller claim that survives scrutiny is worth more than a larger one that does not.

**The most valuable outcome may be a negative one.** If the reliability estimator does not beat temperature scaling (P5 gate), or if between-slice competence variance turns out to be small (P3 gate), that is a well-measured negative result on a widely-assumed-good idea — and the blueprint is right to commit to publishing it. Start `NEGATIVE_RESULTS.md` on day one.
