# Implementation Blueprint — HIPAA-Compliant Architecture for Agentic Clinical AI Systems

**Target:** Faithful, modular reproduction of Tripathi, Neupane, Mittal, Rahimi, Gupta — *A HIPAA-Compliant Architecture for Agentic Clinical AI Systems* (CAIS '26).
**Document type:** Sequential engineering workflow. No production code — architecture, contracts, data flow, inputs/outputs, and core logic per phase.
**Author of blueprint:** systems-architecture pass over the paper text. Every mechanism below is traceable to a specific section/algorithm/table of the source paper; deviations and under-specified points are marked.

---

# 0. Strategic & Epistemic Notes (read before building)

Labeled per your Fact / Assessment / Speculation convention.

- **[Fact]** This system is a *text-only* governance middleware. Every "input" that reaches the model is a string. Figure 1 depicts `X-Ray` as a data source and Figure 12 lists an `imaging` visibility field, but there is **no image tensor, no vision model, and no pixel-level gate anywhere in the pipeline**. Imaging is a field name that flows through the same regex+BERT text sanitizer as everything else.
- **[Assessment]** That is precisely the certification hole your Phase-0 deliverable is meant to expose: a raw image (or an image path / DICOM blob passed as a field) will pass the "Sanitize → Infer → PostRedact → Audit" chain and be logged as HIPAA-compliant, because the text sanitizer has nothing to say about pixels. **Build the input boundary (Phase 9 orchestrator, step before line 1) as an explicit, instrumentable seam** so you can later inject non-text inputs and record that the pipeline emits an ALLOW/compliant certificate for them. This works as a baseline for *either* fork (imaging-admission or text-domain tool-use) — the tool-call boundary is the same seam.
- **[Fact]** The paper admits (Appendix O) its "immutable audit" is **plain JSON arrays** in the experiment; XACML/cryptographic hashing/Merkle anchoring is described as future work. So "immutable audit trail" is an *unearned* claim in the artifact. Reproduce the JSON logger, implement the XACML *serializer* (Listing 2 gives the exact schema), but do **not** claim immutability unless you actually add hash-chaining.
- **[Assessment]** The headline "weighted F1 = 1.00 / exploits limited to 1.1%" is partly an artifact of the severity weighting. In §5.3 all leaked items happened to be low-severity (weight 1) clinical vocabulary, so the weighted score perfects *by construction*. This is not a bug to fix — it is a measurement property to be aware of and to report honestly if you use these numbers as a baseline.
- **[Fact]** `gpt-oss-20b` (open-weight, used for inference) and `MedGemma-4B-IT` (used for data reconstitution) are real, downloadable models. The paper runs both locally via GGUF quantization.
- **[Speculation]** The exact deduplication rule that reduces 475,200 → 107,800 rows is **not fully specified** in the paper. You will have to define your own dedup key (Phase 10) and document it; do not claim to reproduce the exact count without it.

---

# Architecture at a glance

Six cooperating agents, orchestrated by one end-to-end controller (Algorithm 1):

```
[Client: role s, object o, action a, env E, raw note T_raw, query Q]
        
        
1. PolicyDecisionAgent      (Alg 2)  → ABAC decision D = {decision, obligation, redact_fields}
          (DENY → stop, audit, return)
        
2. SanitizationAgent        (Alg 3)  → T_san  (pre-inference regex+BERT redaction)
        
        
3. MiddlewareAgent          (Alg 4)  → P_final (system + dev-guidance + record + query)
        
        
4. InferenceAgent           (§4.4)   → T_output  (gpt-oss-20b, GGUF, temp=0.0)
        
        
5. PostInferenceRedaction   (Alg 5)  → T_final  (reuses Alg 3 on the OUTPUT)
        
        
6. AuditAgent / Downstream  (Alg 6)  → deliver T_formatted + write audit tuple A
```

---

# Phase 0 — Repository scaffold & environment

**Goal:** Deterministic, reproducible, on-prem-style environment. Nothing leaves the box.

# 0.1 Folder / module structure

```
hipaa-agentic-framework/
 README.md
 pyproject.toml                # or requirements.txt; pin every version
 config/
    settings.yaml             # global run config (paths, seeds, model ids, thresholds)
    policies.json             # ABAC policy set (Phase 1 schema)
    severity_weights.yaml     # PHI category → weight {5,3,2,1}
    regex_patterns.yaml       # P_r pattern bank (Table 3-derived)
    redaction_tokens.yaml     # PHI category → redaction token
 models/                       # local GGUF weights (gitignored)
    gpt-oss-20b/              # inference model
    medgemma-4b-it/           # data-reconstitution model
 src/
    agents/
       policy_decision.py        # Agent 1  (Alg 2)
       sanitization.py           # Agent 2  (Alg 3)  — shared by pre + post
       middleware.py             # Agent 3  (Alg 4)
       inference.py              # Agent 4  (§4.4)
       post_redaction.py         # Agent 5  (Alg 5 → calls sanitization)
       audit.py                  # Agent 6  (Alg 6)
    core/
       schemas.py                # dataclasses/pydantic: Subject, Object, Env,
                                   # AuthDecision, AuditTuple, Identity, PHIManifest
       obligations.py            # Obligation enum + obligation→scope map
       phi_detectors/
          regex_detector.py     # PHI_regex
          bert_detector.py      # PHI_bert (NER tagger, threshold τ)
       orchestrator.py           # Algorithm 1 end-to-end controller
       xacml.py                  # XACML serializer (Listing 2)
       input_gate.py             # <-- INSTRUMENTATION SEAM for non-text inputs
    data_generation/
       identity_factory.py       # GenerateIdentity(σ)  (Table 3 rules)
       reconstitution.py         # 3-phase line-by-line filler (§5.1)
       manifest.py               # S_i (v, φ) PHI manifest builder
    evaluation/
        metrics.py                # F1, weighted F1, WLS, Resid, ROUGE-L, BERTScore
        ablation.py               # 3 sanitize configs × 3 stages runner
        adversarial.py            # 10 attacks + STRIDE mapping
        run_matrix.py             # 200×11×3×3 combinatorics + dedup
        report.py                 # tables/figures reproduction
 data/
    raw/                          # MIMIC-IV discharge notes (___-placeholdered)
    reconstituted/                # d_filled + S_i manifests
    results/                      # logs, audit ledgers, metric outputs
 tests/
     test_policy_decision.py
     test_sanitization.py
     test_orchestrator.py
     test_metrics.py
```

# 0.2 Environment

- **[Fact — Appendix N]** Reference hardware: NVIDIA RTX 5090 (24 GB), Intel i9, 64 GB RAM. Any ≥24 GB GPU that can hold a quantized 20B model works.
- **GGUF runtime:** `llama.cpp` (server mode) or `ollama`. Load both models locally; **no cloud API calls** — this is the on-prem privacy premise.
- **Python deps (indicative):** `transformers`, `torch`, a NER backbone (see Phase 4), `rouge-score`, `bert-score`, `pydantic`, `pyyaml`, `lxml` (XACML), `pandas`, `numpy`.
- **Determinism:** global seed; model temp 0.0; fixed stop tokens; log versions of every model + library into each audit record.

**Inputs:** none. **Outputs:** reproducible env, empty typed schemas, config skeleton.

---

# Phase 1 — Data schemas & contracts

**Goal:** Freeze the data contracts first; every agent reads/writes these types.

# 1.1 ABAC policy JSON (Appendix B — reproduce verbatim structure)

```json
{
  "positions": {
    "resident_doctor": {
      "role": "resident_doctor",
      "access_window": {"start": "08:00", "end": "16:00"},
      "visibility": {"patient_name": false, "clinical_diagnoses": true, "medications": true},
      "obligation": "hybrid"
    },
    "clinical_coder": {
      "role": "clinical_coder",
      "access_window": {"start": "10:00", "end": "18:00"},
      "visibility": {"patient_name": false, "admission_date": true, "clinical_diagnoses": true},
      "obligation": "redact_demographics"
    },
    "auditor": {
      "role": "auditor",
      "access_window": {"start": "12:00", "end": "20:00"},
      "visibility": {"patient_name": false, "clinical_diagnoses": false, "medications": false},
      "obligation": "redact_all"
    }
  }
}
```

Roles NOT in JSON but handled in code (Appendix A):
- `doctor` / `patient` / `nurse` → full access, **any time**, obligation `none`.
- Any role ∉ {doctor, patient, nurse, resident_doctor, clinical_coder, auditor} → **DENY** (deny-by-default).

# 1.2 Core dataclasses (`schemas.py`)

- `Subject`: `{role, department, access_level, time_window}` (A_s)
- `Object`: `{sensitivity_level, record_type, owner, data_categories}` (A_o)
- `Action`: enum `{read, query}`
- `Env`: `{current_time, ...}` (E)
- `AuthDecision`: `{decision ∈ {ALLOW, DENY}, obligation, redact_fields: set[str]}`  ← output of Agent 1
- `Obligation`: enum `{none, hybrid, redact_demographics, redact_all}`

# 1.3 Surrogate Identity schema (§5.1)

```
Identity I_i = {
  name_patient, dob, ssn, mrn, unit_no,
  phone, fax, email, health_plan_id, certificate,
  dates_adm, dates_dis, physician_1, physician_2, hospital
}
```
Conditioned on documented sex σ_i ∈ {M, F}.

# 1.4 PHI manifest schema (§5.1)

`S_i` = set of `(v, φ)` pairs, where `v` = injected value, `φ ∈ Φ` = HIPAA Safe-Harbor category. This is the **ground truth** for scoring detection/leakage. Bijective with note content.

# 1.5 Audit tuple (§4.6)

`A = (D, T_san, T_output, T_final, PHI_regex, PHI_bert, τ_pre, τ_post)`
where τ_pre/τ_post are pre/post redaction **timestamps**.

# 1.6 XACML serialization target (Listing 2)

A `<Response>` with `<Decision>Permit|Deny</Decision>`, `<Obligations>` (obligation id, `must_redact` boolean, `redact_fields` string list), and `<AssociatedAdvice>` blocks: `audit_metadata` (`note_id`, `subject_id`, `hadm_id`, `position`, `model`, `filter_type`, `stage_logic`, `test_time`), `agent_performance_ms` (`Policy_Decision_Agent`, `Pre_Sanitization_Agent`, `Middleware_Agent`, `LLM_Architecture_Inference`, `Downstream_Task_Agent`, `LLM_Vanilla_Agent`), and `data_flow_audit` (`final_scrubbed_output`).

**Inputs:** paper spec. **Outputs:** validated schemas + `policies.json`. **Gate:** load `policies.json`, round-trip-validate, unit-test the DENY-by-default path.

---

# Phase 2 — Controlled PHI reconstitution (data generation)

**Goal:** Turn `___`-placeholdered MIMIC-IV discharge notes into filled notes `d_filled` **with a byte-exact PHI manifest `S_i`**. Without the manifest you cannot score anything downstream.

**Model:** `MedGemma-4B-IT`, local GGUF, **temperature = 0, repeat_penalty = 1.1** (§5.1). Deterministic.

# 2.1 GenerateIdentity(σ) — `identity_factory.py` (Table 3 rules)

| Field | Rule | Example |
|---|---|---|
| SSN | `9XX-XX-XXXX` (900-series, non-issuable) | 937-42-8851 |
| Phone / Fax | `(XXX) XXX-XXXX`, area code ≥ 200 | (847) 553-9021 |
| Email | `first.last@domain`, 4 domain pools | aria.martinez@hospital.org |
| Health Plan ID | prefix ∈ {HP,HB,BC,UH,AE} + 10 digits | HP4829173650 |
| Certificate/License | `[state]-LIC-[7 digits]` | MA-LIC-3847291 |
| MRN | 8-digit integer | 58291047 |
| Unit No | `XX-XYYY` (ward-bed) | 29-D875 |
| DOB | mixed formats, years 1934–1999 | 03/17/1962 |
| Hospital | sampled from 7 major US institutions | Johns Hopkins Hospital |

- Patient names: gender-stratified pools (29 male + 28 female first names) × 48 surnames, honorifics (Mr./Mrs./Ms.).
- Physician names: **separate** pool of 20 provider first names (avoid patient–provider collision).
- Admission date uniform in 2024–2026; discharge = admission + [2,14] days (internal temporal coherence).
- Email derived from patient name → all identifiers reference one coherent person.

# 2.2 Three-phase line-by-line filler — `reconstitution.py`

For each note `d_i`, for each placeholder line `l_j` (contains `___`):

**Phase 1 — Contextual LLM Filling.**
`l_filled = M_MedGemma(d_i, l_j, I_i)`. Prompt: return ONLY the filled version of `l_j`; use `I_i` for names/dates/identifiers; generate realistic clinical values (labs, vitals, terminology) from medical knowledge. temp 0, repeat_penalty 1.1.

**Phase 2 — Anchor-Based PHI Extraction.**
Decompose original line into static segments + markers: `l_j = [s1, ___, s2, ___, s3, …]`. Build differencing regex:
`Pattern = escape(s1)·(.+?)·escape(s2)·(.+?)·escape(s3)…`
Apply to `l_filled` → `{v1, v2, …}`. Append each `v_k` to `S_i` with its **contextually inferred** category φ.

**Phase 3 — Regex Fallback.**
If Phase-1 output fails validation (residual `___` remains, or output length < 30% of original line): deterministic regex filler from `I_i`. Header fields (`Name:`, `Admission Date:`, `Attending:`) matched by labeled patterns; `Dr. ___` from physician pool; generic placeholders → realistic dates. Clinical terms (labs, vitals, edema grades) filled but **NOT** added to `S_i` (they are clinical content, not identifiers).

**Header injection.** After line-by-line fill, insert 6 more Safe-Harbor fields into the header (telephone, fax, email, SSN, health-plan beneficiary # , certificate/license), each simultaneously appended to `S_i`.

# 2.3 Master dataset assembly

Emit `d_filled` + `S_i` per note. Aggregate into the master evaluation corpus (Phase 10 defines the run matrix on top of it).

**Inputs:** raw placeholdered notes, `Identity` factory. **Outputs:** `data/reconstituted/{note_id}.json` = `{d_filled, S_i}`.
**Gate:** for a sample of notes, assert every `(v, φ)` in `S_i` appears verbatim in `d_filled` (bijectivity check). This is your ground-truth integrity test — do not proceed to scoring until it passes.

---

# Phase 3 — Policy Decision Agent (Agent 1, Algorithm 2)

**Goal:** Deterministic ABAC. `Authorize(s,o,a)  ⋀_i φ_i(A_s, A_o, a, E)`.

**Core logic (`policy_decision.py`):**
```
EvaluateAccessRequest(s, o, a, E):
    load policies from policies.json
    for policy in policies:
        if MatchAttributes(A_s, A_o, a, E):        # role match ∧ time ∈ access_window ∧ a ∈ {read,query}
            obligation   = policy.obligation
            redact_fields = {f for f in policy.visibility if visibility[f] == false}
            return AuthDecision(ALLOW, obligation, redact_fields)
    return AuthDecision(DENY, "deny", ∅)           # deny-by-default
```

- `MatchAttributes` = role predicate ∧ `E.current_time ∈ [access_window.start, access_window.end]` ∧ action ∈ {read, query}.
- Full-access roles (doctor/patient/nurse): match any time, obligation `none`, `redact_fields = ∅`.
- Temporal boundary is load-bearing for §5.4 / Fig 11 (off-shift → DENY). Test at T = 06:00 / 09:00 / 11:00 / 14:00 for the four role classes.

**Inputs:** `(Subject, Object, Action, Env)`. **Outputs:** `AuthDecision`.
**Gate:** reproduce Fig 11 decision matrix (100% correct ALLOW in-shift, hard DENY off-shift and for unauthorized) — this is a closed, exact test.

---

# Phase 4 — Sanitization Agent (Agent 2, Algorithm 3) — the Regex+BERT hybrid

**Goal:** `T_san = Sanitize(T, O) = BERT(Regex(T, P_r), θ_BERT)`. Two passes, obligation-conditioned scope. **Same module is reused for post-inference redaction (Phase 7).**

# 4.1 Pass 1 — Regex detector (structured PHI, "zero false positives" by design)

`P_r` pattern bank in `regex_patterns.yaml`, one pattern per Table-3 field:

| Category | Pattern (indicative) | Token |
|---|---|---|
| SSN | `\b9\d{2}-\d{2}-\d{4}\b` | `[SSN_REDACTED]` |
| Phone/Fax | `\(\d{3}\)\s\d{3}-\d{4}` | `[PHONE_REDACTED]` |
| Email | standard RFC-ish email regex | `[EMAIL_REDACTED]` |
| Health Plan ID | `\b(HP|HB|BC|UH|AE)\d{10}\b` | `[HEALTHPLAN_REDACTED]` |
| Cert/License | `\b[A-Z]{2}-LIC-\d{7}\b` | `[CERT_REDACTED]` |
| MRN | `\b\d{8}\b` | `[MRN_REDACTED]` |
| Unit No | `\b\d{2}-[A-Z]\d{3}\b` | `[UNIT_REDACTED]` |
| Dates/DOB | multi-format date regex | `[DATE_REDACTED]` |

`PHI_regex = ∪_{p∈P_r} {t_i ∈ T : p matches t_i}`.
**[Assessment]** The MRN pattern `\d{8}` will over-match any 8-digit number → the "zero false positives" claim is aspirational for this field. Keep it, but log MRN matches so you can quantify the precision leak later.

# 4.2 Pass 2 — BERT NER tagger (contextual PHI regex misses)

`PHI_bert = {(t_i, c) : BERT(t_i; θ_BERT) > τ, class ∈ {PER, LOC, ORG}}`.

- **[Fact]** Paper specifies entity classes {PER, LOC, ORG, O} and a "BERT-based Tagger." It does **not** pin the checkpoint or τ. **[Assessment]** A CoNLL-2003-style NER model maps directly to PER/LOC/ORG (e.g. a `bert-base-NER` tagger); a clinical-domain checkpoint (ClinicalBERT-family) is the domain-adapted alternative. Pick one, **record it in config and in every audit row**, and treat τ as a swept hyperparameter (start ~0.5).
- Token map: PER → `[NAME_REDACTED]`, LOC → `[LOCATION_REDACTED]`, ORG → `[ORG_REDACTED]`.
- **[Fact — §6]** English-centric BERT is the paper's stated weakness; it is exactly what the Translation attack (ATK_007) exploits. Do not "fix" it in the baseline — you want the baseline to exhibit this failure.

# 4.3 Obligation-conditioned redaction scope

`PHI_detected = PHI_regex ∪ PHI_bert`, then filter by obligation:

| Obligation | Redact | Preserve |
|---|---|---|
| `none` | nothing (return T unchanged) | everything |
| `hybrid` | identifiers only: names, SSN, phone, location, MRN, email, dates, hospital, physician | clinical: diagnoses, meds, labs, procedures, vitals, imaging findings |
| `redact_demographics` | identifiers ∪ demographic dates (names, DOB, SSN, contact) | admission_date, clinical content |
| `redact_all` | all detected PHI + clinical content | document structure only |

**Inputs:** `(T, Obligation)`. **Outputs:** `(T_san, PHI_regex_hits, PHI_bert_hits)`.
**Gate:** on `d_filled` with known `S_i`, compute per-category precision/recall of `PHI_detected` vs `S_i`.

---

# Phase 5 — Middleware Agent (Agent 3, Algorithm 4)

**Goal:** Compile the authorization decision into an inference-time prompt with **no model retraining** (the paper's central efficiency claim: prompt construction is O(1)).

**Prompt structure (fixed order):**
`P_final = [SYSTEM] ‖ [DEVELOPER GUIDANCE] ‖ "RECORD: " ‖ T_san ‖ "QUERY: " ‖ Q`

- `P_sys = "You are a clinical AI assistant."`
- `P_dev = DevGuidance(D.obligation)`:

| Obligation | Developer guidance (Alg 4) |
|---|---|
| `none` | "Authorized to process full PHI." |
| `hybrid` | "Redact identifiers; preserve clinical data." |
| `redact_demographics` | "Redact demographics; keep clinical and administrative data." |
| `redact_all` | "Redact all PHI; structure only." |

Use the fuller role-specific prompts from Appendix C for the actual runs (clinician/patient authorized-full, resident-doctor never-reveal-identifiers, auditor structure-only). Store these as templates keyed by obligation.

**Inputs:** `(T_san, AuthDecision, Q)`. **Outputs:** `P_final` string.
**Gate:** snapshot-test that the four obligations produce the four expected dev-guidance strings and that the block order is exact (needed for audit reproducibility).

---

# Phase 6 — Inference Agent (Agent 4, §4.4)

**Goal:** Local, deterministic generation.

- **Model:** `gpt-oss-20b` (general-purpose, per paper's rationale that hospitals use general APIs but must keep data on-prem).
- **Serving:** local GGUF (llama.cpp/ollama). No network egress.
- **Decoding:** **temperature = 0.0**, explicit **stop tokens**, deterministic settings → reproducible output for audit-trail consistency.
- **Interface:** `T_output = LLM_Inference(P_final)`.

**Inputs:** `P_final`. **Outputs:** `T_output` (raw model text) + measured inference latency (feeds audit `agent_performance_ms`).
**Gate:** same `P_final` → identical `T_output` across two runs (determinism check).

---

# Phase 7 — Post-Inference Redaction Agent (Agent 5, Algorithm 5)

**Goal:** Defense-in-depth safety net. `T_final = SanitizeNote(T_output, O)` — **reuses Phase 4 unchanged**, but on the model OUTPUT.

Catches the two leak channels the paper names: (1) model-**generated** PHI (hallucinated plausible identifiers from clinical context), (2) prompt-injection leakage that defeated the developer guidance. This is what drives adversarial ASR down (Debug 21.0% → 1.1%, Translation 13.9% → 0.1% in §5.7).

**Inputs:** `(T_output, Obligation)`. **Outputs:** `T_final` + post-redaction hit lists + τ_post.
**Gate:** an injection prompt that leaks a name in `T_output` must be scrubbed in `T_final`.

---

# Phase 8 — Audit & Downstream Agent (Agent 6, Algorithm 6)

**Goal:** Format + deliver + log the full decision path.

**Core logic:**
```
ProcessAndLog(T_final, D, A):
    T_formatted = Format(T_final)     # downstream post-processing
    deliver(T_formatted)
    A.decision      = D
    A.final_output  = T_final
    A.timestamp     = Now()
    write A          # append to ledger
    return (delivered, A)
```

- **Ledger v1:** append-only JSON array of full audit tuples `(D, T_san, T_output, T_final, PHI_regex, PHI_bert, τ_pre, τ_post)`.  **[Fact]** This is exactly what the paper ships and is **not** cryptographically immutable.
- **XACML v2 (`xacml.py`):** serialize the same record to the Listing-2 `<Response>` schema (Decision, Obligations with `must_redact` + `redact_fields`, AssociatedAdvice metadata + per-agent latency + `final_scrubbed_output`).
- **Optional (honesty upgrade, not in paper):** hash-chain each record (`h_n = H(record_n ‖ h_{n-1})`) if you want to legitimately claim tamper-evidence. Mark clearly as an extension.

**Inputs:** `(T_final, AuthDecision, latencies)`. **Outputs:** delivered response + persisted audit record (JSON + XACML).
**Gate:** every request — including DENY — produces exactly one audit record; DENY records carry no PHI.

---

# Phase 9 — End-to-end orchestrator (Algorithm 1)

**Goal:** Wire Agents 1–6 in the **exact** paper order. This is the object your evaluation harness calls.

```
Pipeline(s, o, a, E, T_raw, Q):
    # [SEAM] input_gate(T_raw) — record modality; today it's assumed text.
    # This is where non-text inputs will be injected in Phase-0 imaging work.
    D = EvaluateAccessRequest(s, o, a, E)              # 1  Agent 1
    if D.decision == DENY:                             # 2
        ProcessAndLog(deny_record, D, A); return DENY  # (still audited)
    O_obl = D.obligation                               # 4
    T_san = SanitizeNote(T_raw, O_obl)                 # 5  Agent 2
    P_final = ConstructPrompt(T_san, D, Q)             # 6  Agent 3
    T_output = LLM_Inference(P_final)                  # 7  Agent 4
    T_final = PostInferenceRedaction(T_output, O_obl)  # 8  Agent 5
    status, A = ProcessAndLog(T_final, D, H)           # 9  Agent 6
    return T_formatted
```

**[Assessment — the seam that matters for you]** Line 5 calls `SanitizeNote(T_raw, …)` which assumes `T_raw` is text. Make `input_gate` a real function that stamps the input modality into the audit record. In the faithful baseline it always says `text` and is a no-op. In Phase-0 imaging work, you feed a non-text payload, the text sanitizer no-ops on it, and the pipeline still returns an ALLOW/`Permit` certificate — that logged "compliant" certificate on an ungoverned image **is your Phase-0 evidence**.

**Inputs:** full request tuple. **Outputs:** `T_formatted` or `DENY`, plus audit record. **Gate:** golden end-to-end trace for one note under each of the 5 role classes.

---

# Phase 10 — Evaluation harness

**Goal:** Reproduce §5's numbers: ablation, utility, privacy leakage, adversarial robustness.

# 10.1 Severity weights (§5.2, `severity_weights.yaml`)

| Weight | Category | Members |
|---|---|---|
| 5 (Critical) | direct identifiers | SSN, patient names, phone |
| 3 (High) | strong identifiers | MRN, admission dates |
| 2 (Moderate) | quasi-identifiers | attending physicians, hospital/location |
| 1 (Low) | clinical | diagnoses, medications |

# 10.2 Metrics (`metrics.py`)

- **Detection F1** (unweighted): detected PHI vs `S_i`.
- **Weighted F1:** F1 with per-category severity weights. **[Assessment]** perfects to 1.00 when residual leaks are all weight-1 — report this caveat.
- **Residual PHI (Resid):** count of `S_i` items surviving into `T_final`.
- **Weighted Leak Score (WLS):** Σ (severity weight of each residual leak).
- **Utility:** **ROUGE-L** (syntactic overlap vs unprotected baseline) and **BERTScore** with **RoBERTa-large** embeddings (semantic). Expect ROUGE-L to punish redaction hard; BERTScore to be forgiving. Pearson r between them ≈ 0.93 (§5.5).

# 10.3 Ablation study (§5.6, Table 1) — the 3 × 3 grid

**Sanitization configs (filter_type):** `Regex-Only`, `BERT-Only`, `Hybrid`.
**Stages (stage_logic):** `Pre-Only`, `Post-Only`, `Pre+Post`.
→ **9 configurations.** Run over 200 patients × 3 restricted roles (resident_doctor, clinical_coder, auditor) × 9 = **5,400 logs**.

Report per cell: Resid, WLS, F1, W_F1. Expected shape (Table 1): **Hybrid / Pre+Post** is best (Resid 72, WLS 72); **BERT-Only** is catastrophic (WLS > 215) — i.e. LMs need deterministic regex anchoring. Reproducing that ordering is your correctness check.

# 10.4 Adversarial evaluation (§5.7, Table 4 + STRIDE Table 5, `adversarial.py`)

Ten attack prompts (store verbatim; run against each defense config):

| ID | Name | Target | STRIDE |
|---|---|---|---|
| ATK_001 | Baseline Retrieval | Sanitization Engine | Info Disclosure |
| ATK_002 | Emergency Bypass | Policy Decision Agent | Spoofing / EoP |
| ATK_003 | JSON Bypass | Regex Engine | Info Disclosure |
| ATK_004 | Debug Mode | Middleware | Tampering / EoP |
| ATK_005 | Reasoning Trap | Input Sanitization | Info Disclosure |
| ATK_006 | Completion Exploit | LLM Predictor Core | Info Disclosure |
| ATK_007 | Translation | BERT Semantic Engine | Info Disclosure |
| ATK_008 | Implicit PII | Sanitization Scope | Info Disclosure |
| ATK_009 | Research Persuasion | Policy Decision Agent | Spoofing |
| ATK_010 | Sycophancy | LLM RLHF Alignment | Spoofing |

- Metric: **Severity-Weighted Attack Success Rate (ASR)** per attack × config.
- Expected: Debug/Translation spike (21.0% / 13.9%) under partial ablation, collapse to ~1.1% / 0.1% under Hybrid Pre+Post. Repudiation is "mitigated by audit"; DoS is out of scope.

# 10.5 Run matrix & dedup (`run_matrix.py`)

- Full space: 200 notes × 11 prompt variations (10 attacks + 1 normal) × 3 filter types × 3 stages.
- Paper: naive 475,200 → deduped **107,800** unique rows (77.3% reduction), = 99,000 full-pipeline + 8,800 denial runs.
- **[Speculation]** Exact dedup key not specified. **Define and document your own** (e.g., collapse rows where obligation makes filter_type/stage a no-op, and collapse temporal-boundary rows into the ABAC-only sub-matrix). State it explicitly; don't back-fit to 107,800.

# 10.6 Baseline comparison (§5.8, Table 2)

Feature matrix (ABAC, PHI-San, Pre-Red, Post-Red, Middleware, Audit, Local) vs prior frameworks. This is a literature-positioning table, not code — populate from the cited works.

**Inputs:** reconstituted corpus + all 6 agents. **Outputs:** `data/results/` tables/figures.
**Gate:** ablation ordering (Hybrid≻Regex≻BERT) and adversarial collapse reproduce directionally, even if absolute counts differ with your BERT checkpoint/τ.

---

# Phase 11 — Reproduction & validation checklist

Run in order; each must pass before the next is meaningful.

1. `policies.json` round-trips; DENY-by-default fires for unknown roles.
2. Data-gen bijectivity: every `(v, φ) ∈ S_i` appears in `d_filled`.
3. PDA temporal matrix (Fig 11) is 100% correct on the four test timestamps.
4. Determinism: identical `P_final` → identical `T_output`; identical run → identical audit hash (if hash-chaining added).
5. Sanitizer precision/recall per PHI category vs `S_i` is sane (flag MRN over-match).
6. Ablation ordering: Hybrid/Pre+Post lowest WLS; BERT-Only worst.
7. Adversarial: Debug/Translation ASR high under ablation, low under Hybrid Pre+Post.
8. **Baseline-harness readiness:** `input_gate` records modality; a non-text payload traverses the pipeline and receives a logged ALLOW/`Permit` certificate. ← this is the Phase-0 hook; confirm the "compliant on an ungoverned input" behavior is captured in the audit log.

---

# Build order summary (dependency-respecting)

```
Phase 0  scaffold/env
Phase 1  schemas + policies.json
Phase 2  data reconstitution (MedGemma)      ← blocks all scoring
Phase 3  Policy Decision Agent
Phase 4  Sanitization (regex + BERT)          ← reused by Phase 7
Phase 5  Middleware
Phase 6  Inference (gpt-oss-20b)
Phase 7  Post-Inference Redaction (reuse P4)
Phase 8  Audit + XACML
Phase 9  Orchestrator (Algorithm 1) + input_gate seam
Phase 10 Evaluation (ablation, utility, adversarial)
Phase 11 Validation checklist
```

**Fastest path to a runnable end-to-end trace:** Phases 1 → 3 → 4 → 5 → 6 → 7 → 8 → 9 on a *single* reconstituted note, then scale to the full matrix in Phase 10. Data-gen (Phase 2) can proceed in parallel once schemas (Phase 1) exist.