# Evidence-Reliable Coordination for Multimodal Clinical Agents
# Literature Survey & Evaluation-Paradigm Analysis

> Companion analysis to the concept note *When Specialist Agents Fail to Collaborate: Benchmarking Evidence-Reliable Coordination in Multimodal Clinical AI*.

---

# Part 1 — Related Works and Existing Implementations

# 1.1 Overview

No exact implementation of the proposed Evidence-Reliable Coordinator (ERC) was found in the literature, but a rapidly growing body of work addresses each of its six design principles in isolation. The closest neighbours are a clinical multi-agent uncertainty-lifecycle framework (echocardiography), an evidence-grounded single-coordinator VQA system (CARE), a contribution-aware game-theoretic oncology framework (CoMMa), and the AgentRx benchmark that empirically demonstrates the coordination gap this concept targets.


# 1.2 Clinical Multi-Agent Coordination Frameworks (Direct Baselines)

| System | Venue / Source | What it does | Mechanism relevant to ERC | Gap vs. ERC |
|---|---|---|---|---|
| **AgentRx** | CHIL 2026, arXiv 2605.10286 | Benchmarks single vs. multi-agent LLMs on MIMIC-IV/CXR/Note for mortality & length-of-stay | Reports the calibration divergence: single-agent ECE *improves* with more modalities (0.0233) while majority-vote ECE *degrades* (0.111); Traj-CoA retains raw multimodal input for the final decision-maker (AUROC 0.762) | Diagnoses the gap but proposes no coordinator; no structured evidence packets, no missingness handling, no conflict preservation |
| **MedAgentBoard** | NeurIPS 2025 | Four task categories (Med-VQA, lay summary, EHR prediction, workflow automation); compares multi-agent vs. single-LLM vs. conventional | Shows multi-agent helps only in workflow completeness; trails conventional methods on VQA/EHR | No stress tests for missing modality, tool corruption, or cross-modal conflict; no evidence-provenance metric |
| **MDAgents** | NeurIPS 2024 Oral | Adaptive assignment of solo/group collaboration by task complexity; hierarchical consensus | Dynamic role allocation; reports robustness over solo/group across temperatures | Free-text exchange; no source-evidence re-checking; no explicit uncertainty fields |
| **AgentClinic** | Nature Digital Medicine 2026 | Sequential dialogue-based doctor/patient agents, tool use, 23 simulated biases | Tests temporal evidence reconciliation, staged information reveal | Simulated dialogue, not a modality-fusion coordinator; no structured evidence packets at the coordinator level |
| **MAM (Modular Multi-Agent)** | ACL 2025 Findings | Role-based specialists with multi-round debate | Ablates discussion rounds and roles; notes debate can overwrite correct answers | Relies on free-text debate; no uncertainty representation or provenance |
| **CoMMa** | ICML, arXiv 2602.09159 | Data-decentralized specialists on oncology tumour-board; deterministic embeddings + Shapley-value credit assignment | Contribution-aware marginal-utility attribution per agent; interpretable pathways | Game-theoretic credit, not uncertainty/missingness; no abstention, no source re-checking of raw evidence |
| **Cerebra** | NYU / Razavian lab | Multi-agent dementia risk over EHR, notes, imaging across ~3M patients | Gracefully handles missing MRI; exposes per-agent reasoning | Proprietary; not a benchmark; no controlled perturbation suite, no conflict-preservation metric |
| **"Are we going MAD?"** | NeurIPS 2023 | Benchmarks debate strategies for medical Q&A | Cost/time/accuracy trade-offs; agent-agreement prompting | Medical Q&A only, single modality; no multimodal coordination |

# 1.3 Evidence-Grounded & Provenance Mechanisms

| System | What it does | Overlap with ERC | Gap vs. ERC |
|---|---|---|---|
| **CARE** (arXiv 2603.01607) | Decomposes medical VQA into entity proposal → pixel-level ROI segmentation → grounded reasoning; a VLM coordinator plans tool calls and reviews evidence-answer consistency; RL with verifiable rewards | Closest to ERC principles 1, 3, 6: keeps claims tied to visual evidence and verifies final answer against it | Single coordinator over one image modality, not multi-specialist; no missing-modality or conflict handling; VQA not risk prediction |
| **MedAgent-Pro** (ICLR 2026) | Hierarchical workflow: disease-level RAG plan + patient-level step-by-step reasoning; verifies reliability of each step; integrates visual tools | Step-level verification mirrors ERC's "re-check" principle | Per-step verification within one pipeline, not inter-specialist coordination; the disc-haemorrhage F1 drop is exactly the failure mode it does not yet prevent at the coordinator level |
| **ATRIA** (arXiv 2606.24392) | Orchestrator + analysis + report + literature + review agents with a shared artifact store for iterative, traceable ECG reports; binds every report claim to supporting evidence | Shared artifact store approximates evidence packets; review agent enforces provenance | Single modality (ECG); no missingness or cross-modal conflict logic |
| **Evidence Tracing & Execution Provenance survey** (arXiv 2606.04990) | Surveys provenance mechanisms in LLM agents | Taxonomy of claim→evidence linking | Survey, no clinical benchmark |
| **ClaimVerAgents / tabular claim verification** (arXiv 2604.17225) | Multi-agent User→Planner→Executor→Verifier with revision loops for claim verification | Revision mechanism prevents unidirectional error propagation | General claim verification, not clinical multimodal coordination |

# 1.4 Uncertainty Representation & Disagreement Handling

| System | What it does | Overlap with ERC | Gap vs. ERC |
|---|---|---|---|
| **Managing Uncertainty in LLM-based MAS** (arXiv 2602.23005) | Distinguishes epistemological vs. ontological uncertainty; lifecycle framework: representation, identification, evolution, adaptation; validated on a clinical echocardiography MAS | **Single closest conceptual neighbour to ERC principle 2** — treats uncertainty as a first-class system property propagating through coordination, not just model output | Software-engineering framework, not a controlled benchmark; no comparison arms, no missing-modality/corruption stress tests, no accuracy/calibration/cost reporting |
| **DiverseAgentEntropy** | Quantifies LLM uncertainty by querying diverse-perspective agents | Multi-agent UQ | Black-box LLM UQ, not clinical coordination |
| **MMedAgent-RL** (arXiv 2506.00555) | RL-trained triage + attending-physician agents; curriculum RL teaches the attending to balance imitating vs. correcting noisy/conflicting specialists | Directly addresses ERC principle 4 (disagreement as signal) via learned correction | Learns a correction policy rather than preserving/reportable conflict; no structured evidence packets; VQA only |
| **Diversity Collapse & Manufactured Overconfidence** (OpenReview) | Shows debate makes inter-agent correlation →1; apparent confidence saturates *independently of error* (17× smaller variance than accuracy) | Theoretical justification for ERC's "treat disagreement as signal" and "avoid forced agreement" | Diagnostic theory, no clinical intervention |
| **Confidence Calibration Through Multi-Agent Interaction in VQA** (AAAI) | Studies how multi-agent interaction shifts calibration | Calibration under interaction | VQA, not clinical risk prediction |
| **Voting or Consensus?** (ACL 2025 Findings, arXiv 2502.19130) | Benchmarks decision-making protocols in multi-agent debate; voting +13.2% on reasoning, consensus +2.8% on knowledge; more discussion rounds reduce performance | Protocol choice affects final output | General, no clinical/missing-modality setting |

# 1.5 Missing-Modality Robustness

| System | What it does | Gap vs. ERC |
|---|---|---|
| **Handling Missing Modalities in Patient Clinical Data** (arXiv 2604.18753) | Autoregressive sequence modeling with missingness-aware contrastive pretraining across modalities | Single unified model, not multi-agent; no coordinator-level missingness disclosure or abstention |
| **awesome-missing-modality-for-medical-images** repo | Tracks missing-modality methods (SimMLM, IM-Fuse, mmFormer, ShaSpec…) | Almost entirely supervised segmentation (BraTS); no agent-coordination framing |
| **Cerebra** | Gracefully handles missing MRI in dementia risk | Proprietary, not a benchmark arm |

# 1.6 Error-Cascade Mitigation

| System | What it does | Gap vs. ERC |
|---|---|---|
| **From Spark to Fire** (arXiv 2603.04474) | Models how small local errors propagate through LLM-MAS collaboration; mitigation raises defense success rate 0.32→0.89 via unsupported-claim tracking | General LLM-MAS, not clinical multimodal; no modality-specific stress tests |
| **Multi-Agent Verification (MAV)** (arXiv 2502.20379) | Test-time compute combining multiple verifiers | Verifier-combination, not evidence-reliable coordination of specialists |
| **Multi-agent collusion in healthcare IoT** (arXiv 2512.03097) | Shows adversarial assistants can collude to create false consensus pushing an AI doctor toward harmful prescriptions | Security angle; reinforces ERC's "avoid forced agreement" but offers no reliability coordinator |
| **Adversarial Debate and Voting Mechanisms** (MDPI, 15(7):3676) | Multi-agent LLM framework with adversarial + voting mechanisms to address hallucination | General; no clinical multimodal coordination |

# 1.7 Abstention / Selective Prediction

| System | What it does | Gap vs. ERC |
|---|---|---|
| **Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty** (arXiv 2601.12471) | Argues abstention is vital and largely unaddressed in medical LLM benchmarks | Single-model abstention, not multi-agent selective escalation |
| **When silence is safer: LLM abstention in healthcare** (review) | Reviews abstention behaviours/methods/benchmarks | Survey; no coordination-level abstention |
| **Cascaded LLMs for cost-effective human-AI decision-making** (NeurIPS 2025, arXiv 2506.11887) | Adaptive delegation base→large→human with abstention; deferral + abstention policies with online learning | Single cascade, not multi-specialist coordination |

# 1.8 Consolidated Gap Statement

No located work combines all six ERC principles — structured evidence packets, explicit uncertainty/missingness fields, source-evidence re-checking, conflict preservation, graceful degradation, and evidence-tied explanation — inside a controlled benchmark that compares unified agents, naïve multi-agent systems, supervised fusion models, and the coordinator under identical clinical perturbations. The *Managing Uncertainty* lifecycle paper supplies the conceptual scaffold for principle 2, CARE supplies principles 1/3/6 for single-image VQA, CoMMa supplies principle 4 via game-theoretic credit, and AgentRx supplies both the empirical motivation and the benchmark substrate. The ERC's contribution is the integration and the controlled stress-test suite, not any single mechanism.

---

# Part 2 — Agentic Coordination Evaluation vs. Model Robustness Evaluation

They are related but distinct evaluation paradigms. The core divergence: model robustness probes a single mapping `f(x) → y` under input perturbations, whereas agentic-coordination evaluation probes a *system of interacting components* under architectural and informational stress. A model can be perfectly robust in isolation while the multi-agent system around it fails — and vice versa.

# 2.1 Side-by-Side Comparison

| Dimension | Model Robustness evaluation | Agentic Coordination evaluation |
|---|---|---|
| **Unit of analysis** | A single model's input→output mapping | A topology of interacting specialists + coordinator + tools |
| **What is stressed** | The input `x`: noise, blur, masking, adversarial perturbation, distribution shift, OOD samples | The architecture: inject incorrect/noisy/delayed specialist outputs, remove an agent, force cross-modal conflict, compress communication to summaries only |
| **Core failure modes** | Accuracy degradation under perturbation; miscalibration; adversarial vulnerability; covariate shift | Error cascades across agents; communication-bottleneck information loss; forced/false consensus; sycophancy; manufactured overconfidence from debate; silent pipeline degradation |
| **Treatment of disagreement** | N/A — a single model emits one output | First-class concern: distinguish explainable difference vs. unresolved conflict vs. unreliable output; preserving discordance may be the *correct* behaviour |
| **Ground truth** | Usually a single fixed label per case | Often trajectory-level or absent; in workflow automation there may be no single correct answer, requiring factuality metrics instead |
| **Metric families** | Accuracy/AUROC drop under noise, ECE/Brier, adversarial accuracy, OOD detection (AUROC) | *Plus* error-cascade resistance, evidence fidelity/provenance, selective-abstention curves, coordination cost (tokens, latency, API calls, GPU time), tool-selection accuracy, information-aggregation quality |
| **Cost / efficiency** | Typically ignored | Mandatory: coordination overhead grows roughly quadratically with agent count; CoMMa-style and AgentRx ablations explicitly trade accuracy vs. tokens/calls |
| **"Correct" abstention** | Often penalised (a refusal lowers coverage) | Rewarded: requesting another modality, down-weighting a tool, or escalating to human is positive behaviour when evidence is genuinely missing/conflicting |
| **Explanation quality** | Usually post-hoc attention/saliency | Must be tested for fidelity: does the cited evidence actually support the decision? (TBFact, RadFact-style entailment) |

# 2.2 Why the Distinction Matters Concretely

AgentRx makes the cost of confusing the two visible. On its mortality task, the Qwen single-agent is well calibrated (ECE 0.0233) and the majority-vote multi-agent is badly miscalibrated (ECE 0.111) — yet both report similar AUROC. A pure model-robustness lens (AUROC under modality removal) would declare the systems comparable; a coordination lens reveals that the voting mechanism *distorts probabilistic confidence* and that single-agent calibration improves with modalities while multi-agent calibration degrades. The "Diversity Collapse" result sharpens this: in debate, terminal confidence has 17× smaller variance than accuracy, so apparent confidence saturates independently of error — a failure mode that no input-perturbation robustness test can expose.

# 2.3 Metrics Unique to Coordination Evaluation (Absent from Standard Robustness)

1. **Error-cascade resistance** — performance drop when one specialist or tool is deliberately corrupted (the MedAgent-Pro disc-haemorrhage example); the "From Spark to Fire" defense-success-rate paradigm operationalises this.
2. **Evidence fidelity / provenance** — whether the final answer references source evidence that was actually provided and relevant; TBFact (factual inclusion/distortion/omission, κ = 0.760 with experts) and RadFact are the closest existing instruments.
3. **Selective safety / abstention curves** — risk-coverage trade-off when the system may decline or escalate; distinct from a model's confidence threshold because the agent can *act* (request another modality, re-check) rather than merely abstain.
4. **Coordination cost** — tokens, API calls, latency, GPU time per case; AgentRx, MedAgentBoard, and MDAgents all report this because it determines whether agentic complexity is justified.
5. **Information-aggregation quality** — whether the orchestrator correctly routes, resolves intent, and synthesises multiple agent outputs; measured by orchestrator-level metrics in the Healthcare Agent Orchestrator framework.
6. **Conflict preservation** — whether unresolved cross-modal discordance is reported rather than averaged away; no standard metric exists yet, which is itself a contribution opportunity.

# 2.4 Positioning the Benchmark

Because the two paradigms diverge on unit of analysis, failure modes, and metrics, the benchmark arms (Section 8 of the concept note) and stress conditions (Section 7) are correctly framed as a *coordination* benchmark, not a robustness benchmark. The key methodological moves that follow from the distinction:

1. Keep the supervised multimodal predictor as the non-agent ceiling so that robustness-style gains are not misattributed to coordination.
2. Report calibration *conditional on perturbation type* (missing vs. corrupted vs. conflicting), since AgentRx shows calibration can move opposite to accuracy under multi-agent voting.
3. Treat abstention/escalation as a positive outcome with its own risk-coverage curve rather than as missing predictions.
4. Add a provenance metric (TBFact/RadFact-adapted) so that evidence fidelity is measured rather than narrated.

These four choices are what separate a coordination benchmark from a robustness benchmark applied to an agent, and they are where the proposed ERC study can claim methodological novelty on top of its architectural contribution.

---

# References

# Clinical Multi-Agent Coordination Frameworks

1. AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks — https://arxiv.org/html/2605.10286v1
2. AgentRx (OpenReview) — https://openreview.net/forum?id=wtI0bIBaAc
3. AgentRx code — https://github.com/nyuad-cai/AgentRX
4. AgentClinic website — https://agentclinic.github.io
5. AgentClinic (Nature Digital Medicine) — https://www.nature.com/articles/s41746-026-02674-7
6. AgentClinic (arXiv 2405.07960) — https://arxiv.org/abs/2405.07960
7. AgentClinic code — https://github.com/SamuelSchmidgall/AgentClinic
8. MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents (NEJM AI) — https://ai.nejm.org/doi/full/10.1056/AIdbp2500144
9. MDAgents (NeurIPS 2024) — https://neurips.cc/virtual/2024/poster/96041
10. MDAgents (arXiv 2404.15155) — https://arxiv.org/html/2404.15155v2
11. MDAgents code — https://github.com/mitmedialab/MDAgents
12. CoMMa: Contribution-Aware Medical Multi-Agents From a Game-Theoretic Perspective (arXiv 2602.09159) — https://arxiv.org/html/2602.09159v1
13. MedAgentBoard (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/121792
14. MedAgentBoard (OpenReview) — https://openreview.net/forum?id=BPpG4qQaNj
15. MedAgentBoard (arXiv 2505.12371) — https://arxiv.org/abs/2505.12371
16. MedAgentBoard code — https://github.com/yhzhu99/medagentboard
17. MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis (arXiv 2506.19835) — https://arxiv.org/pdf/2506.19835
18. MAM (ACL 2025 Findings) — https://aclanthology.org/2025.findings-acl.1298.pdf
19. "Are we going MAD?" Benchmarking Multi-Agent Debate (NeurIPS 2023) — https://neurips.cc/virtual/2023/75421

# Evidence-Grounded & Provenance Mechanisms

20. CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning (arXiv 2603.01607) — https://arxiv.org/abs/2603.01607
21. CARE (HTML) — https://arxiv.org/html/2603.01607v1
22. CARE (OpenReview) — https://openreview.net/forum?id=whRAOJiyHM
23. MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis (arXiv 2503.18968) — https://arxiv.org/abs/2503.18968
24. MedAgent-Pro (ICLR 2026) — https://iclr.cc/virtual/2026/poster/10008810
25. MedAgent-Pro code — https://github.com/jinlab-imvr/MedAgent-Pro
26. ATRIA: Adaptive Traceable ECG Reporting with Iterative Agents (arXiv 2606.24392) — https://arxiv.org/abs/2606.24392
27. ATRIA (HTML) — https://arxiv.org/html/2606.24392v1
28. A Survey of Evidence Tracing and Execution Provenance in LLM Agents (arXiv 2606.04990) — https://arxiv.org/html/2606.04990
29. A Multi-Agent Approach for Claim Verification from Tabular Data (arXiv 2604.17225) — https://arxiv.org/html/2604.17225v1
30. ClaimVerAgents (IEEE AICCSA 2025) — https://www.computer.org/csdl/proceedings-article/aiccsa/2025/11315316/2d1RDUeZHS8

# Uncertainty Representation & Disagreement Handling

31. Managing Uncertainty in LLM-based Multi-Agent System Operation (arXiv 2602.23005) — https://arxiv.org/html/2602.23005v1
32. DiverseAgentEntropy: Quantifying Black-Box LLM Uncertainty through Multi-Agent Interaction (OpenReview) — https://openreview.net/forum?id=AJAStQYZaL
33. MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning (arXiv 2506.00555) — https://arxiv.org/abs/2506.00555
34. MMedAgent-RL (HTML) — https://arxiv.org/html/2506.00555v1
35. Diversity Collapse and Manufactured Overconfidence in Multi-Agent LLM Debate (OpenReview PDF) — https://openreview.net/pdf/ef389139d4a3f634c4cfc7c98434e3a21d4cd52d.pdf
36. Confidence Calibration Through Multi-Agent Interaction in VQA (AAAI) — https://ojs.aaai.org/index.php/AAAI/article/view/41117/45078
37. Voting or Consensus? Decision-Making in Multi-Agent Debate (arXiv 2502.19130) — https://arxiv.org/abs/2502.19130
38. Voting or Consensus? (ACL 2025 Findings) — https://aclanthology.org/2025.findings-acl.606
39. Demystifying Multi-Agent Debate: The Role of Confidence and Diversity (arXiv 2601.19921) — https://arxiv.org/html/2601.19921v2

# Missing-Modality Robustness

40. Handling and Interpreting Missing Modalities in Patient Clinical Data (arXiv 2604.18753) — https://arxiv.org/html/2604.18753v1
41. awesome-missing-modality-for-medical-images (GitHub) — https://github.com/han-liu/awesome-missing-modality-for-medical-images
42. Cerebra — Interactive AI Agent for Dementia Risk Assessment — https://cerebra-health.com
43. Cerebra (overview) — https://medium.com/@zljdanceholic/stanfords-cerebra-a-multi-agent-consensus-approach-to-dementia-risk-8f567d8c61ba

# Error-Cascade Mitigation

44. From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration (arXiv 2603.04474) — https://arxiv.org/abs/2603.04474
45. From Spark to Fire (HTML) — https://arxiv.org/html/2603.04474v1
46. Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers (arXiv 2502.20379) — https://arxiv.org/html/2502.20379v1
47. Exposing Multi-Agent Collusion Risks in AI-Based Healthcare (arXiv 2512.03097) — https://arxiv.org/html/2512.03097v1
48. Adversarial Debate and Voting Mechanisms in LLM-Based Multi-Agent Systems (MDPI 15(7):3676) — https://www.mdpi.com/2076-3417/15/7/3676

# Abstention / Selective Prediction

49. Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty (arXiv 2601.12471) — https://arxiv.org/html/2601.12471v2
50. When silence is safer: a review of LLM abstention in healthcare — https://www.researchgate.net/publication/398678260_When_silence_is_safer_a_review_of_LLM_abstention_in_healthcare
51. Cascaded Language Models for Cost-Effective Human-AI Decision-Making (arXiv 2506.11887) — https://arxiv.org/abs/2506.11887
52. Cascaded Language Models (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/119942
53. Cascaded LLMs code — https://github.com/fanconic/cascaded-llms

# Coordination Evaluation Frameworks & Metrics

54. Towards Robust Evaluation of Multi-Agent Systems in Clinical Settings (Microsoft Healthcare Blog; TBFact metric) — https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/towards-robust-evaluation-of-multi-agent-systems-in-clinical-settings/4435119
55. Agent Evaluation vs. Model Evaluation (Maxim AI) — https://www.getmaxim.ai/articles/agent-evaluation-vs-model-evaluation-whats-the-difference-and-why-it-matters
56. Evaluation and Benchmarking of LLM Agents: A Survey (arXiv 2507.21504) — https://arxiv.org/html/2507.21504v1
57. AI Agent Evaluation: Key Methods & Insights (Galileo) — https://galileo.ai/blog/ai-agent-evaluation
58. What is AI Agent Evaluation? (Databricks) — https://www.databricks.com/blog/what-is-agent-evaluation
59. Evaluating AI agents: Real-world lessons from building agentic systems at Amazon (AWS) — https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon
60. AI Agent Observability vs. Benchmarking vs. Evaluation (Galileo) — https://galileo.ai/blog/ai-agent-measurement-guide-observability-benchmarking-evaluation

# Concept-Note Internal Datasets & Resources

61. NCI: TCGA Data Types Collected — https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data/types
62. NCI Genomic Data Commons Data Portal — https://gdc.cancer.gov/access-data/gdc-data-portal
63. ISB-CGC: TCGA Radiology and Pathology Image Data Set — https://isb-cancer-genomics-cloud.readthedocs.io/en/latest/sections/data/TCGA-images.html
64. Landmark of Medical Agent (curated resource list) — https://nus-project.github.io/Landmark-of-medical-agent