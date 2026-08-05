# Paper Review Progress Tracker

# Summary Statistics
- **Total Papers Cataloged:** 100 / ~500
- **Papers Read (Notes Complete):** 100
- **Current Batch:** Batch 20 Complete

---

# Batch Log

# Batch 1 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 001   | SEMA-RAG | Clinical Decision Support | Shift from linear RAG to multi-agent iterative retrieval loops | Complete  |
| 002   | CHI-Bench | Administrative & Operational | Visual UI agents fail at long-horizon tasks due to compounding errors | Complete  |
| 003   | COTCAgent | Clinical Decision Support | POMDP CoT allows agents to proactively ask questions rather than just answer | Complete  |
| 004   | DermAgent | Multi-Tool Agentic System | Plan-Execute-Reflect loop drastically reduces hallucinations in medical VLM diagnosis | Complete  |
| 005   | RL FHIR Tool-Calling | EHR Integration | RLHF can train agents to generate syntactically perfect complex FHIR payloads | Complete  |

# Batch 2 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 006   | Population-Scale Mental Health | Clinical Decision Support | Empathetic dual-agent system scales mental health screening dynamically | Complete  |
| 007   | MedMemoryBench | Clinical Decision Support | Highlights the "Memory Gap" where agents fail to track longitudinal EHR visits | Complete  |
| 008   | ABRA Radiology | Medical Vision | Interactive radiology benchmark shows VLMs need tool-calling for spatial tasks | Complete  |
| 009   | AgentRx | EHR & Operational | Benchmark for agentic orchestration of tabular and unstructured data models | Complete  |
| 010   | Synthetic EHR ABM | Synthetic Data | Skipped: Paper is about traditional epidemiological ABMs for generating synthetic data, not LLM agents performing reasoning tasks. | Skipped ⏭ |

# Batch 3 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 011   | DeepTumorVQA | Medical Vision | VQA benchmarks must decompose reasoning hierarchically (Recognition -> Measurement -> Clinical). | Complete  |
| 012   | Multi-Drone Coordination | Drone Logistics | Skipped: Traditional drone logistics MARL during disaster response, not clinical AI. | Skipped ⏭ |
| 013   | Conversational AI with Eyes, Ears | Multimodal Agents | Dual-agent (Talker/Planner) systems are crucial for safe multimodal medical interfaces. | Complete  |
| 014   | AI-Care (Alzheimer's) | Patient-Facing Copilots | LangGraph orchestrations provide the stateful safety constraints needed for vulnerable patients. | Complete  |
| 015   | Measuring What Matters | Benchmarking Framework | "Maturity Ladder" frameworks are needed to shift focus from raw accuracy to workflow safety. | Complete  |

# Batch 4 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 016   | Security Enhancement for Adversarial Agents | Security & Safety Benchmarking | Agentic Firewalls are required to sanitize inputs to prevent prompt injection and poisoning. | Complete  |
| 017   | MedExAgent | Clinical Decision Support | Agents must be trained (and evaluated) on noisy, contradictory clinical environments. | Complete  |
| 018   | Hygieia (Rare Disease) | Genomics & Omics Agents | Generalist agents can diagnose rare diseases when given dynamic OMIM/ClinVar search tools. | Complete  |
| 019   | SymptomAI | Patient-Facing Copilots | Conversational triage needs RLHF penalties to prevent missing critical "red flag" symptoms. | Complete  |
| 020   | CuraView | Knowledge Retrieval | GraphRAG far outperforms Vector RAG for mitigating medical hallucinations in agents. | Complete  |

# Batch 5 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 021   | Healthcare AI GYM | Benchmarking Framework | Need standardized RL environments (like OpenAI Gym) specifically tailored for healthcare workflows. | Complete  |
| 022   | Empirical Study of Agent Skills | Taxonomy & Methodology | Massive capability gap between "Read" skills (low failure) and "Write" skills (high failure) in medical agents. | Complete  |
| 023   | PhysicianBench | Administrative & Operational | UI navigation inside complex EHRs (EPIC/Cerner clones) causes severe looping failures in ReAct agents. | Complete  |
| 024   | GAZE (Rare Brain MRI) | Medical Vision | Zero-shot diagnosis of rare diseases requires tools (contrast/scrolling) and integrated literature retrieval. | Complete  |
| 025   | ClinicBot | Clinical Decision Support | Prioritized Evidence RAG forces verifiable citations, solving the "Trust" gap for clinicians. | Complete  |

# Batch 6 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 026   | Echo-α Ultrasound Agent | Medical Vision | Agents need spatiotemporal "Video Navigator" tools to parse noisy continuous ultrasound feeds. | Complete  |
| 027   | Detecting Clinical Discrepancies | Patient-Facing Copilots | Dual-stream architectures with Reconciliation Agents prevent contradictory coaching advice. | Complete  |
| 028   | CareGuardAI | Security & Safety Benchmarking | Context-aware, multi-agent guardrails drastically reduce hallucination safety risks vs standard keyword blocklists. | Complete  |
| 029   | Green Shielding Trustworthy AI | Trustworthy AI | Skipped: General AI Trust and sustainability framework; lacks specific healthcare task/agent benchmarking relevance. | Skipped ⏭ |
| 030   | FastOMOP | EHR & Operational | Tool-calling APIs (FastOMOP) prevent the severe hallucinations typical of raw Text-to-SQL agents on EHR data. | Complete  |

# Batch 7 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 031   | Vibe Medicine | Perspective/Opinion | Skipped: Generic perspective on human-AI collaboration paradigms, not an empirical benchmark. | Skipped ⏭ |
| 032   | Thinking Like a Clinician | Clinical Decision Support | Multi-agent Adversarial Debate (Proponent vs Skeptic) prevents confirmation bias on complex cases. | Complete  |
| 033   | Agentic AI for Personalized Physiotherapy | Medical Vision | Edge-to-cloud multi-agent systems are necessary for real-time skeletal pose correction. | Complete  |
| 034   | Interpretable Sepsis Early Warning | Clinical Decision Support | LLM agents orchestrating ODE physiological simulations increase clinician trust in ICU alerts. | Complete  |
| 035   | MedSkillAudit | Taxonomy & Methodology | First domain-specific audit framework for medical agent *tools/APIs*, proving 22% hallucinate on edge cases. | Complete  |

# Batch 8 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 036   | Hospital Quality Improvement | Quality Improvement | Skipped: General QI optimization; does not explicitly test autonomous LLM agents or tool-calling frameworks. | Skipped ⏭ |
| 037   | Modular Medical Dialogue (Bayesian) | Patient-Facing Copilots | Neuro-symbolic agents (LLM + Bayesian Engine) solve the probabilistic diagnostic flaws of monolithic LLMs. | Complete  |
| 038   | MedProbeBench | Benchmarking Framework | Current RAG agents fail at multi-hop synthesis across lengthy, complex clinical guideline documents. | Complete  |
| 039   | Mitigating Racial Bias | Security & Safety Benchmarking | Agentic workflows with "Demographic Auditors" can detect and halt racially biased treatment recommendations. | Complete  |
| 040   | Multimodal Virtual Agent for PTSD | Patient-Facing Copilots | Culturally-adapted 3D avatar agents drastically increase patient engagement for PTSD screening vs text bots. | Complete  |

# Batch 9 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 041   | Neuro-Symbolic Resolution of Recommendation Conflicts | Clinical Decision Support | SMT Solvers (Z3) paired with LLMs solve the multi-morbidity contraindication gap in pure RAG. | Complete  |
| 042   | Persona-Based Requirements Engineering for Explainable Edu Systems | Medical Education Copilots | Multi-agent simulators with specific personas (Patient, Evaluator) provide authentic clinical reasoning practice. | Complete  |
| 043   | Agentic LLMs for Training-Free Neuro-Radiological Image Analysis | Medical Vision | Orchestrating open-source narrow CNNs via a central agent bypasses the need to train massive specialized VLMs. | Complete  |
| 044   | MARCH: Multi-Agent Radiology Clinical Hierarchy | Medical Vision | Hierarchical Multi-Agent systems (Junior/Senior Agents) vastly improve incidental finding recall on full-body CTs. | Complete  |
| 045   | DeepER-Med: Advancing Deep Evidence-Based Research | Research & Systematic Reviews | End-to-end PRISMA state machines (Search, Screen, Extract, Meta-Analysis) reduce review time from months to hours. | Complete  |

# Batch 10 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 046   | RadAgent: Stepwise Interpretation of Chest CT | Medical Vision | Orchestrating deterministic tools (LungSegmenter) solves VLM limitations in fine-grained anatomical localization and measurement. | Complete  |
| 047   | Rethinking Patient Education as Multi-turn Interaction | Patient-Facing Copilots | Health Literacy Agents paired with visual generators (Diffusers) vastly improve patient comprehension vs static pamphlets. | Complete  |
| 048   | Evo-MedAgent: Agents That Remember, Reflect, and Improve | Clinical Decision Support | Integrating an Episodic Memory bank with a Reflection Agent prevents repeated diagnostic errors, enabling continual learning. | Complete  |
| 049   | QuarkMedSearch: Long-Horizon Deep Search Agent | Research & Systematic Reviews | Tree-of-Thoughts (ToT) search agents solve the gap of complex exploratory research where standard single-query RAG fails. | Complete  |
| 050   | VERITAS: Verifiable Epistemic Reasoning for Hypothesis Testing | Medical Vision / Safety | Enforcing active hypothesis testing and targeted zooming on Medical VLMs reduces spatial hallucinations. | Complete  |

# Batch 11 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 051   | Dialectic-Med: Mitigating Diagnostic Hallucinations via Debate | Clinical Decision Support | Counterfactual "Attacker" agents dramatically reduce diagnostic hallucinations in noisy EHR scenarios. | Complete  |
| 052   | Camyla: Scaling Autonomous Research in Med Image Seg | Research Agents | Autonomous "AI Scientists" can generate, test, and iterate PyTorch architectures for medical vision without human intervention. | Complete  |
| 053   | Sense Less, Infer More: Edge Medical Intelligence | Medical Vision (Edge) | Agentic conditional tool-calling (requesting high-res images only when uncertain) enables heavy VLM performance on Raspberry Pi edge devices. | Complete  |
| 054   | BLUEmed: Retrieval-Augmented Multi-Agent Debate | EHR & Operational | Combining RAG FDA guidelines with Multi-Agent debate outperforms zero-shot LLMs in detecting iatrogenic errors. | Complete  |
| 055   | HealthAdminBench: Evaluating Computer-Use Agents | Benchmarking Framework | Current GUI-based Computer-Use Agents fail (34% success) on complex nested healthcare administration tasks. | Complete  |

# Batch 12 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 056   | MedConceal: Clinical Hidden-Concern Reasoning | Benchmarking Framework | Simulates "partial observability" where patient agents conceal symptoms due to stigma, testing the tactfulness of diagnostic agents. | Complete  |
| 057   | MedVR: Annotation-Free Medical Visual Reasoning via RL | Medical Vision | Agentic Reinforcement Learning uses a cropping tool to learn disease localization without expensive bounding box annotations. | Complete  |
| 058   | EMSDialog: Synthetic Multi-person EMS Dialogue | EHR & Operational | Multi-agent roleplay (Paramedic, Patient, Dispatch) successfully generates HIPAA-compliant, chaotic trauma dialogue data. | Complete  |
| 059   | Dual-Memory Self-Learning Diagnostic Agent | Clinical Decision Support | Segregating "Semantic" (fact) and "Episodic" (experience) memory banks cleans up retrieval context for continual learning. | Complete  |
| 060   | LungCURE: Multimodal Real-World Clinical Reasoning | Benchmarking Framework | The first end-to-end benchmark testing combined integration of 3D CTs, genomic sequencing, and EHR notes for oncology. | Complete  |

# Batch 13 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 061   | XrayClaw: Cooperative-Competitive Alignment | Medical Vision | Forcing two agents to debate (one proving, one disproving) cures the "yes-man" hallucination bias in X-ray VLMs. | Complete  |
| 062   | ECG Agentic Cardiovascular Intelligence Edge Review | Research & Systematic Reviews | A comprehensive review highlighting the need to translate raw ECG waveforms into discrete tokens for edge agent reasoning. | Complete  |
| 063   | CARE: Privacy-Compliant Agentic Reasoning | EHR & Operational / Safety | Using local anonymizer agents to scrub PII *before* querying external tools allows agents to safely resolve EHR evidence discordance. | Complete  |
| 064   | SkinGPT-X: Self-Evolving Collaborative Agent | Medical Vision | A Human-in-the-Loop agent queries real dermatologists on low-confidence cases, continuously self-updating its RAG database. | Complete  |
| 065   | Symphony for Medical Coding | EHR & Operational | Uses an Auditor Agent to map assigned ICD-10 codes directly back to exact sentence spans in the EHR, solving the explainability gap for insurance. | Complete  |

# Batch 14 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 066   | Towards a Medical AI Scientist | Research Agents | Outlines the holy grail: a ToT agent that navigates PubMed to formulate novel biological hypotheses and write review papers. | Complete  |
| 067   | Improving Clinical Diagnosis with Counterfactual Reasoning | Clinical Decision Support | Breaking the anchoring bias in diagnostic LLMs by forcing them to explicitly evaluate "what-if" symptom scenarios. | Complete  |
| 068   | MediHive: A Decentralized Agent Collective | Clinical Decision Support | Swaps rigid hierarchical reporting for a decentralized swarm communicating via a shared blackboard, preventing single-point failure hallucinations. | Complete  |
| 069   | AADT: Autonomous Agent-Orchestrated Digital Twins | CDS / Omics | Streaming patient smartwatch data into an agent-managed ODE simulator to run Monte Carlo drug simulations before prescribing. | Complete  |
| 070   | ClinicalAgents: Multi-Agent Dual-Memory Orchestration | Clinical Decision Support | Solves the latency problem of Multi-Agent Systems by using Episodic memory to bypass debate (System 1) for previously seen cases. | Complete  |

# Batch 15 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 071   | Doctorina MedBench: End-to-End Evaluation | Benchmarking Framework | Validates that static QA benchmarks are insufficient; proposes an MDP-based longitudinal patient management simulator. | Complete  |
| 072   | Colon-Bench: Agentic Dense Lesion Annotation | Medical Vision | Automates the creation of massive video datasets by having an agent track and annotate polyps across thousands of frames. | Complete  |
| 073   | OMIND: Knowledge Grounded Mental Health LLMs | Patient-Facing Copilots | Grounds empathetic dialogue agents in specific clinical protocols (CBT) to prevent dangerous hallucinated advice. | Complete  |
| 074   | Rethinking Health Agents: Collaborative Mediators | Methodology & Taxonomy | Argues against "oracle" AI; proposes agents as collaborative mediators that visualize uncertainty to physicians. | Complete  |
| 075   | MedOpenClaw and MedFlowBench | Medical Vision / Benchmarking | Forces agents to use standard radiology software (3D Slicer) to find lesions in full 3D studies, exposing failures of standard VLMs. | Complete  |

# Batch 16 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 076   | Multi-Agent Consistency Verification | Clinical Decision Support | Addresses calibration/safety by using multi-agent variance as a proxy for epistemic uncertainty. | Complete  |
| 077   | CarePilot: Long-Horizon Task Automation | EHR & Administrative | A Computer-Use Agent that navigates legacy EHR GUIs to automate intake and prior authorizations. | Complete  |
| 078   | Dialogue to Question for Guideline Agents | Clinical Text | Synthetically generates multi-turn clinical dialogues from static guidelines to improve RAG retrieval. | Complete  |
| 079   | From Physician Expertise to Clinical Agents | Clinical Decision Support | "Agentic Knowledge Engineering": Agents interview senior doctors and compile their intuition into a local 7B model. | Complete  |
| 080   | Can LLM Agents Generate Real-World Evidence? | Research Agents | Agents autonomously convert clinical hypotheses into OMOP SQL queries, execute PSM in Python, and generate Kaplan-Meier curves. | Complete  |

# Batch 17 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 081   | Cerebra: Multidisciplinary AI Board | Clinical Decision Support | Simulates a tumor/medical board using specialized agents (Neurologist, Geneticist) for multimodal dementia staging. | Complete  |
| 082   | Agentic Automation of BT-RADS Scoring | Medical Vision | Automates glioma tracking by having an agent call coregistration and segmentation tools across longitudinal MRIs. | Complete  |
| 083   | Unified-MAS: Generating Domain-Specific Nodes | Methodology & Taxonomy | Introduces a Meta-Architect agent that automatically designs the optimal multi-agent topology and personas for a new task. | Complete  |
| 084   | Autonomous Robotic Cardiac Ultrasound | Medical Vision / Embodied | A Vision-to-Action agent processes live ultrasound feeds to drive a robotic arm to standard cardiac views. | Complete  |
| 085   | Position: Contestability for Trustworthy AI | Methodology & Taxonomy | Argues that safe multi-agent systems must expose their blackboard reasoning to clinicians and accept manual context overrides. | Complete  |

# Batch 18 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 086   | Caging the Agents: Zero Trust Security | Safety & Ethics | Proposes a Warden architecture to intercept and formally verify agent intent before executing hospital API commands. | Complete  |
| 087   | OpenHospital: Benchmarking Collective Intelligence | Benchmarking Framework | A massive multiplayer environment simulating an entire hospital to test how agent swarms optimize global patient flow. | Complete  |
| 088   | MedPriv-Bench: Privacy-Utility Trade-off | Benchmarking Framework | Quantifies exactly how much diagnostic accuracy is lost when scrubbing PHI from an agent's context window. | Complete  |
| 089   | EviAgent: Evidence-Driven Radiology Reports | Medical Vision | Reduces VLM hallucination by forcing the LLM to write reports strictly conditioned on the boolean output of narrow expert vision models. | Complete  |
| 090   | Six Interventions for Responsible AI Agents | Safety & Ethics | A high-level regulatory framework mapping FDA SaMD guidelines to non-deterministic, autonomous multi-agent systems. | Complete  |

# Batch 19 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 091   | TheraAgent: PET Theranostics | Clinical Decision Support | An agent uses "Self-Evolving Memory" to track the clinical outcomes of its past radiation dosing recommendations, continuously self-improving. | Complete  |
| 092   | When OpenClaw Meets Hospital (Agentic OS) | EHR & Administrative | Introduces an Orchestrator OS that spans multiple hospital software silos, moving data seamlessly via Computer-Use sub-agents. | Complete  |
| 093   | UAV-MARL: Medical Supply Delivery | Embodied / Operational | Decentralized Reinforcement Learning allows physical medical drones to dynamically re-route around weather and shifting triage emergencies. | Complete  |
| 094   | MedMASLab: Orchestration Framework | Benchmarking Framework | Open-source framework that standardizes multi-agent topologies (Swarm, Hierarchy, Ring) to allow rigorous benchmarking across architectures. | Complete  |
| 095   | RexDrug: Multi-Drug Extraction | Research Agents | Uses an Agentic "Think-then-Extract" pipeline to accurately pull complex drug-drug pharmacokinetic interactions from full-text literature. | Complete  |

# Batch 20 (2026-06-30)
| Sl No | Title | Category | Key Takeaway | Status |
|-------|-------|----------|-------------|--------|
| 096   | The Agentic EHR: Proactive Structuring | EHR & Administrative | Swarms of agents monitor unstructured EHR notes in real-time to proactively populate structured JSON/FHIR fields. | Complete  |
| 097   | Benchmarking Limits of RAG in Medicine | Methodology & Taxonomy | Proves that standard RAG fails on temporal multi-hop clinical queries, necessitating iterative Agentic Retrieval. | Complete  |
| 098   | From Copilot to Autopilot (Anesthesia) | Embodied / Treatment | An RL agent controls TCI pumps during surgery based on real-time EEG and hemodynamics, outperforming human anesthesiologists in target depth. | Complete  |
| 099   | Agentic Knowledge Graphs Verification | Research Agents | Agents act as nodes in a knowledge graph, autonomously reviewing new PubMed literature to dynamically update medical claims. | Complete  |
| 100   | The Hundred-Agent Hospital Simulation | EHR / Operational | A generative agent simulation of 100 staff/patients during a pandemic surge, predicting behavioral bottlenecks. | Complete  |

