# DermAgent: A Self-Reflective Agentic System for Dermatological Image Analysis with Multi-Tool Reasoning and Traceable Decision-Making
**Authors:** Yize Liu, Siyuan Yan, Ming Hu, Lie Ju, Xieji Li, Feilong Tang, Wei Feng, Zongyuan Ge
**Year:** 2025 | **Venue:** ArXiv (submitted May 2025, cs.CV)
**ArXiv/DOI:** https://arxiv.org/abs/2605.14403

# Problem Statement
Accurate dermatological diagnosis requires integrating fine-grained visual perception (morphological cues, dermoscopic features) with expert clinical knowledge across a vast, long-tailed taxonomy of skin conditions. While Multimodal Large Language Models (MLLMs) enable interactive medical image analysis, their application in dermatology is critically limited by two fundamental problems: (1) **insufficient domain-specific grounding** — MLLMs rely solely on parametric knowledge acquired during pre-training, with no mechanism to retrieve or verify predictions against external clinical references such as diagnostic guidelines or curated case repositories; and (2) **hallucinations** — morphologically similar lesions demand precise differential diagnosis, and MLLMs frequently generate plausible but incorrect diagnoses. The paper addresses the need for a system that can orchestrate multiple specialist tools, provide stepwise traceable reasoning, ground predictions in external evidence, and self-correct through automated conflict detection.

# Key Contributions
1. **DermAgent framework**: A collaborative multi-tool agent that orchestrates seven specialist vision and language tools within a Plan–Execute–Reflect loop, providing stepwise, inspectable diagnostic reasoning for dermatological image analysis.
2. **Dual-modality retrieval module**: Combines Case-Based Image Retrieval (Case RAG) over 413,210 diagnosed cases with Guideline-Grounded Text Retrieval (Guideline RAG) over 3,199 clinical guideline chunks, grounding predictions in traceable external knowledge.
3. **Deterministic Critic module**: Implements three gate conditions — confidence, coverage, and conflict checks — that trigger targeted self-correction to suppress hallucinations, rather than relying on the LLM's own uncertain judgment.
4. **Extensive empirical validation**: Evaluations on five dermatology benchmarks demonstrate consistent improvements over MLLM and agent baselines across disease diagnosis, concept annotation, and clinical captioning tasks.

# Methodology / Architecture

# Overall Framework: Plan–Execute–Reflect Loop
DermAgent formulates dermatological diagnosis as a **multi-step evidence reasoning problem**. Given a query tuple (I, q) — a skin image I and a clinical question q — the goal is to generate a response R accompanied by a **traceable evidence chain** ℰ, where each element eᵢ ∈ ℰ represents a verified finding from a specialized tool.

The architecture is modeled as a **directed cyclic graph** with three nodes:
- **Chatbot** (LLM Controller, powered by GPT-4o)
- **Tools** (seven specialist modules)
- **Critic** (deterministic post-hoc auditor)

The system maintains:
- A dynamic evidence chain ℰ (accumulating verified findings)
- A feedback memory  (from the Critic)
- A retry counter k (bounded by k_max)

**Workflow steps:**
1. **Task Analysis & Planning**: The Chatbot node performs task analysis to determine the problem scope . It synthesizes current evidence ℰ and prior critic feedback  to generate a coherent plan ₖ = {(tᵢ, θᵢ)} — specific tool calls and their parameters.
2. **Execute**: The Tools node executes planned calls in parallel, collecting results eᵢ and updating the evidence chain: ℰ ← ℰ ∪ {(tᵢ, θᵢ, eᵢ)}.
3. **Reflect (Critic)**: The Critic evaluates the updated ℰ against three logic checks. If deficiencies are found and k < k_max, it generates directional feedback  and routes flow back to the Chatbot for refinement. Otherwise, the agent proceeds to Synthesis.

# Specialist Tool Modules (Seven Tools)

**Visual Perception Tools (4 tools):**

1. **PanDerm Classifier**: Uses DermLIP for zero-shot skin disease classification. Given image I and candidate label set  = {c₁, …, cₙ}, it computes cosine similarity between image and text embeddings, returning ranked predictions with calibrated confidence scores.

2. **MAKE Concept Annotator**: Based on the MAKE framework, detects dermoscopic concepts (e.g., pigment network, streaks) via zero-shot CLIP-based scoring. Outputs structured annotation set  ⊆ ℱ from a predefined feature set.

3. **DermoGPT (Dermatology VQA)**: A dermatology-specialized vision-language model for free-form visual question answering, providing morphological description and diagnostic reasoning.

4. **Qwen3-VL (General VQA)**: Qwen3-VL-8B-Instruct is employed for complementary general VQA tasks.

**Knowledge Retrieval Tools (2 tools):**

5. **Case-Based Image Retrieval (Case RAG)**: Encodes query image with the DermLIP encoder (shared with PanDerm) into a 512-dimensional embedding and performs cosine similarity search over a vector database of **413,210 clinically diagnosed cases** from Derm1M. Each retrieved entry carries a disease label, hierarchical diagnostic category, and clinical description.

6. **Guideline-Grounded Text Retrieval (Guideline RAG)**: Queries a curated knowledge base of **3,199 document chunks** from DermNet (2,751 chunks) and Mayo Clinic (448 chunks). Retrieval follows a **four-stage hybrid pipeline**:
   - Domain-specific stop-word filtering
   - Semantic vector search via Qwen3-Embedding-8B (4,096-dim embeddings)
   - Full-text keyword matching over indexed fields
   - Reciprocal Rank Fusion (RRF, k=60) + re-ranking by Qwen3-Reranker-0.6B (cross-encoder)

**Structural Knowledge Tool (1 tool):**

7. **Ontology Tool**: A hierarchical knowledge graph of skin disease taxonomy with fuzzy name matching. Supports query modes: hierarchy, children, siblings, search.

# Critic-Driven Reflection (Three Gates)

The deterministic Critic module evaluates the evidence chain ℰ against three gate conditions after each execution round:

1. **Confidence Check (f_conf)**: Flags low-confidence outputs — PanDerm predictions with cosine similarity below 90%, or RAG retrievals with similarity below 80%. A flag is raised only when an uninvoked actionable tool remains available, ensuring retries introduce genuinely new evidence.

2. **Coverage Check (f_cov)**: Verifies that task-critical specialist tools have been invoked for the given query type. Prevents the LLM controller from short-circuiting to a VQA model alone without engaging task-specific modules (e.g., PanDerm for diagnosis, MAKE for concept annotation).

3. **Conflict Check (f_con)**: Compares top-ranked predictions across visual specialist tools (PanDerm and Case RAG) and flags unresolved disagreements. Conflicts already addressed through prior Guideline RAG retrieval are accepted, preventing unproductive retry loops.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| HAM10000 | Dermatoscopic images for skin disease classification | 7 diseases, 642 images (evaluation subset) |
| SNU | Fine-grained skin disease classification | 134 disease classes, 500 images |
| Derm7pt | Seven-point checklist dermoscopic concept annotation | 7 dermoscopic concepts, 100 images |
| SkinCon | Dense clinical concept annotation for skin diseases | 32 clinical concepts, 100 images |
| SkinCAP | Multimodal dermatology dataset with clinical captions | 100 images (evaluation subset) |
| Derm1M (for Case RAG) | Million-scale vision-language dermatology dataset | 413,210 diagnosed cases in vector database |
| DermNet + Mayo Clinic (for Guideline RAG) | Clinical guideline text chunks | 3,199 chunks (2,751 DermNet + 448 Mayo Clinic) |

# Key Results & Metrics

# Main Results (Table 1)
| Task/Benchmark | Metric | DermAgent | Previous Best | Improvement |
|----------------|--------|-----------|---------------|-------------|
| HAM10000 (Classification) | Accuracy | 61.83% | MedAgent-Pro: 57.63% | +4.20% |
| HAM10000 (Classification) | Accuracy | 61.83% | Hulu-Med-7B (best MLLM): 52.65% | +9.18% |
| SNU (Fine-grained, 134 classes) | Accuracy | 32.60% | GPT-4o: 15.00% | +17.60% (>2× improvement) |
| Derm7pt (Concept Annotation) | F1-Macro | 65.06% | DermoGPT-RL: 56.86% | +8.20% |
| SkinCon (Concept Annotation) | F1-Macro | 32.95% | GPT-4o: 29.56% | +3.39% |
| SkinCAP (Clinical Captioning) | ROUGE-L | 19.48% | GPT-4o: 16.33% | +3.15% |

# Ablation Study (Table 2 — Leave-One-Out on SkinCAP)
| Removed Component | ROUGE-L Drop (Δ) | Role |
|-------------------|-------------------|------|
| Critic mechanism | −2.21% | Largest contributor; consistency checking & targeted retries |
| Case RAG | −1.47% | Grounds clinical impression in real diagnostic records |
| Guideline RAG | −0.99% | Supplies precise medical terminology |
| DermoGPT | −0.55% | Morphological descriptions |
| PanDerm | −0.51% | Diagnostic hypotheses |
| MAKE | −0.48% | Structured dermoscopic features |
| Ontology Tool | Minimal | Limited impact on captioning task |

# Baselines Compared
**MLLMs:** LLaVA-Med, HuatuoGPT-Vision, Hulu-Med-7B, DermoGPT, DermoGPT-RL, MM-Skin, Qwen3-VL, GPT-4o, GPT-5.2
**Agent Systems:** MDAgents, MedAgent-Pro

# Limitations & Future Work
The paper does not explicitly enumerate a comprehensive limitations section. However, from the methodology and results, the following can be inferred:
- The system depends on GPT-4o as the LLM controller, making it reliant on a proprietary, closed-source model.
- Evaluation uses relatively small evaluation subsets (100–642 images) with stratified sampling and oversampling of underrepresented classes.
- The retry mechanism is bounded by k_max, though the specific value of k_max is not discussed in terms of computational overhead.
- Performance on the fine-grained SNU benchmark (134 classes) remains at 32.60% accuracy, suggesting substantial room for improvement on highly granular classification tasks.
- The approach is specific to dermatology — extending to other medical imaging domains would require new specialist tools and knowledge bases.
- The deterministic Critic uses fixed thresholds (90% for PanDerm, 80% for RAG) which may not generalize optimally across different data distributions.

# Relevance to Our Survey
DermAgent is a highly relevant example of **AI Agents in Healthcare** specifically in the domain of **dermatology/clinical diagnosis**. It belongs to the category of:

- **Multi-tool agentic systems for medical image analysis**: Demonstrates how an LLM controller can orchestrate multiple domain-specific visual and textual tools in a structured reasoning loop.
- **RAG-augmented clinical reasoning**: Showcases dual-modality retrieval (image-based + text-based) to ground diagnostic predictions in external evidence, addressing the hallucination problem.
- **Self-reflective/self-correcting agents**: The deterministic Critic with three gates (confidence, coverage, conflict) represents a principled approach to automated quality control in medical AI.
- **Modalities addressed**: Dermatoscopic images (clinical and dermoscopic skin images), clinical text (guidelines, case descriptions), and structured dermoscopic concepts.

This paper is a strong exemplar of how agentic AI can bridge the gap between pattern recognition and evidence-based clinical reasoning in a specific medical specialty.

# Key Figures & Tables Summary

# Figure 1: DermAgent Architecture Diagram
Shows the complete system architecture as a directed cyclic graph with three nodes (Chatbot, Tools, Critic). The workflow flows from the input query (I, q) through Task Analysis → Planning → Tool Execution (seven tools shown in parallel) → Critic Evaluation → either loop back (if gates fail) or proceed to Synthesis. The seven tools are visually organized into two groups: Visual Perception (PanDerm, MAKE, DermoGPT, Qwen3-VL) and Knowledge Retrieval (Case RAG, Guideline RAG, Ontology Tool). The Critic node shows the three gates: f_conf (confidence), f_cov (coverage), f_con (conflict).

# Figure 2: Qualitative Case Study — Granuloma Annulare
Illustrates a representative diagnostic case where GPT-4o misidentifies a lesion as tinea corporis, while DermAgent correctly diagnoses granuloma annulare through its self-correction mechanism. The flow shows:
1. DermoGPT accurately describes morphological features but hallucinates a diagnosis of eczema
2. PanDerm similarly predicts eczema
3. Case RAG retrieves four highly similar cases of granuloma annulare
4. The f_con (conflict) gate detects the discrepancy and triggers retry with image reinjection
5. The LLM controller recognizes raised borders and central clearing conflict with eczema criteria
6. A refined PanDerm query confirms granuloma annulare
7. Guideline RAG retrieves appropriate clinical guidelines for the final synthesis

# Table 1: Main Performance Comparison
Comprehensive comparison across five benchmarks and three task types (classification, concept annotation, captioning) against 10+ baselines including both MLLMs and agent systems.

# Table 2: Ablation Study (Leave-One-Out)
Shows the contribution of each component on the SkinCAP captioning task, with the Critic mechanism providing the largest improvement (+2.21%), followed by Case RAG (+1.47%) and Guideline RAG (+0.99%).

# Algorithm 1: DermAgent Execution Flow
Formalizes the Plan–Execute–Reflect loop pseudocode, showing the initialization of ℰ, , and k, the iterative planning-execution cycle, the three critic gate checks, and the final synthesis step.
