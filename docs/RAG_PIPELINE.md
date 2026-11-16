# Medical RAG: Design, Known Failure Modes, and the DDx-RAG Pipeline

> **Historical document.** This describes DDx-RAG v1, our first design. Its results are in `IMPLEMENTATION_LOG.md`; its code was removed after the agent replaced it (it is in the git history before the commit "Simplify the code base"). For the current system read `docs/PROJECT_WALKTHROUGH.md`.

This document explains **what retrieval system the agent uses, why, and how it is evaluated**.
Code: `src/clinical_agent/retrieval/` (search engine), `src/clinical_agent/rag/` (QA methods),
`src/clinical_agent/evaluation/mirage.py` (benchmark).

---

## 1. Why the old RAG had to go

The v2 "RAG" (`tools/medical_knowledge.py`) was 9 hand-written disease summaries searched with
TF-IDF. Each chunk repeated the disease name and its keywords twice, and a hand-made synonym map
pointed words like `dimer` straight at `pulmonary_embolism`. The 8 benchmark cases were written
with the corpus's own keywords, so the retrieval was **circular**: 100% on those cases says
nothing about real performance.

The rest of the audit found the following problems. All are fixed in this repo.

| # | Problem | Fix |
|---|---|---|
| B1 | DenseNet outputs went through the sigmoid **twice**. TorchXRayVision already returns calibrated scores, so every value landed in [0.50, 0.73]. | `models/densenet_cxr.py` uses the outputs as they come. |
| B2 | The calibrated operating thresholds were ignored in favour of a hand-picked 0.65 cut-off. | ≥ 0.5 now means "positive at the model's own operating point". |
| B5 | Toy, circular retrieval corpus. | Real corpora (§3) and a real retriever. |
| B6 | Metrics overstated results ("grounding" meant only "evidence exists"; diagnosis matching was by substring). | Removed. MIRAGE uses exact option matching. |
| B7 | A missing disposition silently defaulted to `ADMIT`. | Now defaults to `ESCALATE_TO_PHYSICIAN`. |
| B8 | An API key was committed in `settings.json`. | File deleted. **The key must be rotated.** |
| B3, B4 | MedSAM box-geometry CTR; the "sequential" workup didn't actually hide data. | Agent phase (next). |

---

## 2. What goes wrong in medical RAG, and what we do about each

Medical RAG often makes answers **worse**:
- An 18-expert study (arXiv 2511.06738) found only **22%** of the top-16 retrieved passages were relevant, and standard RAG lowered factuality.
- RAG² (NAACL 2025) found that with Llama-3-8B, vanilla MedRAG *lost* accuracy on MedQA versus no retrieval (57.7 → 56.4).

So every design choice below targets a known failure mode.

| Failure mode | Why it happens | Our fix | Where |
|---|---|---|---|
| **Long-query dilution** | A 177-token MedQA vignette is a poor query. MedCPT's query encoder was trained on short PubMed searches. | Break the question into short, intent-specific queries (max 20 words). | `DDxRAG.plan` |
| **Dense misses exact terms** | Embeddings blur rare tokens: drug names, eponyms, "anti-dsDNA". | Hybrid search: BM25 alongside MedCPT. | `Retriever.candidates` |
| **Sparse misses paraphrase** | "Short of breath" vs "dyspnoea". | The dense side catches it; the plan writes queries in textbook vocabulary. | `PLAN_USER` prompt |
| **Score-scale mismatch** | BM25 scores and cosine similarities aren't comparable. | Reciprocal Rank Fusion (rank-based). | `reciprocal_rank_fusion` |
| **Irrelevant top-k** | Bi-encoders compress the whole text into one vector. | MedCPT cross-encoder rerank over the fused pool; optional `min_score` cut-off. | `Retriever._rerank` |
| **Chunk too small** ("it is contraindicated in…") | Snippets lose their subject when cut. | Title/section prefix on every snippet, plus **neighbour expansion** after ranking: the tail of the previous and the head of the next paragraph of the *same* section. Retrieve small, read bigger. | `Retriever.expand` |
| **Chunk too broad** | Useful sentence diluted by the rest. | Rank at snippet level; the VERIFY step must cite a specific evidence id per claim. | `VERIFY_USER` |
| **Redundancy** | 18 textbooks repeat each other. | Near-duplicate removal (word-set Jaccard ≥ 0.7). | `is_near_duplicate` |
| **Confirmation bias / anchoring** | Similarity search only finds text that *agrees* with the query. | SUPPORT / DISCRIMINATE / **REFUTE** queries, plus **per-query quotas**, so evidence against the leading hypothesis always gets into the context. | `search_many` step 4a |
| **Model obeys irrelevant text** | LLMs over-trust retrieved context. | Retrieval is used to *check* the answer, not only to supply context. The closed-book majority can only be overturned by citing evidence. | `DECIDE_USER`, `VOTE_RULE` |
| **Hallucinated citations** | The model cites evidence that doesn't exist. | Points citing a non-existent evidence id are dropped and counted. | `verify_hypotheses` |
| **Lost in the middle** | Long contexts bury key text. | Budget of about 10 snippets, ordered by score. | `budget` |

---

## 3. Knowledge sources and indexes

| Corpus | Size | Source | Why |
|---|---|---|---|
| Textbooks | 18 books, 125.8k snippets | HF `MedRAG/textbooks` | Same corpus as MedRAG and i-MedRAG |
| StatPearls | 9.3k articles, ~301k snippets | NCBI LitArch, chunked with a port of MedRAG's own chunker | Clinical decision support; same chunks as the baselines |

| Index | Technology | Notes |
|---|---|---|
| Sparse | BM25 via `bm25s` (sparse matrices, English stemming and stopwords) | |
| Dense | MedCPT article encoder → **FAISS `IndexFlatIP`** (exact inner product) | ~427k × 768 fp32 ≈ 1.3 GB RAM. At this size exact search takes milliseconds, so approximate indexes (HNSW/IVF) would cost accuracy for no gain. A production system would use Qdrant or pgvector for updates, metadata filters (source, guideline date) and native hybrid search. |
| Reranker | MedCPT cross-encoder | |

---

## 4. DDx-RAG: the pipeline (`rag/methods.py: DDxRAG`)

**Idea in one line:** answer like a careful clinician. Form a differential **before** seeing the options, look for evidence that would **support, separate or refute** each hypothesis, fact-check the hypotheses against that evidence with citations, and let the evidence overturn the model's own instinct only when it explicitly justifies the change.

```
question + options
   │
   ├─ 0. VOTE      5 closed-book CoT samples (1 request, n=5)      -> agreement = top vote share
   │
   ├─ 1. PLAN      question ONLY (options hidden)                  -> findings, 2-4 hypotheses,
   │                                                                   SUPPORT / DISCRIMINATE / REFUTE queries
   ├─ 2. GATE      agreement >= 0.8 ? light path : full path
   │                 light: only queries about the leading hypothesis, budget 5, no VERIFY
   │                 full : every query, budget 10, VERIFY
   │
   ├─ 3. RETRIEVE  per query: BM25 ∥ MedCPT -> RRF -> cross-encoder rerank
   │               across queries: quota (2 per query) -> fill by score -> dedupe -> neighbour expansion
   │               + the raw question as a safety-net query
   │
   ├─ 4. VERIFY    per hypothesis: points FOR / AGAINST, each citing an evidence id; uncited points dropped
   │
   └─ 5. DECIDE    options visible; evidence + fact-check + closed-book tallies;
                   a different answer from the majority must cite justifying evidence
                   -> answer (falls back to the majority vote if unparseable)
```

**Models:**
- Llama-3.1-8B-Instruct (bf16, the same precision as the published baselines) served by vLLM for steps 0, 1, 4 and 5.
- MedCPT query encoder and cross-encoder plus BM25 for step 3.

**Cost:** 3 calls on the light path, 4 on the full path.

**Benchmark compliance:** MIRAGE's *question-only retrieval* rule says options must not be used for retrieval. Every query DDx-RAG retrieves with comes from the option-free PLAN step. The options appear only in the VOTE and DECIDE steps, and neither of those retrieves anything.

### What is borrowed, and what is ours

| Component | Origin |
|---|---|
| Hybrid BM25 + dense with RRF | MIRAGE / MedRAG ablations; Cormack et al. 2009 |
| MedCPT retriever and cross-encoder | Jin et al., Bioinformatics 2023 |
| Rationale/LLM-written queries | RAG² (Sohn et al., NAACL 2025); i-MedRAG (Xiong et al., PSB 2025) |
| Self-consistency votes as a confidence signal | Wang et al., ICLR 2023 |
| Support/refute evidence labels | Fact verification (FEVER) |
| **Intent-typed queries from an option-free differential, with quotas that guarantee evidence for competing hypotheses and against the leading one** | Ours. Related idea: discriminative features in MedRAG-KG (WWW 2025), which needs a pre-built knowledge graph. |
| **Uncertainty-gated verification depth** (check more when the model is less sure) | Ours |
| **Evidence-justified adjudication** between closed-book and retrieved answers | Ours. Motivated by the "RAG often hurts" findings. |

---

## 5. How it is evaluated

**Benchmark:** MIRAGE (MedQA-US 1,273, MMLU-Med 1,089; optionally MedMCQA 4,183), zero-shot.
- **Corpus:** Textbooks + StatPearls.
- **LLM:** Llama-3.1-8B-Instruct.
- **Tuning:** thresholds are tuned only on 300 questions from the MedQA *training* set (`--dataset medqa_dev`), never on the test set.

**Methods, all rerun by us with the same LLM, corpus and server:**

| Method | What it is |
|---|---|
| `cot` | MedRAG's closed-book prompt |
| `cot_sc` | Majority of 5 |
| `medrag` | Vanilla MIRAGE RAG, MedCPT top-32 |
| `rationale_rag` | RAG²-style rationale query (without its trained filter) |
| `imedrag` | Iterative follow-up queries; 3 rounds × 3 queries, k=16. **Sees the options while writing queries, as in the original, so it is not QOR-compliant.** |
| `ddx_rag` | Ours |

**Published reference points, same corpus family, Llama-3.1-8B** (i-MedRAG paper, Table 2):

| Method | MedQA | MMLU-Med |
|---|---|---|
| CoT | 64.73 | 77.23 |
| MedRAG | 66.54 | 78.05 |
| i-MedRAG | 75.02 | 79.61 |

Our numbers come from our own reruns (same model and precision, but different serving stack and prompt plumbing). **The claims we make are about differences within our setup**, tested with paired McNemar tests.

**Reported for every method:**
- Accuracy with a Wilson 95% confidence interval.
- A paired McNemar test against `cot`.
- **Fixed/broke counts** (how often retrieval rescued vs. ruined a closed-book answer).
- Unparsed answers.
- LLM calls and tokens per question.

**Ablations of DDx-RAG** (`--set ... --tag ...`):

| Tag | Switch | Hypothesis tested |
|---|---|---|
| `no_refute` | `intents=SUPPORT,DISCRIMINATE` | Refuting evidence corrects anchoring |
| `support_only` | `intents=SUPPORT` | Intent diversity matters |
| `no_verify` | `verify=false` | The fact-check table helps the decision |
| `no_votes` | `show_votes=false` | Adjudication against the closed-book answer reduces RAG harm |
| `no_gate` | `gate_threshold=1.01` | Gating saves compute without losing accuracy |
| `bm25` / `dense` | `mode=bm25` / `mode=dense` | Hybrid beats either alone |
| `no_expand` | `expand=false` | Neighbour expansion helps |

**Robustness tests:**
- `--shuffle-options 1`: is the answer stable when options are reordered?
- `--noise 3`: three random snippets injected into every retrieval.

---

## 6. Results

*To be filled from `results/` after the server runs.*
