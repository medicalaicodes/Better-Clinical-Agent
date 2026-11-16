# Better-Clinical-Agent — the complete walkthrough

This is the whole project told as one story: what problem we went after, what we built, how every
important file works (function by function, often line by line), what the experiments showed, and
how to answer the questions an interviewer is likely to ask.

Read it in order the first time. Sections 4 and 5 are the "code tour": keep the code open next to them.

---

## 0. The two-minute version

> I built a medical retrieval-augmented system on top of an open 8B model (Llama-3.1-8B). It searches
> about 427,000 passages from medical textbooks and StatPearls using hybrid retrieval: keyword search
> (BM25) plus a biomedical embedding model (MedCPT), fused and then re-ranked by a cross-encoder.
>
> I then benchmarked it honestly. I reran five published methods on the same model and data (MIRAGE
> benchmark, MedQA and MMLU-Med), with confidence intervals and significance tests. The key finding
> was that retrieval *hurts* a model on questions it already knows: when the model's 5 sampled answers
> agree, it is right 89% of the time, and every RAG method pulled that number down. Retrieval only
> helped on questions where the model was unsure.
>
> That led to the final design: an agent that uses the library to **verify claims, not to decide**.
> It forms hypotheses without seeing answer options, checks specific claims for every hypothesis in
> parallel, may only rule a hypothesis out by citing real evidence, and is challenged by a critic
> before it answers. It's tested end to end and every step leaves an auditable trace.

---

## 1. The story, in order

### 1.1 Where we started
The repository already had a "clinical agent" (a LangGraph state machine with tools for chest X-rays,
lab values and image segmentation). Its "RAG" was a tiny TF-IDF index over 9 guideline documents,
and there had never been an evaluation. Several bugs were found in an audit, e.g. the chest X-ray
model applied a sigmoid twice, so probabilities were wrong.

### 1.2 Phase 1 — build real retrieval and a fair benchmark
1. **Corpus.** We used the same medical corpora as the MedRAG / MIRAGE paper (Xiong et al., 2024),
   so our numbers are comparable to published ones:
   - Textbooks: 18 USMLE textbooks, ~125.8k snippets.
   - StatPearls: ~9.3k clinical articles, ~301k snippets.
2. **Indexes.** BM25 for exact words, MedCPT embeddings + FAISS for meaning, and the MedCPT
   cross-encoder to re-rank. All three are off-the-shelf; nothing is trained.
3. **Methods.** Five baselines from the literature (CoT, self-consistency, MedRAG, RAG²-style,
   i-MedRAG) and our first idea, DDx-RAG.
4. **Harness.** A resumable runner, per-question traces, Wilson confidence intervals, McNemar tests,
   cost per question (calls and tokens).
5. **Server.** A 40 GB A100 running vLLM (an LLM server with an OpenAI-compatible API).

### 1.3 What the benchmark showed (Llama-3.1-8B, bf16)

| Method | LLM calls | MedQA (1,273 q) | MMLU-Med (1,089 q) |
|---|---|---|---|
| CoT (one answer) | 1 | 70.7 | 78.3 |
| **Self-consistency (majority of 5)** | 1 request, 5 samples | **73.7** | **81.9** |
| MedRAG (standard RAG) | 1 | 67.8 (worse than no RAG, p = 0.038) | 77.0 |
| RAG²-style | 2 | 72.1 | — |
| i-MedRAG | ~10.6, 28.5k tokens | 71.9 (paper: 75.0) | — |
| DDx-RAG v1 (ours) | ~3.2 | 71.0 | 81.7 |

- Our CoT baseline (70.7) is stronger than the published one (64.7), so nothing was "made weak" to look good.
- **The key finding:** split the MedQA questions by how much the 5 votes agree.
  - All 5 agree (54% of questions): the majority is right 89.2%. Every RAG method lowered this
    (MedRAG 80.6, RAG² 84.1, i-MedRAG 86.0).
  - 3 or fewer agree: RAG methods beat the majority by 3–13 points.
  - In words: **retrieval hurts when the model is sure and helps when it isn't.**

### 1.4 Ideas we tried, and why they failed (each one measured)
1. **DDx-RAG v1** — the model reads the evidence and may overrule its own votes. It overruled 193
   times: 54 good changes, 91 bad ones. Reading on-topic but misleading passages pulled it off
   correct answers. Its "refute" and "verify" components made no measurable difference (ablations).
2. **Book-scoring probe** — let the cross-encoder score each answer option against textbook
   passages (no LLM reading). Textbook score alone picked the right option 33% of the time (chance is
   25%); between the top two candidates it was a coin flip (51.7%). A relevance model can't tell two
   plausible diagnoses apart — that needs reasoning.
3. **Options-free answering** costs about 31 points on MedQA with an 8B model (46% vs 77% on the same
   questions), because MedQA is written around its options ("which of the following...").

### 1.5 The final design — DxAgent
From those lessons:
- The model must not *read a pile of passages and decide*. It should **check specific claims**.
- Every hypothesis gets checked **at the same time**, so the favourite gets no head start.
- Ruling something out needs a **citation** (a card the library returned, or a finding of the case).
- A **critic** with a fresh context tries to find what was missed.
- Options are hidden during the whole work-up; with `final=select` they appear only for the last pick.

### 1.6 Making it work
The first agent run looked terrible (61.7%), but the traces showed why: about half of the model's JSON
replies were malformed, so the loop barely ran. Fixes: vLLM's JSON mode (constrained decoding), a
critic that may only block if it names a different answer, and clearer prompts. The check run on 20
practice questions then ran cleanly: 0 errors, JSON failures in 1 of 20 questions, 70% correct
(n = 20 — a smoke test, not a result).

---

## 2. The big picture: how the pieces connect

```
                          ┌────────────── offline, once ──────────────┐
  Textbooks + StatPearls ─► corpus.py (download, chunk) ─► build_index.py
                                                          ├─ BM25 index (bm25s)
                                                          └─ MedCPT embeddings (FAISS)

                          ┌────────────── per question ───────────────┐
  question ─► method (rag/methods.py or rag/agent.py)
                 │  uses
                 ├─ llm.py  ──HTTP──► vLLM server (Llama-3.1-8B on the A100)
                 └─ search.py (Retriever) / evidence_tool.py
                        ├─ BM25 + MedCPT dense ─► RRF fusion ─► cross-encoder re-rank
                        └─ quotas, de-duplication, neighbour expansion / sentence cards

  run_mirage.py runs a method over a dataset in parallel threads, saves one JSON line per question
  (answer + full trace + cost), and prints a comparison table with statistics.
```

Folder map (only what matters):

| Path | What it is |
|---|---|
| `src/clinical_agent/config.py` | Settings from `.env` (server URL, model name, data folder) |
| `src/clinical_agent/llm.py` | The one LLM client; JSON / answer parsing; thread pool |
| `src/clinical_agent/retrieval/` | Corpus, indexes, hybrid search, the agent's evidence tool |
| `src/clinical_agent/rag/` | Prompts, baselines, DDx-RAG v1, DxAgent |
| `src/clinical_agent/evaluation/` | MIRAGE loader/runner, statistics |
| `scripts/` | Build indexes, run benchmarks, analyse results, serve vLLM |
| `tests/` | 33 CPU-only tests with a fake LLM and a toy corpus |
| `src/clinical_agent/agent.py`, `clinical_prompts.py`, `tools/`, `schemas.py` | The clinical agent: a LangGraph state machine over imaging, lab and library tools |

---

## 3. Key ideas in plain words (read once, then refer back)

- **RAG (retrieval-augmented generation):** find relevant text, give it to the LLM with the question.
- **Chunk / snippet:** a short piece of a document (here a paragraph or so) — the unit we search.
- **BM25:** classic keyword search. Scores a passage by how many of the query's words it has, giving
  rare words more weight and not over-rewarding long passages. Great for exact terms ("D-dimer").
- **Dense retrieval / embeddings:** turn text into a vector of 768 numbers such that similar
  meanings land close together. Finds paraphrases BM25 misses ("shortness of breath" ≈ "dyspnea").
- **Bi-encoder vs cross-encoder:** a bi-encoder embeds the query and the passage *separately*
  (fast, can pre-compute 427k passages). A cross-encoder reads them *together* (slow, more precise),
  so we use it only to re-rank a short list.
- **MedCPT:** NCBI's biomedical retriever, trained on 255M real PubMed search logs. It comes as a
  query encoder, an article encoder and a cross-encoder.
- **FAISS IndexFlatIP:** exact nearest-neighbour search by inner product. With 427k vectors it takes
  milliseconds, so approximate indexes (HNSW, IVF) wouldn't gain anything worth the accuracy loss.
- **RRF (reciprocal rank fusion):** merge two ranked lists by summing `1 / (60 + rank)` from each.
  Uses ranks, not raw scores, so BM25 and cosine scores never need to be on the same scale.
- **Self-consistency:** ask the same question several times with some randomness and take the
  majority. The agreement level is a free confidence signal.
- **vLLM:** a fast LLM server. Batches many requests on the GPU and caches shared prompt prefixes.
- **Constrained decoding / JSON mode:** the server only allows tokens that keep the output valid JSON.
- **McNemar test:** compares two methods on the *same* questions, using only the questions where
  they disagree. **Wilson interval:** a confidence interval for a percentage that behaves well near 0/100%.

---

## 4. Code tour — the shared foundation

### 4.1 `config.py` — settings
A plain `Settings` class. `read_env_file()` reads `KEY=VALUE` lines from `.env`; each setting
comes from a real environment variable first, then `.env`, then the default:
`LLM_BASE_URL` (default `http://localhost:8000/v1`), `LLM_MODEL` (`llama31-8b`), `LLM_TEMPERATURE`,
`LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `DATA_DIR`, `TORCH_DEVICE`. Two folders are derived:
`corpus_dir = data_dir/corpus`, `index_dir = data_dir/index`. `get_settings()` reads them once. **Why:** no paths or server URLs are
hard-coded; the same code runs on a laptop (Ollama) or the A100 (vLLM) by changing `.env`.

### 4.2 `llm.py` — the single door to the language model

**`class LLM`**
- `__init__`: builds an `openai.OpenAI` client pointed at our own server (`base_url`). vLLM speaks
  the OpenAI API, so the official SDK works unchanged. `max_retries=3` lets the SDK retry timeouts.
  `self._usage = threading.local()` gives every thread its own call/token counters.
- `reset_usage()` / `usage()` / `_count(response)`: per-thread cost accounting. The benchmark answers
  each question in one thread, so after a question, `usage()` is exactly that question's cost.
- `chat(prompt, system, temperature, max_tokens, n, seed, on_token, json_mode)`:
  1. builds `messages` (optional system message + the user prompt);
  2. `request = {model, messages, temperature, max_tokens}`; adds `seed` if given;
  3. `if json_mode: request["response_format"] = {"type": "json_object"}` — constrained decoding;
  4. streaming branch (`on_token`) is for the interactive demo only;
  5. normal branch: `create(n=n, ...)` asks for n samples **in one request** (vLLM shares the prompt
     computation, so 5 samples cost far less than 5 calls);
  6. a `while len(texts) < n` loop tops up for servers that ignore `n` (Ollama, llama.cpp); each
     extra call gets `seed + len(texts)` — with the same seed every "independent" sample came back
     identical (a bug we found in testing).

**`parse_json(text)`** — robust JSON extraction: strips `<think>...</think>` (reasoning models),
looks inside a ```json fence if present, tries the whole text and then the span from the first `{`
to the last `}`, each with and without trailing commas. Returns a dict or `None`.

**`is_placeholder(text)`** — true for template echoes like `"<query>"` or `"..."` (no 2+ letter
words). Small models sometimes copy the example JSON instead of filling it in.

**`parse_choice(text, valid)`** — gets the option letter. First from JSON keys `answer_choice` /
`answer` / `choice` with a strict regex that accepts `B`, `(B)`, `B.`, `B. Heart` but rejects
`A/B/C/...` (the model echoing the format hint) and words like `BETA`. Then free-text fallbacks
(`"answer is (B)"`), each with `(?!/)` to avoid the same echo.

(The thread pool that runs many questions at once lives in `evaluation/mirage.py: run_method`.
Threads are enough because each one mostly waits for the server, and vLLM batches all waiting
requests together on the GPU.)

### 4.3 `retrieval/corpus.py` — getting and cutting the documents
- `Snippet(id, title, content, source)`; `.text` = title + content (what we index).
- `CorpusStore` keeps four parallel lists (`ids`, `titles`, `contents`, `sources`) instead of
  427k objects — less memory. **A snippet's row number is its id in every index**, so the row order
  must never change; that's why the store is saved as one consolidated `corpus.jsonl`.
- `download_textbooks` uses `huggingface_hub.snapshot_download` of `MedRAG/textbooks` (`chunk/*`).
- `download_statpearls` downloads NCBI's ~2 GB tar to a `.part` file, renames on success, and
  extracts with `filter="data"` (blocks path tricks in archives).
- `chunk_statpearls_article(path)` is a port of MedRAG's own chunker, so our snippet boundaries match
  the published baselines: paragraphs under 200 characters are glued onto the previous snippet if the
  result stays under 1,000; lists are handled; subtitles become part of the title
  ("Pulmonary Embolism -- Evaluation -- Imaging").

### 4.4 `retrieval/indexes.py` — the three search engines
- **`BM25Index`** wraps `bm25s` (sparse-matrix BM25, ~100× faster than `rank_bm25`). Text is
  lower-cased, English stop-words removed, words stemmed with PyStemmer ("embolism" → "embol").
  `search()` tokenises queries with `return_ids=False` so bm25s maps words onto its own vocabulary
  (unknown words are ignored), skips queries that were only stop-words, and drops hits with score 0.
- **`MedCPTEncoder`** — two models ("towers"). `encode_queries` uses the query encoder with max
  length 256 (whole MedQA questions fit). `encode_articles` feeds `[title, content]` pairs to the
  article encoder (max 512), exactly as MedCPT was trained. Both take the **[CLS] vector** as the
  embedding. On GPU the models run in fp16 (`model.half()`): twice as fast, no measurable loss.
- **`DenseIndex`** — `faiss.IndexFlatIP` (exact inner product; MedCPT was trained with inner
  product, not cosine). Embeddings are saved as `embeddings.f16.npy` (~0.65 GB) and converted back
  to float32 on load. `search()` skips `-1` rows that FAISS uses as padding.
- **`MedCPTReranker`** — the cross-encoder. `score(pairs)` tokenises `(query, passage)` together,
  truncates to 512 tokens, returns one logit per pair (higher = more relevant; not a probability).

### 4.5 `retrieval/search.py` — the hybrid retriever (the heart of the RAG)

**Data classes**
- Queries are plain strings.
- `Evidence(row, snippet, text, score)` — one retrieved snippet, plus `source` and `title` copied from it.

**`reciprocal_rank_fusion(rankings, k=60)`** — for each list, every row gets `1 / (k + rank)`;
the sums decide the fused order. A passage ranked well by *both* BM25 and MedCPT beats one ranked
first by only one of them (the test `test_rrf_rewards_documents_ranked_well_by_both` checks this).

**`is_near_duplicate(text, others, 0.7)`** — Jaccard overlap of word sets ≥ 0.7. Textbooks repeat
each other; duplicates waste the evidence budget.

**`class Retriever`**
- `load_retriever(index_dir, device)` (a plain function at the bottom of the file) — loads
  `corpus.jsonl`, the BM25 folder, the FAISS embeddings and the cross-encoder, then builds the Retriever.
- `self.lock = threading.Lock()` — the benchmark runs many threads but there's one copy of each
  torch model; the lock makes threads take turns using it.
- `candidates(texts, k, mode)` — runs BM25 and/or dense search for a batch of queries, then RRF.
  If the requested index doesn't exist it falls back to the one that does (a bug we hit early).
- `search_many(queries, budget=10, per_query=2, pool=30, mode, rerank, dedupe, expand)`:
  1. **Candidates:** `pool` fused candidates per query (inside the lock).
  2. **Re-rank:** the cross-encoder re-scores each query's candidates.
  3. **Quota pass (4a):** each query first gets its own top `per_query` hits. Without this, one
     strong query could fill the whole budget by itself (RAG² searches with its rationale AND the
     question, and both should be heard).
  4. **Fill (4b):** the remaining budget goes to the best leftovers from any query.
  5. `add_if_new()` skips a snippet that's already chosen or is a near-copy of one.
  6. **Expand:** each snippet gets ±400 characters of its neighbouring paragraphs, but only from the
     same section — "retrieve small, read big", so a snippet starting "It is contraindicated..."
     still says what "it" is.
- `search(text, k)` — single-query convenience wrapper (what vanilla RAG uses).

### 4.6 `retrieval/evidence_tool.py` — the agent's `consult_reference` tool
This turns a short claim into a few **evidence cards** that read on their own.

**Helpers**
- `SECTIONS_FOR` maps a claim type to StatPearls section words:
  presentation → "history and physical", tests → "evaluation", mechanism → "pathophysiology", etc.
- `MIXED_SECTIONS = ("differential diagnos",)` — skipped: they discuss many diseases at once.
- `NEEDS_PREVIOUS` — a regex for sentences starting with "It / This / These / However..." that need
  the sentence before them.
- `words(text)` — content words, lower-cased, without stop-words.
- `split_sentences(text)` — splits at `. ! ?` followed by a capital/digit/bracket and at line
  breaks; a piece ending in an abbreviation like "Chap." is glued onto the next piece.

**`class EvidenceTool`**
- `__init__` groups StatPearls rows by article name once (`"gilbert syndrome" -> [rows]`), so finding
  an article is a dictionary lookup, and pre-computes each article's title words.
- `find_article(hypothesis)` — exact title match, otherwise the title with the highest word overlap
  (Jaccard ≥ 0.6). Results are remembered in `article_for_name`.
- `chunks_from_own_article(hypothesis, claim, about)` — rows from that article, minus mixed
  sections, narrowed to the right section if one exists, sorted by words shared with the claim,
  top 8. **Why article-first:** in the probe, passages about a *different* disease were counted as
  evidence; searching inside the disease's own article fixes attribution.
- `chunks_from_whole_library(claim)` — the normal hybrid search, top 4, minus mixed sections.
- `consult(requests)` — the tool, in three steps:
  1. candidate chunks for every claim (own article first, then the whole library);
  2. **one cross-encoder batch** scoring every (claim, sentence) pair of the turn — all hypotheses
     are scored together, so none goes "first";
  3. per claim, the best sentences become at most 2 cards from **different documents** (two cards =
     two independent sources); each card keeps up to 3 good sentences of its chunk, completed by
     `complete_sentence()` (previous sentence for "It...", the list after "include:"), in original order,
     headed by `source_header()` ("StatPearls: Silicosis, Introduction"). Below `min_score`
     nothing is returned, and the agent sees **"NOT FOUND"** rather than loosely related filler.

### 4.7 `rag/prompts.py` and `rag/methods.py` — the published baselines
- CoT and MedRAG prompts are copied **word for word** from the MedRAG toolkit, so baselines match
  the paper's setup.
- `MCQ(id, question, options, answer, dataset)`; `options_text()` prints "A. ...".
- `CoT` — one call, answer parsed with `parse_choice`.
- `sample_votes` + `majority` + `SelfConsistency` — 5 samples at temperature 0.7 in one request,
  count the letters, majority wins (ties go to the later letter — deterministic).
- `MedRAG` — the whole question as one query, dense MedCPT top-32, pasted into MedRAG's prompt.
  Deliberately no hybrid/re-rank/dedupe/expansion: this is the published setup.
- `RationaleRAG` (RAG²-style) — first write a short rationale *without options*, then search with
  both the rationale (textbook language) and the question (patient story).
- `IMedRAG` — 3 rounds × 3 follow-up questions, each answered by vanilla RAG, then a final answer.
  Note: like the original, it sees the options while writing follow-ups.
- DDx-RAG v1 (our first idea) was removed from the code once the agent replaced it; its results
  are in the implementation log and its code is in the git history.
- `METHODS` maps names to classes; `RETRIEVAL_FREE = {"cot", "cot_sc"}` lets the runner skip
  loading the 3 GB index for those.

---

## 5. Code tour — the agent (`rag/agent.py` + `rag/agent_prompts.py`)

### 5.1 The prompts, and why each rule is there
`AGENT_SYSTEM` makes the model an attending physician who reasons from its own knowledge first and
treats the library as a reference to **check claims**. Four rules come straight from measured failures:
1. "A passage that merely discusses a disease is NOT evidence the patient has it." (v1 was pulled
   toward on-topic passages.)
2. "If the evidence does not address the claim, its status is NOT_ADDRESSED. Silence is never
   refutation."
3. "For every hypothesis, name the finding that fits it WORST." (Against anchoring.)
4. "Prefer the hypothesis that explains the most findings, including important normal results."

`FRAME_USER` asks for findings, what is asked, up to k hypotheses with 2 checkable claims each, and a
leading answer that "must BE the answer" ("Hydralazine", never "next step in management").
`ROUND_USER` shows the case, findings (F1, F2...), the shuffled dossier, *which claims are waiting
for judgement*, open critic issues, harness notes, the turn budget, and the four tools.
`CRITIC_SYSTEM/USER` — a senior reviewer: make the strongest case for the best alternative, find
unexplained findings, propose missed diagnoses, flag weak rule-outs; may only mark an issue
answer-changing if it writes the alternative answer.
`MAP_*` — match free-text answers to options; told to watch direction words (increased/decreased).
`SELECT_*` — with `final=select`, the options are revealed here, after the whole work-up.

### 5.2 Helpers at the top of `agent.py`
- `text_from(value)` — model text, or `""` if missing or a placeholder echo.
- `id_from(value)` — `"e3 "` → `"E3"`. Ids can't go through `text_from` (no real words, so they'd
  look like placeholders — this exact bug made every citation fail in an early test).
- `list_from(parsed, key)` — `parsed[key]` only if it's really a list (the model sometimes writes null).
- `question_sentence(question)` — only the final "Which of the following...?" sentence, for the
  matcher, which must never see the patient story.

### 5.3 `WorkUp` — everything known about one case
A plain class whose `__init__` sets: `findings`, `asked`, `hypotheses` (name → why, worst_fit, status, origin, claims,
rule_out), `claims` ("C1" → hypothesis, text, about, cards, status, card), `cards` ("E1" → card),
`issues` ("I1" → critic objection), `leading` (current best hypothesis + answer), `turns`,
`critiques`, `notes_for_agent`, `events` (a diary), `parse_failures`, `raw_failures`.
Small methods: `log()`, `live_hypotheses()`, `open_issue_ids()`, `claims_waiting_for_status()`
(claims with cards but no status yet).

### 5.4 `DxAgent.__init__` — the knobs
`k=4` hypotheses, `claims=2` per hypothesis per turn, `max_rounds=4` agent turns,
`max_critiques=2`, `max_new_hypotheses=2` per turn, `library` (False = the "no library" ablation),
`critic`, `article_first`, `cards_per_claim=2`, `min_score`, `final="map"|"select"`,
`max_tokens=1500`. It builds the `EvidenceTool` only if the library is on and there are turns.
Every knob is settable from the command line: `--set max_rounds=0 final=select`.

### 5.5 `__call__(q)` — one question, start to finish
1. `workup = self.frame(q)`; remember `first_guess` (so we can measure whether the loop helped).
2. `shuffler = random.Random(q.id)` — hypotheses are shuffled differently every turn, but the same
   way every run (reproducible).
3. If there are turns: `look_up_new_claims()` (all claims at once), then `run_turns()`.
4. `match_to_options()` maps both the first guess and the final answer → `first_choice`,
   `open_choice` (the strictly options-free result, always recorded).
5. `final="select"` → `pick_option()` (options revealed now). Otherwise use `open_choice`.
6. If nothing matched, a normal CoT answer with options is used and flagged `mapping_fallback`.

### 5.6 `ask()` — every structured call goes through here
Calls `llm.chat(..., temperature=0.0, json_mode=True)`, parses, and on failure counts it and keeps
the first 1,500 characters of the reply (`raw_failures`) so failures can be read, not guessed.

### 5.7 `frame(q)` — step 1
Fills `findings`, `asked` (falls back to the question sentence), up to k hypotheses via
`add_hypothesis(..., added_by="frame")`, and `leading` (falls back to the top hypothesis).

### 5.8 Bookkeeping
- `find_hypothesis(name)` — exact (case-insensitive) match, then "one name contains the other"
  ("TTP" ↔ "Thrombotic Thrombocytopenic Purpura (TTP)").
- `add_claim()` — skips empty and **repeated** claims; new id `C{n}`; `cards=None` means "not looked
  up yet", `[]` means "looked up, nothing found".
- `add_hypothesis()` — if it already exists and the critic proposes it again after a rule-out, it
  comes back to life; otherwise a new entry with its claims; logged with who added it.
- `look_up_new_claims()` — collects every claim with `cards is None`, sends them to
  `tool.consult()` **in one batch**, numbers the returned cards `E{n}`. With the library off, every
  claim simply gets `[]`.

### 5.9 Rendering for prompts
`findings_text()` numbers findings F1, F2...; `dossier_text()` writes each hypothesis (shuffled) with
status, why, worst fit, every claim with its verdict and cards (or "NOT FOUND");
`open_issues_text()` and `notes_text()` (notes are shown once, then cleared).

### 5.10 `run_turns()` — the loop and its brakes
```
while turns < max_rounds:
    wants_critique, asked_for_more = agent_turn()
    look_up_new_claims()                      # evidence for anything new
    if not wants_critique and not out_of_turns: continue
    if wants_critique and asked_for_more: defer ("read the new evidence first"), continue
    if critiques >= max_critiques: stop
    blocking = critique()
    stop if: no serious new objection, OR same answer after two critiques, OR no turns left
    look_up_new_claims()                      # the critic's new diagnoses need evidence too
```
These brakes are why an agent ↔ critic argument can't run forever; the test
`test_loop_always_ends_when_the_critic_never_agrees` uses a stubborn critic and a flip-flopping agent
and checks the loop still ends within the caps.

### 5.11 `agent_turn()` — one turn
Builds `ROUND_USER` (incl. "claims waiting for your judgement: C3, C4"), calls `ask()`, then:
- `read_claim_statuses()` — accepts statuses **only for the claims we asked about** (the model once
  invented C9–C24 until it hit the length limit);
- `read_replies_to_critic()` — a reply closes an issue for now; the critic sees the reply next time;
- `run_tools()` — `request_critique`; `consult_reference` (same cap per hypothesis, so the favourite
  can't hog the library); `add_hypothesis` (capped); `rule_out`;
- a turn asking for nothing new counts as "done → send to the critic";
- updates `leading` and logs the turn.

### 5.12 `rule_out()` — evidence or nothing
Accepted only if the citation exists: a card id (`E7`) or a finding id within range (`F3`); not
accepted for the last live hypothesis. Rejections become a note the agent reads next turn
("Rule-out rejected: 'E99' is not a card or finding id").

### 5.13 `critique()` and `record_new_issues()`
Builds the critic prompt with the dossier, rule-outs, current answer, and its own earlier issues
next to the agent's replies. Then: reopens earlier issues it still disputes; records new issues — an
issue is **serious** only if `changes_answer` is true *and* it names an alternative answer *and*
that answer differs from the current one (the first run's critic flagged 296/298 reviews without
naming alternatives); adds up to 2 missed diagnoses (checked like any other); restores rule-outs it
thinks were hasty.

### 5.14 `match_to_options()` and `pick_option()`
The matcher sees the free-text answers, the question sentence and the options — **never the case**.
`pick_option()` (final=select) sees the case, a summary of the work-up (per hypothesis: live/ruled
out, claims supported/contradicted), open doubts, and the options.

### 5.15 `trace()` — what is saved
findings, asked, first guess (+ its option), final, open_choice, final_step, mapping_fallback,
turns, critiques, open issues, every hypothesis with full claims, all cards, all issues, the event
diary, parse failures. Everything needed to audit or demo any answer.

---

## 5b. Code tour — the clinical agent (`agent.py`, `tools/`, `schemas.py`)

This is the interactive agent from the original project, rewritten simply. It works up one patient
like an emergency doctor: read the notes, order tests one at a time, write a report.

### `schemas.py` — the data
Pydantic classes (pydantic checks the types when an object is made): `PatientCase` (age, complaint,
vitals, notes, X-ray path, lab values), `FindingItem` (one finding, present/absent, value + unit),
`ToolResult` (a tool's findings + summary + raw output), `DifferentialHypothesis` (condition, suspicion
tier, rationale), `ClinicalReport` (diagnosis, disposition, differential, impression, recommendations,
evidence trail). The enums fix the allowed values, e.g. `DispositionType` = admit / discharge / escalate.

### `tools/` — four tools with the same shape
Every tool has a `name`, a one-line `description` (shown to the LLM so it knows what each tool does)
and `run(case, **options)` returning a `ToolResult`.
- `laboratory.py` — `REFERENCE_RANGES` maps each lab to (low, high, unit, meaning if high, meaning if
  low). Each value is marked HIGH / LOW / NORMAL; a normal value is an "absent" finding.
- `chest_xray.py` — TorchXRayVision DenseNet-121. `load_model()` keeps only labels with a real
  threshold (NaN thresholds = labels never trained); `classify()` normalises, crops and resizes the
  image to 224×224 and returns calibrated scores (0.5 = positive). Bug fixed in Phase 1: the old code
  applied a sigmoid twice.
- `medsam.py` — MedSAM gets three boxes (right lung, left lung, heart) and outlines each; we measure
  lung symmetry (smaller/bigger area), collapse %, and the cardiothoracic ratio (heart width / chest
  width; > 0.50 = enlarged heart).
- `medical_knowledge.py` — calls the same hybrid retriever we benchmarked (`retriever.search(query, k)`).
Image models load the first time they're used (`self.model is None` → `load_model()`).

### `agent.py` — the LangGraph state machine
- `WorkupState` lists what flows between steps: case, differential, evidence, tools used, next tool,
  reason, query, turns, max turns, report.
- Four methods are the graph's nodes:
  - `intake` — prompt with age, complaint, vitals, notes → pertinent positives/negatives (repeats
    removed by `unique_texts`) and a first differential (`hypotheses_from`, tiers via `tier_from`).
  - `choose` — lists the unused tools with their descriptions plus the evidence so far; the LLM picks
    one or says STOP. An unknown name falls back to the first unused tool; the turn limit forces STOP.
  - `run_tool` — runs it (with the library query if there is one) and appends the result.
  - `report` — writes the final report. **Anything other than a clear admit or discharge becomes
    ESCALATE_TO_PHYSICIAN** — the agent never silently sends someone home because a reply was unclear.
- `build_graph()` wires them: intake → choose → (run_tool → choose)* → report, with `next_step()`
  deciding after each choose. `diagnose(case)` builds the start state and runs the graph.
- `on_token` / `on_event` are optional hooks: the demo uses them to print the reasoning live and each
  step's result.

### `demo_execution.py`
`python demo_execution.py --case 0` prints the patient, then streams every step through those hooks
and prints the final report. Cases 8–10 are stress tests where escalation is the right answer.

## 6. A real case, step by step (from the check run, question `train-7`)

**Case:** 54-year-old man, worsening chronic cough, shortness of breath, low fever; **30 years in a
sandblasting factory**; chest X-ray: **eggshell calcification of hilar lymph nodes**.
"This patient is at increased risk for which of the following conditions?"
(Options, hidden until the end: COPD, lung adenocarcinoma, pulmonary embolism, **tuberculosis** — correct.)

1. **Frame (call 1).** 12 findings extracted; asked = "the condition he is at increased risk for";
   hypotheses: Silicosis, Pulmonary tuberculosis, COPD, Pulmonary fibrosis, each with 2 claims.
2. **Library, all 8 claims at once.** Examples:
   - "Silicosis is caused by inhalation of crystalline silica dust" → card E1, *StatPearls:
     Silicosis, Introduction* → **SUPPORTED**.
   - "Pulmonary tuberculosis is caused by Mycobacterium tuberculosis" → card from *Primary Lung
     Tuberculosis, Introduction* → judged NOT_ADDRESSED (the card is about TB burden, not the claim).
   - COPD claim → a Harrison's textbook card about dusts and chronic bronchitis.
3. **Turn 1 (call 2).** Leading: Silicosis. Requests a critique.
4. **Critique 1 (call 3).** Strongest alternative: tuberculosis. One serious objection (names COPD
   as the alternative); proposes **Asbestosis** — which goes to the library like any hypothesis
   ("Asbestosis is caused by inhalation of asbestos fibers" → *StatPearls: Asbestosis, Evaluation*).
5. **Turn 2 (call 4).** Rebuts the COPD objection: "COPD does not explain the calcified hilar lymph
   nodes." (A citation-backed reply, as the rules ask.)
6. **Critique 2 (call 5).** Proposes sarcoidosis; nothing serious left → loop ends.
7. **Answer (calls 6–7).** Free-text conclusion "Silicosis is the condition..." — note the agent
   answered the *diagnosis*, not the *risk* asked about. When the options are revealed, the final
   step picks **D. Mycobacterium tuberculosis infection** (silicosis raises TB risk) — correct.
8. **Cost:** 7 calls, ~13k prompt tokens, ~3k generated.

**What this case shows honestly:** retrieval and attribution work well (every card is from the right
article); the critic adds alternatives and the agent defends with findings; but an 8B model still
mixes up "diagnosis" with "what the question asks" — exactly the two-hop weakness we measured.

---

## 7. Evaluation and scripts

### 7.1 `evaluation/mirage.py`
- `load_questions(dataset)` — MIRAGE's `benchmark.json` (MedQA, MMLU-Med, MedMCQA, PubMedQA*, BioASQ)
  or `medqa_dev`: a fixed sample of 300 MedQA **training** questions (seed 0) for tuning — the test
  sets are never used for tuning. `--limit` takes the first N of that same sample.
- `run_method()` — answers every question not already in the output file (**resumable**), in
  parallel threads; each row stores id, gold, pred, correct, usage (calls/tokens), error, trace. One
  failing question records its error instead of killing a two-hour run; each line is flushed.
- `summarize()` / `comparison_table()` — accuracy, Wilson 95% CI, McNemar p vs a baseline,
  fixed/broke counts, unparsed, errors, calls/question, tokens/question — on the questions every
  method answered.

### 7.2 `evaluation/stats.py`
- `wilson_interval(correct, total)`.
- `mcnemar_exact(a, b)` — b01 = A wrong & B right, b10 = A right & B wrong; under "no difference"
  each disagreement is a coin flip; exact binomial two-sided p-value.

### 7.3 Scripts
- `build_index.py download|build` — downloads corpora; builds BM25 and MedCPT embeddings; refuses
  to run on CPU by accident (the server's torch once silently lost the GPU).
- `run_mirage.py` — `--dataset --methods --limit --workers --set k=v --tag --run-name
  --report-only`. `--set key=value` values are turned into numbers / True / False where they look
  like one and passed to the method's constructor. Writes `<method>.config.json` next to results.
- `analyze_results.py` — every method's accuracy split by how much the 5 votes agree (the
  "retrieval hurts when sure" table), per-subject accuracy, fixed/broke ids, and for the agent:
  harm, recovery, critic activity, open issues, mapping fallbacks.
- `serve_vllm.sh` — `vllm serve` with bf16, max length 16,384, `--gpu-memory-utilization`
  (`GPU_UTIL`), `--max-num-seqs` (`MAX_SEQS`), prefix caching, optional `QUANT=fp8`; serves the
  local snapshot folder so it works offline.

### 7.4 Tests (33, CPU only, under a minute)
`conftest.py` provides a 6-snippet toy corpus with a BM25-only retriever and a `FakeLLM` that replies
by keyword. Tests cover RRF, de-duplication, quotas, neighbour expansion, the StatPearls chunker,
answer parsing (incl. the `A/B/C/...` echo), every baseline's flow, statistics, and the agent: options
never reach the agent or critic, the matcher never sees the case, fake citations are rejected, the
critic's new diagnosis is checked in the library, the loop always ends, invented claim ids and repeated
claims are ignored, evidence cards are attributed and whole. For the clinical agent: the whole LangGraph
workup runs (intake → choose → tool → choose → report), the turn limit stops it, an unclear disposition
is escalated to a human, and each tool returns the right findings (the image tests use the real models).

---

## 8. Infrastructure and what went wrong on the server
- **Environments with uv** (fast pip replacement): one for the project, one for vLLM (it pins its own torch).
- **Driver 535 = CUDA 12.2.** Default wheels were CUDA 13: torch silently saw no GPU, vLLM failed
  with `libcudart.so.13`. Fixed with CUDA-12 builds (`--torch-backend=cu126/cu128`, `vllm==0.11.0`)
  and pinned `transformers<5` (v5 broke vLLM 0.11's tokenizer code).
- **Offline serving** of a partial download failed; the script now serves the local snapshot folder.
- **Throughput:** the client showed 588% CPU, which looked like a bottleneck, but vLLM's own metrics
  showed 99% GPU and 48/48 requests running — the LLM was the bottleneck. Lesson: measure the server side.
- **Shared GPU:** another user took 17 GB; we ran vLLM with `GPU_UTIL=0.5 MAX_SEQS=16`.
- **Secrets:** tokens typed into hidden prompts and used for one command; nothing saved on the server.

---

## 9. Questions you should expect, with answers

**Why hybrid retrieval and not just embeddings?**
Embeddings miss exact terms (drug names, lab names, eponyms); BM25 misses paraphrases. RRF merges them
using ranks, so the two scoring scales never need calibrating. MIRAGE also found fused retrievers best.

**Why a cross-encoder re-ranker?**
A bi-encoder compresses each passage into one vector before seeing the query; a cross-encoder reads
query and passage together and can tell "D-dimer rules OUT PE" from "D-dimer is raised in PE". It's
too slow for 427k passages, so it only re-ranks ~30 candidates per query.

**Why FAISS flat and not a vector database (Pinecone, Weaviate, pgvector)?**
427k × 768 floats fits in memory and exact search takes milliseconds; a vector DB adds a service to
run without improving accuracy at this size. At tens of millions of passages, or with frequent
updates, filtering and multi-tenancy, I'd move to an approximate index (HNSW/IVF) or a vector DB.

**How did you chunk?**
Exactly like MedRAG (to stay comparable): paragraph-level snippets, short paragraphs merged up to
1,000 characters, section titles carried into the snippet title. At read time we "retrieve small, read
big" (neighbour expansion) or, in the agent, cut to whole sentences with their context.

**How do you prevent hallucination?**
Claims are checked against cards with source headers; statuses must cite a card; rule-outs must cite
a card or a finding (checked in code); "NOT FOUND" instead of filler; every step is in the trace.

**Why didn't RAG beat self-consistency?**
At 8B the model already "knows" most answers it gets right; giving it plausible but off-target text
changes some right answers to wrong ones (we counted: 91 harmful vs 54 helpful overrules in v1). The
benefit only shows on questions where it's unsure, so retrieval should be targeted, not always-on.

**How would you deploy this?**
Answer alone when the 5 samples agree (89% accurate on 54% of MedQA), and escalate the rest — with the
agent's trace (hypotheses, evidence, open doubts) attached for the clinician. Retrieval and the agent
loop run only on the uncertain cases, which also cuts cost.

**Why not LangGraph for the new agent?**
The loop is small (frame → check → turn → critique → answer) and the benchmark needs exact control
and counting of every call; strict JSON validated in our code is more reliable on an 8B model than
native function calling. The clinical agent (`agent.py`) does use LangGraph: its intake → choose →
run tool → report loop is a StateGraph, and the DxAgent loop would map onto one the same way.

**Why Llama-3.1-8B?**
Open weights, fits one GPU, and it's the model the published baselines (i-MedRAG) report — so our
numbers are directly comparable. Fine-tuning or a bigger model is a separate experiment.

**Is anything trained?**
No. All retrievers and the LLM are used off the shelf; the gain or loss comes from the pipeline only.

**How do you know the numbers aren't noise?**
Wilson intervals, paired McNemar tests on the same questions, and tuning only on MedQA *training*
questions. A 20-question run is a smoke test; we never quote it as a result.

**What would you do next?**
Score the agent on an options-free benchmark (DDXPlus — diagnosis from symptoms, no options); make
retrieval conditional on uncertainty; try a reasoning model as the critic; and port the loop into the
LangGraph clinical agent so the triage + escalation runs end to end.

---

## 10. Honest limitations (say these before they're found)
- RAG did not beat self-consistency on MedQA with an 8B model; we report that openly.
- The agent costs ~8 calls and ~22k tokens per question — it should run only on uncertain cases.
- An 8B model rarely uses rule-out, sometimes answers the diagnosis instead of the question asked, and
  sometimes never proposes the right diagnosis (hypothesis generation is the bottleneck).
- The full agent has only been run on 20 practice questions after its fixes — it's verified to work,
  not yet benchmarked.
