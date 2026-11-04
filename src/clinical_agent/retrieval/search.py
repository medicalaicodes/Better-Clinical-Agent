"""
The retriever: turns one or many queries into a small, diverse, relevant set of evidence.

For every query:
    1. candidates   BM25 top-k  and  MedCPT top-k                (sparse + dense)
    2. fuse         Reciprocal Rank Fusion of the two rankings     (no score-scale mismatch)
    3. rerank       MedCPT cross-encoder on the fused pool         (reads query + passage together)

Then, across all queries (`search_many`):
    4. select       first a quota per query, then fill by score    (every intent gets a voice)
                    and skip near-duplicates                       (the textbooks repeat each other)
    5. expand       add the neighbouring paragraphs                ("retrieve small, read big")

Every step can be switched off. That is how the baselines and the ablations are built.
"""

import os
import random
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .corpus import CorpusStore, Snippet
from .indexes import BM25Index, DenseIndex, Hits, MedCPTEncoder, MedCPTReranker


@dataclass
class Query:
    text: str
    intent: str = "QUESTION"   # QUESTION | SUPPORT | DISCRIMINATE | REFUTE | RATIONALE | AGENT
    about: str = ""            # which hypothesis this query is about (if any)


@dataclass
class Evidence:
    row: int                   # position of the snippet in the CorpusStore
    snippet: Snippet
    text: str                  # snippet text, possibly expanded with its neighbours
    score: float               # cross-encoder score (or fused score when reranking is off)
    intents: List[str] = field(default_factory=list)   # which query intents found it
    queries: List[str] = field(default_factory=list)   # which query texts found it

    @property
    def source(self) -> str:
        return self.snippet.source

    @property
    def title(self) -> str:
        return self.snippet.title


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: Sequence[Hits], k: int = 60) -> Hits:
    """Merges several ranked lists into one.

    Each document gets   sum over lists of  1 / (k + its rank in that list).
    Only ranks are used, never raw scores, so BM25 scores (0..30) and cosine scores (0..1)
    never have to be put on the same scale. k=60 is the standard value (Cormack et al., 2009).
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (row, _score) in enumerate(ranking):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank + 1)

    # Best fused score first.
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def word_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_near_duplicate(text: str, others: Sequence[str], threshold: float = 0.7) -> bool:
    """True if `text` shares >= 70% of its words with any of `others` (Jaccard overlap).
    Cheap, and good enough to catch the same fact copied across two textbooks."""
    words = word_set(text)
    for other in others:
        other_words = word_set(other)
        union = words | other_words
        if not union:
            continue
        overlap = len(words & other_words) / len(union)
        if overlap >= threshold:
            return True
    return False


# ----------------------------------------------------------------------------------------
# The retriever
# ----------------------------------------------------------------------------------------

class Retriever:
    def __init__(self, corpus: CorpusStore, bm25: Optional[BM25Index] = None,
                 dense: Optional[DenseIndex] = None, reranker: Optional[MedCPTReranker] = None):
        if bm25 is None and dense is None:
            raise ValueError("Retriever needs at least one of: BM25 index, dense index.")

        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

        # snippet id -> row, so we can find a snippet's neighbours ("Book_12" -> "Book_13").
        self.row_of_id = {}
        for row, doc_id in enumerate(corpus.ids):
            self.row_of_id[doc_id] = row

        # Robustness test: when > 0, this many random snippets are added to every result.
        self.noise_snippets = 0

        # The benchmark runs many questions in parallel threads, but they all share the
        # same torch models. One lock keeps the models from being used concurrently.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, index_dir: str, device: Optional[str] = None, use_bm25: bool = True,
             use_dense: bool = True, use_reranker: bool = True) -> "Retriever":
        """Loads what scripts/build_index.py wrote into `index_dir`."""
        corpus = CorpusStore.load(os.path.join(index_dir, "corpus.jsonl"))

        bm25 = None
        if use_bm25:
            bm25 = BM25Index.load(os.path.join(index_dir, "bm25"))

        dense = None
        if use_dense:
            encoder = MedCPTEncoder(device=device)
            dense = DenseIndex.load(os.path.join(index_dir, "medcpt"), encoder=encoder)

        reranker = None
        if use_reranker:
            reranker = MedCPTReranker(device=device)

        return cls(corpus, bm25, dense, reranker)

    # ------------------------------------------------------------------------------------
    # Steps 1-2: candidates from BM25 and/or MedCPT, fused with RRF
    # ------------------------------------------------------------------------------------
    def candidates(self, texts: Sequence[str], k: int = 50, mode: str = "hybrid") -> List[Hits]:
        """One ranked candidate list per query text.

        mode = "hybrid" (both, fused), "bm25", or "dense".
        If the requested index does not exist, it falls back to the one that does.
        """
        if mode == "dense" and self.dense is None:
            mode = "bm25"
        if mode == "bm25" and self.bm25 is None:
            mode = "dense"

        sparse_lists = None
        if self.bm25 is not None and mode in ("hybrid", "bm25"):
            sparse_lists = self.bm25.search(texts, k)

        dense_lists = None
        if self.dense is not None and mode in ("hybrid", "dense"):
            dense_lists = self.dense.search(texts, k)

        # Only one kind available -> nothing to fuse.
        if sparse_lists is None:
            return dense_lists
        if dense_lists is None:
            return sparse_lists

        fused_lists = []
        for sparse_hits, dense_hits in zip(sparse_lists, dense_lists):
            fused = reciprocal_rank_fusion([sparse_hits, dense_hits])
            fused_lists.append(fused[:k])
        return fused_lists

    # ------------------------------------------------------------------------------------
    # Steps 1-5 for a list of queries
    # ------------------------------------------------------------------------------------
    def search_many(self, queries: Sequence[Query], budget: int = 10, per_query: int = 2,
                    pool: int = 30, mode: str = "hybrid", rerank: bool = True,
                    min_score: Optional[float] = None, dedupe: bool = True,
                    expand: bool = True) -> List[Evidence]:
        """Runs several queries and merges their results into one evidence list.

        budget     total number of snippets returned
        per_query  how many snippets each query is guaranteed (if it finds good ones)
        pool       how many fused candidates per query go to the cross-encoder
        min_score  drop anything the cross-encoder scores below this (None = keep all)
        """
        queries = [q for q in queries if q.text and q.text.strip()]
        if not queries:
            return []

        # --- Steps 1-3: candidates + rerank, separately for each query ---
        with self._lock:
            query_texts = [q.text for q in queries]
            candidate_lists = self.candidates(query_texts, k=pool, mode=mode)

            ranked_lists = []
            for query, hits in zip(queries, candidate_lists):
                if rerank and self.reranker is not None:
                    hits = self._rerank(query.text, hits)
                ranked_lists.append(hits)

        chosen: Dict[int, Evidence] = {}     # row -> Evidence, in the order they were picked

        def take(row: int, score: float, query: Query) -> None:
            """Adds one snippet to the evidence set (or tags it if it is already there)."""
            if row in chosen:
                # Two queries found the same snippet: remember both reasons it was found.
                chosen[row].intents.append(query.intent)
                chosen[row].queries.append(query.text)
                return

            text = self.corpus[row].content
            if dedupe:
                already = [e.snippet.content for e in chosen.values()]
                if is_near_duplicate(text, already):
                    return

            chosen[row] = Evidence(row=row, snippet=self.corpus[row], text=text, score=score,
                                   intents=[query.intent], queries=[query.text])

        # --- Step 4a: quota ---
        # Each query first gets its own top `per_query` hits. Without this, a few strong
        # SUPPORT queries would fill the whole budget and the REFUTE / DISCRIMINATE
        # evidence (the part that fights anchoring) would never reach the model.
        for query, hits in zip(queries, ranked_lists):
            taken = 0
            for row, score in hits:
                if taken >= per_query or len(chosen) >= budget:
                    break
                if min_score is not None and score < min_score:
                    break                      # the rest of this list is even worse
                take(row, score, query)
                taken += 1                     # a shared snippet still counts for this query

        # --- Step 4b: fill the remaining budget with the best leftovers from any query ---
        leftovers = []
        for query, hits in zip(queries, ranked_lists):
            for row, score in hits:
                if row not in chosen:
                    leftovers.append((score, row, query))
        leftovers.sort(key=lambda item: item[0], reverse=True)

        for score, row, query in leftovers:
            if len(chosen) >= budget:
                break
            if min_score is not None and score < min_score:
                break
            take(row, score, query)

        evidence = sorted(chosen.values(), key=lambda e: e.score, reverse=True)

        # Robustness test: inject random, irrelevant snippets.
        if self.noise_snippets:
            evidence += self._random_evidence(self.noise_snippets)

        # --- Step 5: expand each snippet with its neighbouring paragraphs ---
        if expand:
            for item in evidence:
                item.text = self.expand(item.row)

        return evidence

    def search(self, text: str, k: int = 10, **kwargs) -> List[Evidence]:
        """Single-query search, i.e. what vanilla RAG does."""
        return self.search_many([Query(text)], budget=k, per_query=k, **kwargs)

    # ------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------
    def _rerank(self, query: str, hits: Hits) -> Hits:
        """Re-scores candidates with the cross-encoder and re-sorts them."""
        if not hits:
            return hits

        pairs = []
        for row, _old_score in hits:
            pairs.append((query, self.corpus[row].text))
        new_scores = self.reranker.score(pairs)

        reranked = []
        for (row, _old_score), new_score in zip(hits, new_scores):
            reranked.append((row, float(new_score)))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def expand(self, row: int, max_neighbour_chars: int = 400) -> str:
        """Returns the snippet with the end of the previous paragraph and the start of the
        next one, but only if they belong to the same article section.

        Why: chunking cuts text mid-thought, e.g. a snippet that starts "It is contraindicated
        in..." no longer says what "it" is. We rank on the small snippet (precise) and then
        show the model a slightly bigger window (context). Neighbours are capped at 400
        characters so the prompt does not balloon.
        """
        snippet = self.corpus[row]

        # Snippet ids look like "<article>_<paragraph number>", e.g. "article_17".
        prefix, _, number = snippet.id.rpartition("_")
        if not number.isdigit():
            return snippet.content

        def neighbour(offset: int) -> Optional[Snippet]:
            other_row = self.row_of_id.get(f"{prefix}_{int(number) + offset}")
            if other_row is None:
                return None
            if self.corpus.titles[other_row] != snippet.title:
                return None        # different section: don't glue unrelated text together
            return self.corpus[other_row]

        before = neighbour(-1)
        after = neighbour(+1)

        parts = []
        if before is not None:
            parts.append("..." + before.content[-max_neighbour_chars:])
        parts.append(snippet.content)
        if after is not None:
            parts.append(after.content[:max_neighbour_chars] + "...")
        return " ".join(parts)

    def _random_evidence(self, n: int) -> List[Evidence]:
        """n random snippets, marked as NOISE (fixed seed, so runs are repeatable)."""
        rng = random.Random(len(self.corpus))
        rows = rng.sample(range(len(self.corpus)), n)
        noise = []
        for row in rows:
            noise.append(Evidence(row=row, snippet=self.corpus[row], text=self.corpus[row].content,
                                  score=float("-inf"), intents=["NOISE"], queries=[]))
        return noise
