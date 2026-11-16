"""
The retriever: turns one or more queries into a small, varied, relevant set of passages.

For every query:
    1. candidates   BM25 top-k  and  MedCPT top-k        (keywords + meaning)
    2. fuse         Reciprocal Rank Fusion of the two    (uses ranks, so the scores never need
                                                          to be on the same scale)
    3. re-rank      MedCPT cross-encoder on the fused list

Then, over all the queries together (search_many):
    4. select       first a quota for every query, then the best leftovers,
                    skipping near-duplicates (the textbooks repeat each other a lot)
    5. expand       add a bit of the neighbouring paragraphs ("retrieve small, read big")

Every step can be switched off - that's how the published baselines are rebuilt exactly.
"""

import os
import re
import threading

from .corpus import load_corpus_store
from .indexes import MedCPTEncoder, MedCPTReranker, load_bm25, load_dense


class Evidence:
    """One retrieved snippet, ready to show to a model."""

    def __init__(self, row, snippet, text, score):
        self.row = row            # the snippet's position in the CorpusStore
        self.snippet = snippet
        self.text = text          # the snippet, possibly with its neighbours added
        self.score = score        # cross-encoder score (or the fused score if re-ranking is off)
        self.source = snippet.source
        self.title = snippet.title


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings, k=60):
    """Merges several ranked lists into one.

    Every row gets, from each list it appears in, 1 / (k + its rank). The totals decide the
    new order. A passage ranked well by BOTH lists beats one ranked first by only one of them.
    k = 60 is the standard value from the original paper (Cormack et al., 2009).
    """
    totals = {}
    for ranking in rankings:
        rank = 1
        for row, score in ranking:
            if row not in totals:
                totals[row] = 0.0
            totals[row] += 1.0 / (k + rank)
            rank += 1

    fused = list(totals.items())
    fused.sort(key=get_score, reverse=True)
    return fused


def get_score(pair):
    # Used for sorting (row, score) pairs by their score.
    return pair[1]


def word_set(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_near_duplicate(text, others, threshold=0.7):
    """True if `text` shares at least 70% of its words with any of `others`.
    Cheap, and good enough to catch the same fact copied in two textbooks."""
    words = word_set(text)
    for other in others:
        other_words = word_set(other)
        all_words = words | other_words
        if not all_words:
            continue
        overlap = len(words & other_words) / len(all_words)
        if overlap >= threshold:
            return True
    return False


# ----------------------------------------------------------------------------------------
# The retriever
# ----------------------------------------------------------------------------------------

class Retriever:
    def __init__(self, corpus, bm25=None, dense=None, reranker=None):
        if bm25 is None and dense is None:
            raise ValueError("The retriever needs at least one index: BM25 or dense.")

        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

        # snippet id -> row, so we can find a snippet's neighbours ("Book_12" -> "Book_13").
        self.row_of_id = {}
        for row in range(len(corpus.ids)):
            self.row_of_id[corpus.ids[row]] = row

        # The benchmark answers many questions in parallel threads, but there is only one copy
        # of each model. This lock makes threads take turns using them.
        self.lock = threading.Lock()

    # ------------------------------------------------------------------------------------
    # Steps 1-2: candidates from BM25 and/or MedCPT, merged with RRF
    # ------------------------------------------------------------------------------------
    def candidates(self, texts, k=50, mode="hybrid"):
        """One ranked list of (row, score) per query text.
        mode is "hybrid" (both), "bm25" or "dense". If the index we asked for isn't there,
        we quietly use the one that is."""
        if mode == "dense" and self.dense is None:
            mode = "bm25"
        if mode == "bm25" and self.bm25 is None:
            mode = "dense"

        keyword_lists = None
        if self.bm25 is not None and mode in ("hybrid", "bm25"):
            keyword_lists = self.bm25.search(texts, k)

        meaning_lists = None
        if self.dense is not None and mode in ("hybrid", "dense"):
            meaning_lists = self.dense.search(texts, k)

        if keyword_lists is None:
            return meaning_lists
        if meaning_lists is None:
            return keyword_lists

        fused_lists = []
        for keyword_hits, meaning_hits in zip(keyword_lists, meaning_lists):
            fused = reciprocal_rank_fusion([keyword_hits, meaning_hits])
            fused_lists.append(fused[:k])
        return fused_lists

    def rerank(self, query, hits):
        """Scores every candidate again with the cross-encoder and re-sorts them."""
        if not hits:
            return hits

        pairs = []
        for row, old_score in hits:
            pairs.append((query, self.corpus[row].text))
        new_scores = self.reranker.score(pairs)

        reranked = []
        for i in range(len(hits)):
            row = hits[i][0]
            reranked.append((row, float(new_scores[i])))
        reranked.sort(key=get_score, reverse=True)
        return reranked

    # ------------------------------------------------------------------------------------
    # Steps 1-5 for a list of queries
    # ------------------------------------------------------------------------------------
    def search_many(self, queries, budget=10, per_query=2, pool=30, mode="hybrid",
                    rerank=True, dedupe=True, expand=True):
        """Runs several query strings and merges what they find into one evidence list.

        budget     how many snippets come back in total
        per_query  how many snippets each query is promised first (if it finds good ones)
        pool       how many fused candidates per query go to the cross-encoder
        """
        queries = [query for query in queries if query and query.strip()]
        if not queries:
            return []

        # Steps 1-3: candidates and re-ranking, separately for every query.
        with self.lock:
            candidate_lists = self.candidates(queries, k=pool, mode=mode)
            ranked_lists = []
            for query, hits in zip(queries, candidate_lists):
                if rerank and self.reranker is not None:
                    hits = self.rerank(query, hits)
                ranked_lists.append(hits)

        chosen = {}            # row -> Evidence, in the order we picked them

        # Step 4a - the quota: every query first gets its own top `per_query` snippets.
        # Without this, one strong query could fill the whole budget by itself.
        for hits in ranked_lists:
            taken = 0
            for row, score in hits:
                if taken >= per_query or len(chosen) >= budget:
                    break
                self.add_if_new(chosen, row, score, dedupe)
                taken += 1

        # Step 4b - fill whatever budget is left with the best leftovers from any query.
        leftovers = []
        for hits in ranked_lists:
            for row, score in hits:
                if row not in chosen:
                    leftovers.append((row, score))
        leftovers.sort(key=get_score, reverse=True)
        for row, score in leftovers:
            if len(chosen) >= budget:
                break
            self.add_if_new(chosen, row, score, dedupe)

        evidence = list(chosen.values())
        evidence.sort(key=evidence_score, reverse=True)

        # Step 5 - give each snippet a little of its neighbouring paragraphs.
        if expand:
            for item in evidence:
                item.text = self.expand(item.row)
        return evidence

    def add_if_new(self, chosen, row, score, dedupe):
        """Adds one snippet to the chosen set, unless it's already there or nearly a copy."""
        if row in chosen:
            return
        text = self.corpus[row].content
        if dedupe:
            already = []
            for item in chosen.values():
                already.append(item.snippet.content)
            if is_near_duplicate(text, already):
                return
        chosen[row] = Evidence(row, self.corpus[row], text, score)

    def search(self, text, k=10, **settings):
        """A single query - what plain RAG does."""
        return self.search_many([text], budget=k, per_query=k, **settings)

    # ------------------------------------------------------------------------------------
    # Neighbour expansion
    # ------------------------------------------------------------------------------------
    def expand(self, row, max_neighbour_chars=400):
        """The snippet plus the end of the paragraph before it and the start of the one after
        it - but only if they belong to the same section.

        Why: chunking cuts text mid-thought. A snippet starting "It is contraindicated in..."
        doesn't say what "it" is. We search with the small snippet (precise) and then show a
        slightly bigger window (context), capped at 400 characters on each side.
        """
        snippet = self.corpus[row]

        # Snippet ids look like "<article>_<paragraph number>", e.g. "article_17".
        prefix, underscore, number = snippet.id.rpartition("_")
        if not number.isdigit():
            return snippet.content

        parts = []
        before = self.neighbour(prefix, int(number) - 1, snippet.title)
        if before is not None:
            parts.append("..." + before.content[-max_neighbour_chars:])
        parts.append(snippet.content)
        after = self.neighbour(prefix, int(number) + 1, snippet.title)
        if after is not None:
            parts.append(after.content[:max_neighbour_chars] + "...")
        return " ".join(parts)

    def neighbour(self, prefix, number, title):
        """The snippet "<prefix>_<number>", if it exists and sits in the same section."""
        other_row = self.row_of_id.get(f"{prefix}_{number}")
        if other_row is None:
            return None
        if self.corpus.titles[other_row] != title:
            return None          # a different section: don't glue unrelated text together
        return self.corpus[other_row]


def evidence_score(item):
    # Used for sorting Evidence objects by score.
    return item.score


def load_retriever(index_dir, device=None, use_bm25=True, use_dense=True, use_reranker=True):
    """Loads everything scripts/build_index.py wrote into `index_dir`."""
    corpus = load_corpus_store(os.path.join(index_dir, "corpus.jsonl"))

    bm25 = None
    if use_bm25:
        bm25 = load_bm25(os.path.join(index_dir, "bm25"))

    dense = None
    if use_dense:
        dense = load_dense(os.path.join(index_dir, "medcpt"), MedCPTEncoder(device=device))

    reranker = None
    if use_reranker:
        reranker = MedCPTReranker(device=device)

    return Retriever(corpus, bm25, dense, reranker)
