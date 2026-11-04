"""
The three search components:

    BM25Index       keyword search    (exact words: drug names, eponyms, lab names)
    DenseIndex      meaning search    (MedCPT embeddings + FAISS: catches paraphrases)
    MedCPTReranker  re-scoring        (reads query + passage together; slow, so used on a short list)

MedCPT (Jin et al., Bioinformatics 2023) was trained by NCBI on 255M real PubMed search
logs, which makes it a strong off-the-shelf biomedical retriever. We use it as is (no training).
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# A ranked result list: [(row in the corpus, score), ...], best first.
Hits = List[Tuple[int, float]]


def pick_device(device: Optional[str]) -> str:
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------------------------
# BM25: keyword search
# ----------------------------------------------------------------------------------------

class BM25Index:
    """Thin wrapper over `bm25s` (BM25 with sparse matrices, ~100x faster than rank_bm25).

    Text is lower-cased, English stopwords removed and words stemmed
    ("embolism"/"embolic" -> "embol"), both for the documents and the queries.
    """

    def __init__(self, retriever=None):
        self._retriever = retriever
        self._stemmer = None

    def _get_stemmer(self):
        if self._stemmer is None:
            import Stemmer
            self._stemmer = Stemmer.Stemmer("english")
        return self._stemmer

    def build(self, texts: Sequence[str]) -> "BM25Index":
        import bm25s
        tokens = bm25s.tokenize(list(texts), stopwords="en", stemmer=self._get_stemmer(),
                                show_progress=True)
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=True)
        return self

    def save(self, path: str) -> None:
        self._retriever.save(path)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        import bm25s
        return cls(bm25s.BM25.load(path))

    def search(self, queries: Sequence[str], k: int) -> List[Hits]:
        import bm25s

        # return_ids=False gives the query words as strings; bm25s maps them onto the
        # index vocabulary itself (words it has never seen are simply ignored).
        query_tokens = bm25s.tokenize(list(queries), stopwords="en", stemmer=self._get_stemmer(),
                                      return_ids=False, show_progress=False)
        n_docs = self._retriever.scores["num_docs"]

        results: List[Hits] = []
        for tokens in query_tokens:
            if not tokens:                    # the query was nothing but stopwords
                results.append([])
                continue

            doc_rows, scores = self._retriever.retrieve([tokens], k=min(k, n_docs), show_progress=False)

            hits = []
            for row, score in zip(doc_rows[0], scores[0]):
                if score > 0:                 # score 0 = shares no word with the query
                    hits.append((int(row), float(score)))
            results.append(hits)
        return results


# ----------------------------------------------------------------------------------------
# MedCPT dense retrieval
# ----------------------------------------------------------------------------------------

class MedCPTEncoder:
    """MedCPT has two separate encoders ("towers"): one for short queries, one for articles.
    Both use the [CLS] token's vector as the text embedding (768 numbers)."""

    QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, device: Optional[str] = None, query_max_len: int = 256):
        self.device = pick_device(device)
        # MedCPT was trained on short queries (~64 tokens). We allow 256 so whole
        # questions fit for the vanilla-RAG baseline; DDx-RAG's own queries are short anyway.
        self.query_max_len = query_max_len
        self._query_model = None      # (tokenizer, model), loaded on first use
        self._article_model = None

    def _load(self, name: str):
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(self.device).eval()
        if self.device.startswith("cuda"):
            model = model.half()          # fp16 on GPU: 2x faster, no measurable quality loss
        return tokenizer, model

    def encode_queries(self, queries: Sequence[str], batch_size: int = 64) -> np.ndarray:
        import torch

        if self._query_model is None:
            self._query_model = self._load(self.QUERY_MODEL)
        tokenizer, model = self._query_model

        batches = []
        for start in range(0, len(queries), batch_size):
            batch = list(queries[start:start + batch_size])
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=self.query_max_len,
                                return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls_vectors = model(**encoded).last_hidden_state[:, 0, :]
            batches.append(cls_vectors.float().cpu().numpy())

        if not batches:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(batches)

    def encode_articles(self, titles: Sequence[str], contents: Sequence[str], batch_size: int = 128,
                        show_progress: bool = True) -> np.ndarray:
        import torch
        from tqdm import tqdm

        if self._article_model is None:
            self._article_model = self._load(self.ARTICLE_MODEL)
        tokenizer, model = self._article_model

        batches = []
        starts = range(0, len(titles), batch_size)
        for start in tqdm(starts, desc="MedCPT encode", disable=not show_progress):
            # The article encoder takes (title, text) pairs, exactly as in MedCPT's training.
            pairs = []
            for title, content in zip(titles[start:start + batch_size], contents[start:start + batch_size]):
                pairs.append([title, content])

            encoded = tokenizer(pairs, truncation=True, padding=True, max_length=512,
                                return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls_vectors = model(**encoded).last_hidden_state[:, 0, :]
            batches.append(cls_vectors.float().cpu().numpy())

        return np.concatenate(batches)


class DenseIndex:
    """Exact nearest-neighbour search over MedCPT article embeddings with FAISS.

    IndexFlatIP = brute-force inner product. For ~427k vectors that takes a few milliseconds
    per query, so approximate indexes (HNSW, IVF) would lose accuracy for no real speed-up.
    Inner product (not cosine) because that is what MedCPT was trained with.
    """

    def __init__(self, index=None, encoder: Optional[MedCPTEncoder] = None):
        self._index = index
        self._embeddings = None
        self.encoder = encoder or MedCPTEncoder()

    def build(self, titles: Sequence[str], contents: Sequence[str], batch_size: int = 128) -> "DenseIndex":
        embeddings = self.encoder.encode_articles(titles, contents, batch_size=batch_size)
        self._set_embeddings(embeddings)
        return self

    def _set_embeddings(self, embeddings: np.ndarray) -> None:
        import faiss
        self._embeddings = embeddings
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # Stored as fp16 to halve the disk size (~0.65 GB); converted back to fp32 on load.
        np.save(os.path.join(path, "embeddings.f16.npy"), self._embeddings.astype(np.float16))

    @classmethod
    def load(cls, path: str, encoder: Optional[MedCPTEncoder] = None) -> "DenseIndex":
        index = cls(encoder=encoder)
        embeddings = np.load(os.path.join(path, "embeddings.f16.npy")).astype(np.float32)
        index._set_embeddings(embeddings)
        return index

    def search(self, queries: Sequence[str], k: int) -> List[Hits]:
        if not queries:
            return []

        query_vectors = self.encoder.encode_queries(queries)
        k = min(k, self._index.ntotal)
        scores, rows = self._index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

        results: List[Hits] = []
        for row_list, score_list in zip(rows, scores):
            hits = []
            for row, score in zip(row_list, score_list):
                if row >= 0:                  # FAISS pads with -1 when it has fewer than k
                    hits.append((int(row), float(score)))
            results.append(hits)
        return results


# ----------------------------------------------------------------------------------------
# Cross-encoder reranking
# ----------------------------------------------------------------------------------------

class MedCPTReranker:
    """MedCPT cross-encoder. Unlike the bi-encoder, it reads the query and the passage
    TOGETHER, so it can tell "D-dimer rules OUT PE" from "D-dimer is raised in PE".
    Too slow for 427k passages, fine for the ~30 candidates per query we give it.
    Output is a relevance logit: higher = more relevant (not a probability)."""

    MODEL = "ncbi/MedCPT-Cross-Encoder"

    def __init__(self, device: Optional[str] = None, max_len: int = 512):
        self.device = pick_device(device)
        self.max_len = max_len
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).to(self.device).eval()
        if self.device.startswith("cuda"):
            self._model = self._model.half()

    def score(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()

        batches = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            queries = [query for query, _ in batch]
            passages = [passage for _, passage in batch]

            encoded = self._tokenizer(queries, passages, truncation=True, padding=True,
                                      max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1)
            batches.append(logits.float().cpu().numpy())

        if not batches:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(batches)
