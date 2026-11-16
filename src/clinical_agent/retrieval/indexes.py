"""
The three search engines the retriever combines:

    BM25Index       keyword search    - exact words: drug names, eponyms, lab names
    DenseIndex      meaning search    - MedCPT embeddings + FAISS: finds paraphrases
    MedCPTReranker  re-scoring        - reads query and passage together; slow, so only used
                                        on a short list of candidates

MedCPT (Jin et al., Bioinformatics 2023) was trained by NCBI on 255 million real PubMed search
logs, which makes it a strong biomedical retriever out of the box. We use it as is - no training.

A search result everywhere in this package is a list of (row, score) pairs, best first,
where "row" is the snippet's position in the CorpusStore.
"""

import os

import numpy as np


def pick_device(device):
    """Use the device we were told to, otherwise the GPU if there is one, otherwise the CPU."""
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------------------------
# BM25 - keyword search
# ----------------------------------------------------------------------------------------

class BM25Index:
    """A thin wrapper around the `bm25s` library (BM25 on sparse matrices, very fast).

    Documents and queries are treated the same way: lower-cased, common English words
    ("the", "of") removed, and every word cut to its stem ("embolism" -> "embol").
    """

    def __init__(self, retriever=None):
        self.retriever = retriever          # the bm25s object; None until built or loaded
        self.stemmer = None

    def get_stemmer(self):
        if self.stemmer is None:
            import Stemmer
            self.stemmer = Stemmer.Stemmer("english")
        return self.stemmer

    def build(self, texts):
        import bm25s
        tokens = bm25s.tokenize(list(texts), stopwords="en", stemmer=self.get_stemmer(), show_progress=True)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=True)
        return self

    def save(self, path):
        self.retriever.save(path)

    def search(self, queries, k):
        import bm25s

        # return_ids=False gives the query words as plain strings; bm25s then looks them up in
        # its own vocabulary and simply ignores words it has never seen.
        query_tokens = bm25s.tokenize(list(queries), stopwords="en", stemmer=self.get_stemmer(),
                                      return_ids=False, show_progress=False)
        number_of_documents = self.retriever.scores["num_docs"]

        results = []
        for tokens in query_tokens:
            if not tokens:                  # the query was nothing but stop-words
                results.append([])
                continue

            rows, scores = self.retriever.retrieve([tokens], k=min(k, number_of_documents), show_progress=False)

            hits = []
            for row, score in zip(rows[0], scores[0]):
                if score > 0:               # score 0 means it shares no word with the query
                    hits.append((int(row), float(score)))
            results.append(hits)
        return results


def load_bm25(path):
    import bm25s
    return BM25Index(bm25s.BM25.load(path))


# ----------------------------------------------------------------------------------------
# MedCPT dense retrieval
# ----------------------------------------------------------------------------------------

class MedCPTEncoder:
    """MedCPT has two separate encoders: one for short queries, one for articles.
    Both use the vector of the first ([CLS]) token as the embedding: 768 numbers per text."""

    QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, device=None, query_max_len=256):
        self.device = pick_device(device)
        # MedCPT was trained on short queries; 256 tokens is enough for a whole MedQA question.
        self.query_max_len = query_max_len
        self.query_model = None             # (tokenizer, model), loaded the first time it's needed
        self.article_model = None

    def load_model(self, name):
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(self.device).eval()
        if self.device.startswith("cuda"):
            model = model.half()            # half precision on the GPU: twice as fast, same results
        return tokenizer, model

    def encode_queries(self, queries, batch_size=64):
        import torch

        if self.query_model is None:
            self.query_model = self.load_model(self.QUERY_MODEL)
        tokenizer, model = self.query_model

        batches = []
        for start in range(0, len(queries), batch_size):
            batch = list(queries[start:start + batch_size])
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=self.query_max_len,
                                return_tensors="pt").to(self.device)
            with torch.no_grad():
                first_token_vectors = model(**encoded).last_hidden_state[:, 0, :]
            batches.append(first_token_vectors.float().cpu().numpy())

        if not batches:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(batches)

    def encode_articles(self, titles, contents, batch_size=128):
        import torch
        from tqdm import tqdm

        if self.article_model is None:
            self.article_model = self.load_model(self.ARTICLE_MODEL)
        tokenizer, model = self.article_model

        batches = []
        for start in tqdm(range(0, len(titles), batch_size), desc="MedCPT encode"):
            # The article encoder takes [title, text] pairs - exactly how MedCPT was trained.
            pairs = []
            for title, content in zip(titles[start:start + batch_size], contents[start:start + batch_size]):
                pairs.append([title, content])

            encoded = tokenizer(pairs, truncation=True, padding=True, max_length=512,
                                return_tensors="pt").to(self.device)
            with torch.no_grad():
                first_token_vectors = model(**encoded).last_hidden_state[:, 0, :]
            batches.append(first_token_vectors.float().cpu().numpy())

        return np.concatenate(batches)


class DenseIndex:
    """Exact nearest-neighbour search over the MedCPT article vectors, with FAISS.

    IndexFlatIP compares the query with every single vector (inner product). For about 427k
    vectors that takes a few milliseconds, so an approximate index (HNSW, IVF) would only lose
    accuracy without saving any real time. Inner product, not cosine, because that's what
    MedCPT was trained with.
    """

    def __init__(self, encoder=None):
        self.encoder = encoder or MedCPTEncoder()
        self.embeddings = None
        self.index = None

    def set_embeddings(self, embeddings):
        import faiss
        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def build(self, titles, contents, batch_size=128):
        self.set_embeddings(self.encoder.encode_articles(titles, contents, batch_size=batch_size))
        return self

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Saved in 16-bit to halve the file (~0.65 GB); turned back into 32-bit when loaded.
        np.save(os.path.join(path, "embeddings.f16.npy"), self.embeddings.astype(np.float16))

    def search(self, queries, k):
        if not queries:
            return []

        query_vectors = self.encoder.encode_queries(queries)
        k = min(k, self.index.ntotal)
        scores, rows = self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

        results = []
        for row_list, score_list in zip(rows, scores):
            hits = []
            for row, score in zip(row_list, score_list):
                if row >= 0:                # FAISS pads with -1 when it has fewer than k results
                    hits.append((int(row), float(score)))
            results.append(hits)
        return results


def load_dense(path, encoder=None):
    index = DenseIndex(encoder)
    embeddings = np.load(os.path.join(path, "embeddings.f16.npy")).astype(np.float32)
    index.set_embeddings(embeddings)
    return index


# ----------------------------------------------------------------------------------------
# Cross-encoder re-ranking
# ----------------------------------------------------------------------------------------

class MedCPTReranker:
    """MedCPT's cross-encoder. Unlike the encoders above, it reads the query and the passage
    TOGETHER, so it can tell "D-dimer rules OUT embolism" from "D-dimer is raised in embolism".
    Far too slow for 427k passages, but fine for the ~30 candidates per query we give it.
    It returns a relevance score (higher = more relevant), not a probability."""

    MODEL = "ncbi/MedCPT-Cross-Encoder"

    def __init__(self, device=None, max_len=512):
        self.device = pick_device(device)
        self.max_len = max_len
        self.tokenizer = None
        self.model = None

    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.half()

    def score(self, pairs, batch_size=64):
        """pairs: [(query, passage), ...]  ->  one score per pair."""
        import torch

        if self.model is None:
            self.load()

        batches = []
        for start in range(0, len(pairs), batch_size):
            queries = []
            passages = []
            for query, passage in pairs[start:start + batch_size]:
                queries.append(query)
                passages.append(passage)

            encoded = self.tokenizer(queries, passages, truncation=True, padding=True,
                                     max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits.squeeze(-1)
            batches.append(logits.float().cpu().numpy())

        if not batches:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(batches)
