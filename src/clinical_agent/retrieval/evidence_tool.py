"""
The agent's consult_reference tool: turns a short claim into a few self-contained "evidence cards".

    claim:  "Gilbert syndrome causes unconjugated hyperbilirubinemia triggered by fasting"  (about=mechanism)
    card:   StatPearls: Gilbert Syndrome, Pathophysiology
            "Gilbert syndrome results from reduced activity of UGT1A1 ... Fasting, illness and stress
             can unmask it."

Design rules, each answering a problem we measured or a requirement of the agent:
  1. Article first. A claim about a disease is first searched INSIDE that disease's own StatPearls
     article (matched by title), then across the whole library. This fixes attribution: in the
     book-scoring probe, passages about a *different* disease were counted as evidence.
  2. Section routing. Claims about the presentation go to "History and Physical", tests to
     "Evaluation", mechanisms to "Pathophysiology/Etiology", treatment to "Treatment / Management".
     "Differential diagnosis" sections are skipped: they mix many diseases.
  3. Sentences, not pages. The MedCPT cross-encoder (good at relevance, which is all it is used for
     here) picks the few sentences that address the claim. The agent reads short, targeted text.
  4. Nothing half-baked. Sentences are always whole; a sentence starting with "It/This/These..."
     brings its previous sentence (so the subject is known); a sentence ending in ":" brings the
     list that follows; every card is headed by its source and section, so it reads on its own.
  5. Quality over quantity. Below a relevance threshold nothing is returned: an honest
     "NOT FOUND" is better than loosely related filler.
  6. Corroboration. Up to two cards per claim, from different documents when possible.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

# Which section-title words each kind of claim should be looked up in.
SECTIONS_FOR = {
    "presentation": ("history and physical", "clinical", "presentation", "signs", "symptoms"),
    "tests": ("evaluation", "laboratory", "imaging", "workup", "histopathology"),
    "mechanism": ("pathophysiology", "etiology", "pathogenesis", "mechanism", "histopathology", "genetics"),
    "treatment": ("treatment", "management", "therapy", "pharmacotherapy"),
}
MIXED_SECTIONS = ("differential diagnos",)

# A sentence starting like this depends on the sentence before it.
NEEDS_PREVIOUS = re.compile(r"^(It|Its|This|These|They|Their|Them|Such|Both|The latter|The former|"
                            r"In this|In these|However|Therefore|Thus|Also)\b")

STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "with", "to", "is", "are", "by", "for",
             "as", "at", "from", "that", "this", "its", "be", "can", "may", "which", "disease",
             "syndrome", "type", "due"}


def words(text: str) -> set:
    """Content words of a text, lower-cased: "Gilbert's Syndrome" -> {"gilbert"}."""
    result = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word not in STOPWORDS and len(word) > 1:
            result.add(word)
    return result


# A full stop after these is not the end of a sentence ("see Chap. 9", "e.g. fasting").
ABBREVIATIONS = ("chap.", "fig.", "figs.", "e.g.", "i.e.", "vs.", "al.", "approx.", "dr.", "no.", "ref.", "sect.")


def split_sentences(text: str) -> List[str]:
    """Whole sentences only. Splits after . ! ? when the next sentence starts with a capital,
    a digit or a bracket, and also at line breaks (StatPearls lists are one item per line).
    A piece that ends with an abbreviation is glued back onto the next one."""
    sentences = []
    for line in text.split("\n"):
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", line.strip())
        carry = ""
        for piece in pieces:
            piece = (carry + " " + piece).strip() if carry else piece.strip()
            carry = ""
            if not piece:
                continue
            last_word = piece.split()[-1].lower()
            if last_word.endswith(ABBREVIATIONS):
                carry = piece                     # "...(Chap." -> wait for "9)."
                continue
            sentences.append(piece)
        if carry:
            sentences.append(carry)
    return sentences


class EvidenceTool:
    def __init__(self, retriever, cards_per_claim: int = 2, sentences_per_card: int = 3,
                 min_score: float = 0.0, article_first: bool = True, global_chunks: int = 4,
                 article_chunks: int = 8):
        if retriever.reranker is None:
            raise ValueError("The evidence tool needs the MedCPT cross-encoder.")
        self.retriever = retriever
        self.corpus = retriever.corpus
        self.cards_per_claim = cards_per_claim
        self.sentences_per_card = sentences_per_card
        self.min_score = min_score             # cross-encoder logit below which nothing is shown
        self.article_first = article_first
        self.global_chunks = global_chunks     # chunks from the whole-library search
        self.article_chunks = article_chunks   # chunks taken from the hypothesis's own article

        # StatPearls titles look like "Gilbert Syndrome -- Pathophysiology". Group rows by article
        # name once, so "find the article about X" is a dictionary lookup later.
        self.article_rows: Dict[str, List[int]] = {}
        for row, title in enumerate(self.corpus.titles):
            if " -- " in title:
                article = title.split(" -- ")[0].strip().lower()
                self.article_rows.setdefault(article, []).append(row)
        self.article_words = {}
        for name in self.article_rows:
            self.article_words[name] = words(name)

    # ------------------------------------------------------------------------------------
    # Finding the right chunks
    # ------------------------------------------------------------------------------------
    @lru_cache(maxsize=4096)
    def find_article(self, hypothesis: str) -> Optional[str]:
        """The StatPearls article about this hypothesis, or None.

        Exact title match first; otherwise the article whose title words overlap most with the
        hypothesis name (Jaccard >= 0.6), so "Gilbert's syndrome" finds "Gilbert Syndrome"
        but "Crigler-Najjar type II" does not grab some unrelated liver article.
        """
        name = hypothesis.lower().replace("'s", "").strip()
        if name in self.article_rows:
            return name
        target = words(name)
        if not target:
            return None
        best, best_overlap = None, 0.0
        for article, article_words in self.article_words.items():
            if not article_words:
                continue
            overlap = len(target & article_words) / len(target | article_words)
            if overlap > best_overlap:
                best, best_overlap = article, overlap
        return best if best_overlap >= 0.6 else None

    def article_candidates(self, hypothesis: str, claim: str, about: str) -> List[int]:
        """Rows from the hypothesis's own article, in the right sections, best word overlap first."""
        article = self.find_article(hypothesis) if hypothesis else None
        if article is None:
            return []
        rows = []
        for row in self.article_rows[article]:
            title = self.corpus.titles[row].lower()
            if any(mixed in title for mixed in MIXED_SECTIONS):
                continue
            rows.append(row)

        # Prefer the sections that match the kind of claim (if any such section exists).
        wanted = SECTIONS_FOR.get(about, ())
        in_section = []
        for row in rows:
            title = self.corpus.titles[row].lower()
            if any(word in title for word in wanted):
                in_section.append(row)
        if in_section:
            rows = in_section

        # Cheap pre-filter before the cross-encoder: most shared words with the claim.
        claim_words = words(claim)

        def shared_words(row):
            return len(claim_words & words(self.corpus.contents[row]))

        rows.sort(key=shared_words, reverse=True)
        return rows[:self.article_chunks]

    def global_candidates(self, claim: str) -> List[int]:
        evidence = self.retriever.search(claim, k=self.global_chunks, expand=False, dedupe=True)
        rows = []
        for item in evidence:
            if not any(mixed in item.snippet.title.lower() for mixed in MIXED_SECTIONS):
                rows.append(item.row)
        return rows

    # ------------------------------------------------------------------------------------
    # Turning chunks into self-contained cards
    # ------------------------------------------------------------------------------------
    @staticmethod
    def source_header(title: str, source: str) -> str:
        # "Gilbert Syndrome -- Pathophysiology" -> "StatPearls: Gilbert Syndrome, Pathophysiology"
        label = "StatPearls" if source == "statpearls" else "Textbook"
        return f"{label}: {title.replace(' -- ', ', ')}"

    @staticmethod
    def complete(sentences: List[str], index: int) -> List[int]:
        """The sentence at `index` plus whatever it needs to be understood on its own."""
        keep = [index]
        if index > 0 and NEEDS_PREVIOUS.match(sentences[index]):
            keep.insert(0, index - 1)              # "It is caused by..." -> what is "it"?
        if sentences[index].rstrip().endswith(":"):
            keep += list(range(index + 1, min(index + 6, len(sentences))))   # the list that follows
        return keep

    def consult(self, requests: Sequence[dict]) -> List[List[dict]]:
        """requests: [{"hypothesis", "claim", "about"}, ...] -> one list of cards per request.

        All claims of a round arrive together and are answered together (one cross-encoder
        batch), so no hypothesis is checked "first".
        """
        # 1. Candidate chunks per claim (the retriever locks its own models internally).
        chunk_lists = []
        for request in requests:
            rows = []
            if self.article_first:
                rows += self.article_candidates(request.get("hypothesis", ""), request["claim"],
                                                request.get("about", "other"))
            for row in self.global_candidates(request["claim"]):
                if row not in rows:
                    rows.append(row)
            chunk_lists.append(rows)

        # 2. Every (claim, sentence) pair of the whole round in ONE cross-encoder batch.
        pairs, owners = [], []          # owners[i] = (request index, row, sentence index)
        sentences_of = {}
        for r, (request, rows) in enumerate(zip(requests, chunk_lists)):
            for row in rows:
                if row not in sentences_of:
                    sentences_of[row] = split_sentences(self.corpus.contents[row])
                for s, sentence in enumerate(sentences_of[row]):
                    if len(sentence) < 25:          # headings and fragments
                        continue
                    pairs.append((request["claim"], sentence))
                    owners.append((r, row, s))
        with self.retriever._lock:
            scores = list(self.retriever.reranker.score(pairs)) if pairs else []

        # 3. Per claim: best sentences, grouped into at most `cards_per_claim` cards from
        #    different documents, each completed so it reads on its own.
        by_request = [[] for _ in requests]
        for (r, row, s), score in zip(owners, scores):
            by_request[r].append((float(score), row, s))

        results = []
        for scored in by_request:
            scored.sort(reverse=True)
            cards, used_documents = [], set()
            for score, row, s in scored:
                if score < self.min_score or len(cards) >= self.cards_per_claim:
                    break
                document = self.corpus.titles[row].split(" -- ")[0]
                if document in used_documents:
                    continue
                used_documents.add(document)

                # The best sentence, plus up to (sentences_per_card - 1) other good sentences
                # from the same chunk, each completed, shown in their original order.
                same_chunk = []
                for other_score, other_row, index in scored:
                    if other_row == row and other_score >= self.min_score:
                        same_chunk.append(index)
                keep = set()
                for index in same_chunk[:self.sentences_per_card]:
                    keep.update(self.complete(sentences_of[row], index))
                text = " ".join(sentences_of[row][i] for i in sorted(keep))
                cards.append({"source": self.source_header(self.corpus.titles[row], self.corpus.sources[row]),
                              "text": text, "score": round(score, 2), "doc_id": self.corpus.ids[row]})
            for card in cards:
                card["single_source"] = len(cards) == 1
            results.append(cards)
        return results
