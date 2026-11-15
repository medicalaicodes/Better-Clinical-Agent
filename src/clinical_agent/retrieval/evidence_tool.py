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
    """Cuts a passage into whole sentences.

    A new sentence starts after . ! or ? when the next word begins with a capital, a digit or a
    bracket, and at every line break (StatPearls lists put one item per line). A full stop after
    an abbreviation like "Chap." doesn't count - we glue that piece onto the next one.
    """
    sentences = []
    for line in text.split("\n"):
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", line.strip())

        unfinished = ""
        for piece in pieces:
            piece = piece.strip()
            if unfinished:
                piece = unfinished + " " + piece
                unfinished = ""
            if not piece:
                continue

            last_word = piece.split()[-1].lower()
            if last_word.endswith(ABBREVIATIONS):
                unfinished = piece         # "...see (Chap." - the sentence isn't over yet
                continue
            sentences.append(piece)

        if unfinished:
            sentences.append(unfinished)
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
        self.min_score = min_score             # below this cross-encoder score we'd rather say NOT FOUND
        self.article_first = article_first
        self.global_chunks = global_chunks     # chunks taken from the whole-library search
        self.article_chunks = article_chunks   # chunks taken from the hypothesis's own article

        # StatPearls titles look like "Gilbert Syndrome -- Pathophysiology". Grouping the rows by
        # article name once means "find the article about X" is just a dictionary lookup later.
        self.article_rows: Dict[str, List[int]] = {}
        for row, title in enumerate(self.corpus.titles):
            if " -- " in title:
                article = title.split(" -- ")[0].strip().lower()
                if article not in self.article_rows:
                    self.article_rows[article] = []
                self.article_rows[article].append(row)

        self.article_words = {}
        for article in self.article_rows:
            self.article_words[article] = words(article)

        self.article_for_name = {}            # remembers find_article() answers; names repeat a lot

    # ------------------------------------------------------------------------------------
    # Step 1: which chunks might hold the answer to a claim
    # ------------------------------------------------------------------------------------
    def find_article(self, hypothesis: str) -> Optional[str]:
        """The StatPearls article about this hypothesis, or None.

        An exact title match wins. Otherwise we take the title sharing the most words with the
        name, but only if the overlap is big (Jaccard >= 0.6) - so "Gilbert's syndrome" finds
        "Gilbert Syndrome", while "Crigler-Najjar type II" doesn't grab some random liver article.
        """
        if hypothesis in self.article_for_name:
            return self.article_for_name[hypothesis]

        name = hypothesis.lower().replace("'s", "").strip()
        best = None
        if name in self.article_rows:
            best = name
        else:
            target = words(name)
            best_overlap = 0.0
            for article, article_words in self.article_words.items():
                if not target or not article_words:
                    continue
                overlap = len(target & article_words) / len(target | article_words)
                if overlap > best_overlap:
                    best = article
                    best_overlap = overlap
            if best_overlap < 0.6:
                best = None

        self.article_for_name[hypothesis] = best
        return best

    def chunks_from_own_article(self, hypothesis: str, claim: str, about: str) -> List[int]:
        """Chunks from the hypothesis's own article, preferring the section that fits the claim."""
        if not hypothesis:
            return []
        article = self.find_article(hypothesis)
        if article is None:
            return []

        rows = []
        for row in self.article_rows[article]:
            title = self.corpus.titles[row].lower()
            if any(mixed in title for mixed in MIXED_SECTIONS):
                continue                       # differential-diagnosis sections talk about everything
            rows.append(row)

        # A claim about a lab test should be checked in "Evaluation", not in "Treatment".
        wanted = SECTIONS_FOR.get(about, ())
        in_right_section = []
        for row in rows:
            title = self.corpus.titles[row].lower()
            if any(word in title for word in wanted):
                in_right_section.append(row)
        if in_right_section:
            rows = in_right_section

        # Articles can have 50+ chunks; the cross-encoder only needs to see the likeliest few.
        claim_words = words(claim)

        def shared_words(row):
            return len(claim_words & words(self.corpus.contents[row]))

        rows.sort(key=shared_words, reverse=True)
        return rows[:self.article_chunks]

    def chunks_from_whole_library(self, claim: str) -> List[int]:
        evidence = self.retriever.search(claim, k=self.global_chunks, expand=False, dedupe=True)
        rows = []
        for item in evidence:
            if any(mixed in item.snippet.title.lower() for mixed in MIXED_SECTIONS):
                continue
            rows.append(item.row)
        return rows

    # ------------------------------------------------------------------------------------
    # Step 3 helpers: making a card that reads on its own
    # ------------------------------------------------------------------------------------
    @staticmethod
    def source_header(title: str, source: str) -> str:
        # "Gilbert Syndrome -- Pathophysiology"  ->  "StatPearls: Gilbert Syndrome, Pathophysiology"
        if source == "statpearls":
            label = "StatPearls"
        else:
            label = "Textbook"
        return f"{label}: {title.replace(' -- ', ', ')}"

    @staticmethod
    def complete(sentences: List[str], index: int) -> List[int]:
        """The sentence at `index`, plus whatever it needs to make sense by itself."""
        keep = [index]
        if index > 0 and NEEDS_PREVIOUS.match(sentences[index]):
            keep.insert(0, index - 1)          # "It is caused by..." - we need to know what "it" is
        if sentences[index].rstrip().endswith(":"):
            last = min(index + 6, len(sentences))
            for following in range(index + 1, last):
                keep.append(following)         # "Causes include:" is useless without the list
        return keep

    def make_card(self, row: int, sentences: List[str], best_indexes: List[int], score: float) -> dict:
        keep = set()
        for index in best_indexes[:self.sentences_per_card]:
            keep.update(self.complete(sentences, index))

        # Back in the order they were written, so the card reads like the original text.
        text = " ".join(sentences[i] for i in sorted(keep))
        return {
            "source": self.source_header(self.corpus.titles[row], self.corpus.sources[row]),
            "text": text,
            "score": round(score, 2),
            "doc_id": self.corpus.ids[row],
        }

    # ------------------------------------------------------------------------------------
    # The tool itself
    # ------------------------------------------------------------------------------------
    def consult(self, requests: Sequence[dict]) -> List[List[dict]]:
        """requests: [{"hypothesis", "claim", "about"}, ...]  ->  a list of cards for each request.

        All the claims of one turn come in together and get scored together, so no hypothesis
        is looked up "first".
        """
        # Step 1 - candidate chunks for each claim: its own article first, then the whole library.
        chunks_for = []
        for request in requests:
            rows = []
            if self.article_first:
                rows += self.chunks_from_own_article(request.get("hypothesis", ""), request["claim"],
                                                     request.get("about", "other"))
            for row in self.chunks_from_whole_library(request["claim"]):
                if row not in rows:
                    rows.append(row)
            chunks_for.append(rows)

        # Step 2 - score every (claim, sentence) pair of the turn in one cross-encoder batch.
        pairs = []
        where = []                             # where[i] = (which request, chunk row, sentence index)
        sentences_of = {}
        for request_number, request in enumerate(requests):
            for row in chunks_for[request_number]:
                if row not in sentences_of:
                    sentences_of[row] = split_sentences(self.corpus.contents[row])
                for index, sentence in enumerate(sentences_of[row]):
                    if len(sentence) < 25:
                        continue               # headings and scraps aren't evidence
                    pairs.append((request["claim"], sentence))
                    where.append((request_number, row, index))

        scores = []
        if pairs:
            with self.retriever._lock:          # the models are shared between benchmark threads
                scores = list(self.retriever.reranker.score(pairs))

        scored_for = []
        for _ in requests:
            scored_for.append([])
        for (request_number, row, index), score in zip(where, scores):
            scored_for[request_number].append((float(score), row, index))

        # Step 3 - for each claim, its best sentences become at most `cards_per_claim` cards,
        # each from a different document, so two cards mean two independent sources.
        results = []
        for scored in scored_for:
            scored.sort(reverse=True)
            cards = []
            documents_used = set()

            for score, row, index in scored:
                if score < self.min_score or len(cards) >= self.cards_per_claim:
                    break
                document = self.corpus.titles[row].split(" -- ")[0]
                if document in documents_used:
                    continue
                documents_used.add(document)

                # The best few sentences of this same chunk, best first.
                best_indexes = []
                for other_score, other_row, other_index in scored:
                    if other_row == row and other_score >= self.min_score:
                        best_indexes.append(other_index)
                cards.append(self.make_card(row, sentences_of[row], best_indexes, score))

            for card in cards:
                card["single_source"] = len(cards) == 1
            results.append(cards)
        return results
