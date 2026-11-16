"""
The agent's consult_reference tool: turns a short claim into a few self-contained "evidence cards".

    claim:  "Silicosis is caused by inhalation of crystalline silica dust"   (about = mechanism)
    card:   StatPearls: Silicosis, Introduction
            "Many industries place workers at risk of inhaling silica-containing dust..."

The rules, and why each one is there:
  1. Article first. A claim about a disease is first searched INSIDE that disease's own StatPearls
     article (found by its title), then in the whole library. In an earlier experiment, passages
     about a DIFFERENT disease were counted as evidence; looking in the right article fixes that.
  2. The right section. Presentation claims go to "History and Physical", tests to "Evaluation",
     mechanisms to "Pathophysiology", treatment to "Treatment / Management". "Differential
     diagnosis" sections are skipped - they talk about many diseases at once.
  3. Sentences, not pages. The MedCPT cross-encoder picks the few sentences that best match the
     claim, so the agent reads short, targeted text.
  4. Nothing half-baked. Sentences are always whole. One that starts with "It" or "This" brings the
     sentence before it; one that ends with ":" brings the list after it; every card starts with
     its source and section, so it makes sense on its own.
  5. Quality over quantity. If nothing scores high enough, nothing comes back - the agent sees
     "NOT FOUND", which is better than loosely related filler.
  6. Two sources. Up to two cards per claim, from two different documents when possible.
"""

import re

# Which words in a section title fit each kind of claim.
SECTIONS_FOR = {
    "presentation": ["history and physical", "clinical", "presentation", "signs", "symptoms"],
    "tests": ["evaluation", "laboratory", "imaging", "workup", "histopathology"],
    "mechanism": ["pathophysiology", "etiology", "pathogenesis", "mechanism", "histopathology", "genetics"],
    "treatment": ["treatment", "management", "therapy", "pharmacotherapy"],
}

# Sections that discuss many diseases side by side.
MIXED_SECTIONS = ["differential diagnos"]

# A sentence starting with one of these leans on the sentence before it.
LEANS_ON_PREVIOUS = ["It ", "Its ", "This ", "These ", "They ", "Their ", "Them ", "Such ", "Both ",
                     "The latter ", "The former ", "In this ", "In these ", "However", "Therefore",
                     "Thus", "Also "]

# A full stop after one of these doesn't end the sentence ("see Chap. 9", "e.g. fasting").
ABBREVIATIONS = ["chap.", "fig.", "figs.", "e.g.", "i.e.", "vs.", "al.", "approx.", "dr.", "no.", "ref.", "sect."]

# Words too common to say anything about which article we're in.
STOPWORDS = ["the", "a", "an", "of", "and", "or", "in", "on", "with", "to", "is", "are", "by", "for",
             "as", "at", "from", "that", "this", "its", "be", "can", "may", "which", "disease",
             "syndrome", "type", "due"]


def words(text):
    """The meaningful words of a text, lower-cased: "Gilbert's Syndrome" -> {"gilbert"}."""
    result = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word not in STOPWORDS and len(word) > 1:
            result.add(word)
    return result


def split_sentences(text):
    """Cuts a passage into whole sentences.

    A new sentence starts after ". ", "! " or "? " when the next character is a capital letter,
    a digit or a bracket, and at every line break (StatPearls lists put one item per line).
    If a piece ends with an abbreviation like "Chap.", the sentence isn't over yet, so we glue
    it onto the next piece.
    """
    sentences = []
    for line in text.split("\n"):
        # (?<=[.!?]) means "right after . ! or ?", and (?=[A-Z0-9(\[]) means "right before a
        # capital, digit or bracket". So we only split at real sentence boundaries.
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
            ends_with_abbreviation = False
            for abbreviation in ABBREVIATIONS:
                if last_word.endswith(abbreviation):
                    ends_with_abbreviation = True
            if ends_with_abbreviation:
                unfinished = piece
                continue

            sentences.append(piece)

        if unfinished:
            sentences.append(unfinished)
    return sentences


def complete_sentence(sentences, index):
    """The positions of the sentence at `index` plus whatever it needs to make sense alone."""
    keep = [index]

    # "It is caused by..." - we need the sentence before it to know what "it" is.
    if index > 0:
        for start in LEANS_ON_PREVIOUS:
            if sentences[index].startswith(start):
                keep.insert(0, index - 1)
                break

    # "Causes include:" is useless without the list that follows it.
    if sentences[index].rstrip().endswith(":"):
        last = min(index + 6, len(sentences))
        for following in range(index + 1, last):
            keep.append(following)

    return keep


def source_header(title, source):
    """'Gilbert Syndrome -- Pathophysiology'  ->  'StatPearls: Gilbert Syndrome, Pathophysiology'"""
    if source == "statpearls":
        label = "StatPearls"
    else:
        label = "Textbook"
    return label + ": " + title.replace(" -- ", ", ")


def is_mixed_section(title):
    for mixed in MIXED_SECTIONS:
        if mixed in title.lower():
            return True
    return False


class EvidenceTool:
    def __init__(self, retriever, cards_per_claim=2, sentences_per_card=3, min_score=0.0,
                 article_first=True, global_chunks=4, article_chunks=8):
        if retriever.reranker is None:
            raise ValueError("The evidence tool needs the MedCPT cross-encoder.")
        self.retriever = retriever
        self.corpus = retriever.corpus
        self.cards_per_claim = cards_per_claim
        self.sentences_per_card = sentences_per_card
        self.min_score = min_score           # below this score we'd rather say NOT FOUND
        self.article_first = article_first
        self.global_chunks = global_chunks   # chunks taken from the whole-library search
        self.article_chunks = article_chunks # chunks taken from the hypothesis's own article

        # StatPearls titles look like "Gilbert Syndrome -- Pathophysiology". We group all rows
        # by article name once, so "find the article about X" is just a dictionary lookup later.
        self.article_rows = {}
        for row in range(len(self.corpus.titles)):
            title = self.corpus.titles[row]
            if " -- " in title:
                article = title.split(" -- ")[0].strip().lower()
                if article not in self.article_rows:
                    self.article_rows[article] = []
                self.article_rows[article].append(row)

        self.article_words = {}
        for article in self.article_rows:
            self.article_words[article] = words(article)

        self.article_for_name = {}           # remembers answers of find_article(); names repeat a lot

    # ------------------------------------------------------------------------------------
    # Step 1: which chunks might hold the answer to a claim
    # ------------------------------------------------------------------------------------
    def find_article(self, hypothesis):
        """The StatPearls article about this hypothesis, or None.

        An exact title match wins. Otherwise we take the title that shares the most words with
        the name - but only if the overlap is big (at least 60%), so "Gilbert's syndrome" finds
        "Gilbert Syndrome" while "Crigler-Najjar type II" doesn't grab a random liver article.
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
            for article in self.article_words:
                article_words = self.article_words[article]
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

    def chunks_from_own_article(self, hypothesis, claim, about):
        """Chunks from the hypothesis's own article, preferring the section that fits the claim."""
        if not hypothesis:
            return []
        article = self.find_article(hypothesis)
        if article is None:
            return []

        rows = []
        for row in self.article_rows[article]:
            if not is_mixed_section(self.corpus.titles[row]):
                rows.append(row)

        # A claim about a lab test should be checked in "Evaluation", not in "Treatment".
        wanted = SECTIONS_FOR.get(about, [])
        in_right_section = []
        for row in rows:
            title = self.corpus.titles[row].lower()
            for word in wanted:
                if word in title:
                    in_right_section.append(row)
                    break
        if in_right_section:
            rows = in_right_section

        # An article can have 50+ chunks. The cross-encoder only needs the likeliest few:
        # the ones sharing the most words with the claim.
        claim_words = words(claim)
        scored_rows = []
        for row in rows:
            shared = len(claim_words & words(self.corpus.contents[row]))
            scored_rows.append((shared, row))
        scored_rows.sort(reverse=True)

        best_rows = []
        for shared, row in scored_rows[:self.article_chunks]:
            best_rows.append(row)
        return best_rows

    def chunks_from_whole_library(self, claim):
        evidence = self.retriever.search(claim, k=self.global_chunks, expand=False, dedupe=True)
        rows = []
        for item in evidence:
            if not is_mixed_section(item.snippet.title):
                rows.append(item.row)
        return rows

    # ------------------------------------------------------------------------------------
    # Step 3 helper: one card that reads on its own
    # ------------------------------------------------------------------------------------
    def make_card(self, row, sentences, best_indexes, score):
        keep = set()
        for index in best_indexes[:self.sentences_per_card]:
            for position in complete_sentence(sentences, index):
                keep.add(position)

        # Put the sentences back in their original order, so the card reads like the source.
        parts = []
        for position in sorted(keep):
            parts.append(sentences[position])

        return {
            "source": source_header(self.corpus.titles[row], self.corpus.sources[row]),
            "text": " ".join(parts),
            "score": round(score, 2),
            "doc_id": self.corpus.ids[row],
        }

    # ------------------------------------------------------------------------------------
    # The tool itself
    # ------------------------------------------------------------------------------------
    def consult(self, requests):
        """requests: [{"hypothesis": ..., "claim": ..., "about": ...}, ...]
        returns:  one list of cards for each request, in the same order.

        All the claims of one turn arrive together and are scored together, so no hypothesis
        gets looked up "first".
        """
        # Step 1 - candidate chunks for every claim: its own article first, then the whole library.
        chunks_for = []
        for request in requests:
            rows = []
            if self.article_first:
                rows = self.chunks_from_own_article(request.get("hypothesis", ""), request["claim"],
                                                    request.get("about", "other"))
            for row in self.chunks_from_whole_library(request["claim"]):
                if row not in rows:
                    rows.append(row)
            chunks_for.append(rows)

        # Step 2 - every (claim, sentence) pair of the whole turn goes to the cross-encoder in one go.
        pairs = []
        where = []               # where[i] says which request, chunk and sentence pairs[i] came from
        sentences_of = {}        # chunk row -> its sentences (cut once, reused)
        for request_number in range(len(requests)):
            claim = requests[request_number]["claim"]
            for row in chunks_for[request_number]:
                if row not in sentences_of:
                    sentences_of[row] = split_sentences(self.corpus.contents[row])
                for index in range(len(sentences_of[row])):
                    sentence = sentences_of[row][index]
                    if len(sentence) < 25:
                        continue          # headings and scraps aren't evidence
                    pairs.append((claim, sentence))
                    where.append((request_number, row, index))

        scores = []
        if pairs:
            with self.retriever.lock:     # the model is shared between benchmark threads
                scores = self.retriever.reranker.score(pairs)

        scored_for = []          # scored_for[request_number] = [(score, row, sentence index), ...]
        for request in requests:
            scored_for.append([])
        for i in range(len(where)):
            request_number, row, index = where[i]
            scored_for[request_number].append((float(scores[i]), row, index))

        # Step 3 - for each claim, its best sentences become at most `cards_per_claim` cards,
        # each from a different document, so two cards really are two sources.
        results = []
        for scored in scored_for:
            scored.sort(reverse=True)
            cards = []
            documents_used = []

            for score, row, index in scored:
                if score < self.min_score or len(cards) >= self.cards_per_claim:
                    break
                document = self.corpus.titles[row].split(" -- ")[0]
                if document in documents_used:
                    continue
                documents_used.append(document)

                # The best few sentences of this same chunk, best first.
                best_indexes = []
                for other_score, other_row, other_index in scored:
                    if other_row == row and other_score >= self.min_score:
                        best_indexes.append(other_index)
                cards.append(self.make_card(row, sentences_of[row], best_indexes, score))

            for card in cards:
                card["single_source"] = (len(cards) == 1)
            results.append(cards)
        return results
