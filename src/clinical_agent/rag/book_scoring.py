"""
Book scoring: the library judges the model's candidate answers; the model never reads it.

Why this exists (what the MIRAGE runs showed):
  - When the 5 closed-book votes agree, they are right ~89% of the time, and every RAG
    method we tried pulls that accuracy DOWN (the model reads on-topic text and gets misled).
  - When the votes are split, the right answer is usually still among the candidates;
    the failure is choosing between them.

So instead of pasting passages into the prompt, we use Bayes' rule, the way diagnosis is
taught to medical students:

    posterior(candidate)  ∝  prior(candidate)       x  likelihood(case | candidate)
                             = share of the 5 votes   = how well the textbook description
                                                        of the candidate fits the case

The likelihood comes from the MedCPT cross-encoder, a small model that only outputs a
relevance number. It cannot be "talked into" anything, which is the whole point.

This file has two halves:
  1. FEATURES (needs the retriever, GPU recommended): for every candidate, fetch its textbook
     passages and score them against the case, finding by finding.
  2. COMBINER (CPU, seconds): a conditional logit - softmax over a question's candidates of
     w . features - fitted on MedQA *training* questions, plus a tiny MLP as an ablation.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

# A deliberately empty "patient". Some passages score high against ANY clinical text
# (generic overviews, very common diseases). Subtracting the score a passage gets for this
# neutral query keeps only the part that is specific to THIS patient - the same idea as
# "calibrate before use" (Zhao et al., 2021) for LLM prompts.
NULL_QUERY = "A patient presents to the clinic."

# Sections that list many diseases side by side. A good match there says nothing about
# WHICH disease the passage supports, so these are skipped.
MIXED_SECTIONS = ("differential diagnos",)


# ----------------------------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------------------------

def split_case(question: str, max_sentences: int = 12) -> List[str]:
    """Splits a vignette into its 'findings' (sentences), dropping the question sentence.

    "A 45-year-old man has fever. His pulse is 120/min. Which drug ...?"
      -> ["A 45-year-old man has fever.", "His pulse is 120/min."]

    Short sentences are what MedCPT was trained on (PubMed search queries are a few words),
    so scoring finding by finding keeps the cross-encoder in its comfort zone. A short
    question with no vignette (typical for MMLU) is kept whole as a single "finding".
    """
    pieces = []
    for line in question.split("\n"):
        # Split after . ! ? when the next sentence starts with a capital letter or a digit.
        pieces += re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line.strip())

    findings = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece) < 15:          # "Hb% 10 gm/dL" fragments carry too little on their own
            continue
        if piece.endswith("?") or piece.endswith('?"'):
            continue                 # the question itself is not a finding
        findings.append(piece)

    if not findings:
        return [question.strip()]
    return findings[:max_sentences]


def question_type(question: str) -> str:
    """Rough type of a MedQA question from its last sentence (for per-type reporting only).

    Book scoring only makes sense when the answer is something a textbook describes
    (a disease, an organism, a structure). For "next step in management" or ethics
    questions we expect it to help little - this tag lets us check that.
    """
    tail = question.strip()[-300:].lower()
    if re.search(r"ethic|tell the patient|most appropriate response|sensitivity|specificity|odds ratio|"
                 r"relative risk|study design|\bbias\b|confound|number needed", tail):
        return "ethics_stats"
    if re.search(r"next (best )?step|most appropriate (next|initial|management|treatment|pharmacotherapy|therapy)|"
                 r"best (initial )?(treatment|management)|should be (administered|given|prescribed)", tail):
        return "management"
    if re.search(r"most likely diagnosis|diagnosis is most likely|most likely cause of (this|his|her)|"
                 r"most likely causal organism|patient most likely has", tail):
        return "diagnosis"
    if re.search(r"mechanism|pathophysiolog|pathogenesis|most likely (responsible|involved)|adverse effect|"
                 r"side effect|deficien", tail):
        return "mechanism"
    return "other"


def vote_shares(votes: Dict[str, int], letters: Sequence[str]) -> Dict[str, float]:
    """Turns {"B": 3, "C": 2} into smoothed shares over ALL options.

    The +0.5 (a Jeffreys-style pseudo-count) keeps an option with 0 votes possible:
    log(0) would make the book unable to ever rescue it.
    """
    counts = {letter: votes.get(letter, 0) for letter in letters}
    total = sum(counts.values()) + 0.5 * len(letters)
    return {letter: (counts[letter] + 0.5) / total for letter in letters}


# ----------------------------------------------------------------------------------------
# Part 1: features (the "likelihood" side)
# ----------------------------------------------------------------------------------------

class BookScorer:
    """Scores how well each candidate's textbook passages fit the case.

    For each candidate:
      1. Retrieve its passages with the candidate text as the query ("Hemolytic uremic
         syndrome" -> passages about HUS), skipping mixed "differential diagnosis" sections.
      2. Cross-encode every (finding, passage) pair, minus the passage's NULL_QUERY score.
      3. For each finding keep the best passage, then average over findings:
             fit = mean over findings of  max over passages of  [CE(finding, p) - CE(null, p)]
         This is the naive-Bayes picture: each finding contributes its own evidence.
    We also keep the whole-case score (one long query) to compare the two styles.
    """

    def __init__(self, retriever, passages_per_candidate: int = 8, pool: int = 30):
        if retriever.reranker is None:
            raise ValueError("Book scoring needs the MedCPT cross-encoder (use_reranker=True).")
        self.retriever = retriever
        self.k = passages_per_candidate
        self.pool = pool

    def passages_for(self, candidate: str):
        """The candidate's own textbook passages, best first."""
        # Ask for a few extra in case some are mixed sections we will drop.
        evidence = self.retriever.search(candidate, k=self.k + 4, pool=self.pool, mode="hybrid",
                                         rerank=True, dedupe=True, expand=False)
        kept = []
        for item in evidence:
            title = item.snippet.title.lower()
            if any(section in title for section in MIXED_SECTIONS):
                continue
            kept.append(item)
        return kept[:self.k]

    def score_question(self, question: str, options: Dict[str, str]) -> Dict[str, dict]:
        """letter -> feature dict for one question."""
        findings = split_case(question)
        case_text = " ".join(findings)
        letters = sorted(options)

        # --- 1. retrieve passages for every candidate ---
        passages = {letter: self.passages_for(options[letter]) for letter in letters}

        # --- 2. build ALL cross-encoder pairs of this question, score them in one batch ---
        pairs = []
        for letter in letters:
            for item in passages[letter]:
                pairs.append((NULL_QUERY, item.snippet.text))
                pairs.append((case_text, item.snippet.text))
                for finding in findings:
                    pairs.append((finding, item.snippet.text))
        with self.retriever._lock:
            scores = list(self.retriever.reranker.score(pairs)) if pairs else []

        # --- 3. unpack the flat score list in the same order it was built ---
        features = {}
        position = 0
        for letter in letters:
            items = passages[letter]
            if not items:
                # The library has nothing on this candidate: the combiner will rely on the
                # votes alone (the "no_passages" flag lets it learn how much to discount).
                features[letter] = {"fit_findings": 0.0, "fit_case": 0.0, "coverage": 0.0,
                                    "no_passages": 1.0, "per_finding": [0.0] * len(findings),
                                    "passage_ids": []}
                continue

            null_scores, case_scores = [], []
            finding_scores = np.zeros((len(items), len(findings)))
            for p, _item in enumerate(items):
                null_scores.append(scores[position])
                case_scores.append(scores[position + 1])
                finding_scores[p] = scores[position + 2: position + 2 + len(findings)]
                position += 2 + len(findings)

            null_scores = np.array(null_scores)
            # Contrast: how much better the passage fits THIS patient than an empty patient.
            case_contrast = np.array(case_scores) - null_scores
            finding_contrast = finding_scores - null_scores[:, None]

            best_per_finding = finding_contrast.max(axis=0)     # one number per finding
            features[letter] = {
                "fit_findings": float(best_per_finding.mean()),
                "fit_case": float(case_contrast.max()),
                # How well retrieval found the candidate at all (cross-encoder score of
                # candidate-as-query vs its best passage). Low = the book barely knows it.
                "coverage": float(items[0].score),
                "no_passages": 0.0,
                "per_finding": [round(float(x), 3) for x in best_per_finding],
                "passage_ids": [item.snippet.id for item in items],
            }
        return features


# ----------------------------------------------------------------------------------------
# Part 2: the combiner (prior x likelihood, with learned weights)
# ----------------------------------------------------------------------------------------

# The columns the combiner sees, per candidate. Because a softmax over one question's
# candidates ignores anything that is the same for all of them, only DIFFERENCES between
# candidates matter - which is exactly a likelihood RATIO in Bayes terms.
FEATURE_SETS = {
    "votes":            ["log_prior"],
    "book_findings":    ["fit_findings"],
    "book_case":        ["fit_case"],
    "votes+findings":   ["log_prior", "fit_findings"],
    "votes+case":       ["log_prior", "fit_case"],
    # agreement x fit: lets the book matter MORE when the votes are split
    "votes+findings+x": ["log_prior", "fit_findings", "split_x_fit"],
    "full":             ["log_prior", "fit_findings", "fit_case", "split_x_fit", "coverage", "no_passages"],
}


def build_matrix(rows: List[dict], columns: Sequence[str]):
    """rows (one per question, with 'votes' and 'book') -> list of (X, gold index) per question.

    log_prior    log of the smoothed vote share (the Bayesian prior)
    split_x_fit  (1 - agreement) * fit_findings: zero when the votes are unanimous
    """
    questions = []
    for row in rows:
        letters = sorted(row["book"])
        shares = vote_shares(row["votes"], letters)
        valid_votes = {k: v for k, v in row["votes"].items() if k in letters}
        agreement = max(valid_votes.values()) / sum(valid_votes.values()) if valid_votes else 0.0

        X = []
        for letter in letters:
            f = dict(row["book"][letter])
            f["log_prior"] = math.log(shares[letter])
            f["split_x_fit"] = (1.0 - agreement) * f["fit_findings"]
            X.append([f[c] for c in columns])
        gold = letters.index(row["gold"]) if row["gold"] in letters else -1
        questions.append((np.array(X, dtype=np.float32), gold, letters))
    return questions


def fit_conditional_logit(questions, l2: float = 0.01, hidden: int = 0, epochs: int = 300, seed: int = 0):
    """Fits score(candidate) = w . x  (or a tiny MLP if hidden > 0), softmax per question.

    hidden = 0  -> linear: the weights ARE the Bayes trust levels, readable in a slide.
    hidden > 0  -> MLP ablation: can learn interactions we did not write by hand.
    Returns a function mapping an (n_candidates x n_features) matrix to scores.
    """
    import torch

    torch.manual_seed(seed)
    data = [(torch.tensor(X), gold) for X, gold, _ in questions if gold >= 0]
    n_features = data[0][0].shape[1]

    # Standardise columns so the L2 penalty treats them equally (scale kept for prediction).
    stacked = torch.cat([X for X, _ in data])
    mean, std = stacked.mean(0), stacked.std(0)
    # A column that never varies in the tuning data (e.g. "no passages found" is always 0)
    # carries no information. Dividing by its ~0 spread would blow it up on a test question
    # where it suddenly is 1, so such columns are switched off (divided by infinity -> 0).
    std = torch.where(std < 1e-6, torch.full_like(std, float("inf")), std)

    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(n_features, hidden), torch.nn.Tanh(),
                                    torch.nn.Linear(hidden, 1))
    else:
        model = torch.nn.Linear(n_features, 1, bias=False)   # a bias cancels in the softmax

    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = 0.0
        for X, gold in data:
            scores = model((X - mean) / std).squeeze(-1)
            loss = loss - torch.log_softmax(scores, dim=0)[gold]
        loss = loss / len(data)
        for parameter in model.parameters():
            loss = loss + l2 * (parameter ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model((torch.tensor(X) - mean) / std).squeeze(-1).numpy()

    # Report linear weights in original units: w / std (how much 1 unit of the feature counts).
    predict.weights = None
    if not hidden:
        predict.weights = (model.weight.detach()[0] / std).numpy().tolist()
    return predict


def predict_letters(predict, questions) -> List[Optional[str]]:
    letters_out = []
    for X, _gold, letters in questions:
        scores = predict(X)
        letters_out.append(letters[int(np.argmax(scores))])
    return letters_out


def cross_validated_accuracy(questions, folds: int = 5, **fit_args) -> float:
    """k-fold accuracy on the TUNING set only, used to choose a feature set honestly."""
    order = np.random.RandomState(0).permutation(len(questions))
    correct = 0
    for fold in range(folds):
        test_idx = set(order[fold::folds].tolist())
        train = [q for i, q in enumerate(questions) if i not in test_idx]
        held = [q for i, q in enumerate(questions) if i in test_idx]
        predict = fit_conditional_logit(train, **fit_args)
        for (X, gold, _letters) in held:
            correct += int(np.argmax(predict(X)) == gold)
    return correct / len(questions)


def majority_letter(votes: Dict[str, int]) -> Optional[str]:
    """Same tie rule as the cot_sc baseline (ties go to the later letter).

    Unparsed votes are saved under the key "null" (JSON's spelling of Python's None);
    they are not a candidate and must never win the majority."""
    counts = Counter({k: v for k, v in votes.items() if k not in ("None", "null")})
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], k))
