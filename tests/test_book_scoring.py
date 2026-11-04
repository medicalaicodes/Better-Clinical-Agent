import random

import numpy as np

from src.clinical_agent.rag import book_scoring as bs
from src.clinical_agent.retrieval.search import word_set


class OverlapReranker:
    """Stands in for the MedCPT cross-encoder: score = number of shared words."""

    def score(self, pairs):
        return np.array([len(word_set(q) & word_set(p)) for q, p in pairs], dtype=np.float32)


def test_split_case_drops_the_question_sentence():
    case = ("A 30-year-old woman has sudden dyspnea after a long flight. Her D-dimer is raised. "
            "Which of the following is the most likely diagnosis?")
    assert bs.split_case(case) == ["A 30-year-old woman has sudden dyspnea after a long flight.",
                                   "Her D-dimer is raised."]
    assert bs.split_case("Which nerve supplies the deltoid?") == ["Which nerve supplies the deltoid?"]


def test_question_type():
    assert bs.question_type("... Which of the following is the most likely diagnosis?") == "diagnosis"
    assert bs.question_type("... What is the most appropriate next step in management?") == "management"


def test_book_scorer_prefers_the_candidate_whose_textbook_fits(toy_retriever):
    toy_retriever.reranker = OverlapReranker()
    scorer = bs.BookScorer(toy_retriever, passages_per_candidate=2)
    case = ("A woman has absent breath sounds on the left. There is hyperresonance on the affected side. "
            "Which of the following is the most likely diagnosis?")
    features = scorer.score_question(case, {"A": "Pulmonary embolism", "B": "Pneumothorax"})
    assert features["B"]["fit_findings"] > features["A"]["fit_findings"]
    assert features["B"]["passage_ids"][0] == "PTX_0"
    assert len(features["B"]["per_finding"]) == 2


def test_combiner_learns_to_trust_the_book_when_it_is_informative():
    # Synthetic world: votes are right 60% of the time, the book points at the gold answer.
    rng = random.Random(0)
    rows = []
    for _ in range(200):
        gold = rng.choice("ABCD")
        wrong = rng.choice([x for x in "ABCD" if x != gold])
        top = gold if rng.random() < 0.6 else wrong
        votes = {top: 3, rng.choice("ABCD"): 2}
        book = {x: {"fit_findings": (2.0 if x == gold else 0.0) + rng.gauss(0, 0.5), "fit_case": 0.0,
                    "coverage": 1.0, "no_passages": 0.0} for x in "ABCD"}
        rows.append({"gold": gold, "votes": votes, "book": book})

    questions = bs.build_matrix(rows, bs.FEATURE_SETS["votes+findings"])
    predict = bs.fit_conditional_logit(questions)
    assert predict.weights[1] > 0                                   # trusts the book
    accuracy = np.mean([l == r["gold"] for l, r in zip(bs.predict_letters(predict, questions), rows)])
    votes_only = np.mean([bs.majority_letter(r["votes"]) == r["gold"] for r in rows])
    assert accuracy > votes_only
