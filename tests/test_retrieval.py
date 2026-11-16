from src.clinical_agent.retrieval import reciprocal_rank_fusion
from src.clinical_agent.retrieval.search import is_near_duplicate


def test_rrf_rewards_documents_ranked_well_by_both():
    sparse = [(1, 12.0), (2, 9.0), (3, 1.0)]
    dense = [(2, 0.91), (4, 0.80), (1, 0.10)]
    fused = [row for row, _ in reciprocal_rank_fusion([sparse, dense])]
    assert fused[:2] == [2, 1]           # in both lists beats top of one list
    assert set(fused) == {1, 2, 3, 4}


def test_near_duplicate_detection():
    a = "D-dimer is useful to exclude pulmonary embolism when pretest probability is low."
    b = "D-dimer is useful to exclude pulmonary embolism when the pretest probability is low."
    assert is_near_duplicate(a, [b])
    assert not is_near_duplicate(a, ["Pneumothorax causes absent breath sounds."])


def test_single_query_search_finds_the_right_snippet(toy_retriever):
    hits = toy_retriever.search("absent breath sounds hyperresonance", k=2, expand=False)
    assert hits[0].snippet.id == "PTX_0"


def test_quota_gives_every_query_its_share_of_a_small_budget(toy_retriever):
    queries = [
        "pulmonary embolism CT pulmonary angiography D-dimer",
        "pulmonary embolism D-dimer pretest probability",
        "pneumothorax breath sounds",
    ]
    evidence = toy_retriever.search_many(queries, budget=3, per_query=1, expand=False)
    assert len(evidence) == 3
    # Two strong embolism queries can't crowd out the third one: its snippet is still there.
    assert "PTX_0" in [item.snippet.id for item in evidence]


def test_neighbour_expansion_stays_inside_the_section(toy_retriever):
    row = toy_retriever.row_of_id["PE_1"]
    text = toy_retriever.expand(row)
    assert "CT pulmonary angiography" in text        # previous paragraph, same section
    assert "chest radiograph" in text                 # next paragraph, same section
    row = toy_retriever.row_of_id["PPCM_0"]
    assert "Pneumothorax" not in toy_retriever.expand(row)   # different article: not added


def test_statpearls_chunker_on_a_small_article(tmp_path):
    from src.clinical_agent.retrieval.corpus import chunk_statpearls_article

    article = tmp_path / "article-1.nxml"
    article.write_text(
        "<book-part><title>Pulmonary Embolism</title>"
        "<sec><title>Evaluation</title>"
        "<p>" + "Pretest probability guides testing. " * 10 + "</p>"
        "<p>Short follow-up sentence.</p>"                      # < 200 chars: glued on
        "<p><bold>Imaging</bold></p>"                            # a subtitle
        "<p>" + "CT pulmonary angiography is the test of choice. " * 6 + "</p>"
        "</sec></book-part>"
    )
    snippets = chunk_statpearls_article(str(article))

    assert [s["id"] for s in snippets] == ["article-1_0", "article-1_1"]
    assert snippets[0]["title"] == "Pulmonary Embolism -- Evaluation"
    assert snippets[0]["content"].endswith("Short follow-up sentence.")
    assert snippets[1]["title"] == "Pulmonary Embolism -- Evaluation -- Imaging"
