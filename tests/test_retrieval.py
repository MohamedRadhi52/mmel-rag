from langchain_core.documents import Document
from mmel_rag.retrieval import dedupe_by_page
from mmel_rag.retrieval import normalize_cs_refs

def test_dedupe_by_page_keeps_first_occurrence():
    docs = [
        Document(page_content="A", metadata={"source": "cs25.pdf", "page": 10}),
        Document(page_content="B", metadata={"source": "cs25.pdf", "page": 10}),
        Document(page_content="C", metadata={"source": "cs25.pdf", "page": 11}),
    ]

    deduped = dedupe_by_page(docs)

    # On attend 2 pages uniques: 10 et 11
    assert len(deduped) == 2
    assert deduped[0].page_content == "A"
    assert deduped[1].metadata["page"] == 11


def test_normalize_cs_refs_basic_patterns():
    text = "Voir CS-25.1309(a) et AMC 25.1309 pour plus de détails."
    normalized = normalize_cs_refs(text)
    # Adaptation à ton implémentation réelle, mais l'idée :
    assert "CS-25.1309" in normalized
    assert "AMC 25.1309" in normalized
