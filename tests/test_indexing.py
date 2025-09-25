from mmel_rag.indexing import looks_like_toc

def test_looks_like_toc_detects_classic_toc():
    toc_text = """
    SOMMAIRE
    1 Généralités............................... 1
    1.1 Objet................................... 2
    1.2 Domaine d'application.................. 3
    """
    assert looks_like_toc(toc_text) is True


def test_looks_like_toc_not_triggered_on_normal_page():
    normal_page = """
    1.2 Applicability

    This paragraph describes the applicability of CS-25.1309(a).
    The objective is to ensure that failures do not lead to catastrophic events.
    """
    assert looks_like_toc(normal_page) is False
