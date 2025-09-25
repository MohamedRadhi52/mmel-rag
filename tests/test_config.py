# Classe de test pour verifier la presence des clés et des documents necessaires
import os
from pathlib import Path

from mmel_rag.config import PROJECT_ROOT, DOCS_DIR, DEVSET_CSV


REQUIRED_KEYS = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY"]


# Vérifie la présence des clés
def check_env_keys():
    for k in REQUIRED_KEYS:
        v = os.getenv(k)
        assert v, f"Variable d'environnement manquante : {k}"


def check_docs_and_devset():
    # vérifier que DOCS_DIR existe
    assert DOCS_DIR.exists(), f"Dossier docs introuvable : {DOCS_DIR}"

    # vérifier qu'il contient exactement deux PDFs
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    assert len(pdfs) == 2, (
        f"On attend 2 PDF dans {DOCS_DIR}, trouvé {len(pdfs)} "
    )

    # vérifier que DEVSET_CSV existe
    assert DEVSET_CSV.exists(), f"Fichier dev set introuvable : {DEVSET_CSV}"

    # vérifier qu'il y a exactement un CSV dans DOCS_DIR
    csvs = list(DOCS_DIR.glob("*.csv"))
    assert len(csvs) == 1, (
        f"On attend 1 CSV dans {DOCS_DIR}, trouvé {len(csvs)}  "
    )


def main():
    check_env_keys()
    check_docs_and_devset()

if __name__ == "__main__":
    main()
