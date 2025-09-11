from __future__ import annotations

import argparse

from .qa import answer_question
from .evaluation import run_full_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG réglementaire (port du notebook rag_regulatory.ipynb)."
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Question à poser au système RAG (si omis, lance seulement l'évaluation si --eval).",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["vector", "bm25translate", "hybrid"],
        help="Mode de retrieval à utiliser pour la Q/R (par défaut : hybrid).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Nombre de documents à récupérer pour la Q/R (k du retriever).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Lance l'évaluation retrieval (DEVSET) exactement comme dans le notebook.",
    )

    args = parser.parse_args()

    if args.eval:
        run_full_eval()

    if args.question:
        answer = answer_question(args.question, mode=args.mode, k=args.k)
        print("\n=== RÉPONSE ===\n")
        print(answer)


if __name__ == "__main__":
    main()
