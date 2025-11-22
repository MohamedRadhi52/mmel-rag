from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .retrieval import get_docs


# Construit un contexte texte à partir des documents récupérés
# (source + page + extrait), tronqué à max_chars caractères
def _build_context(docs, max_chars: int = 4000):
    chunks = []
    total = 0

    for d in docs:
        header = f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}] "
        text = (d.page_content or "").replace("\n", " ").strip()
        snippet = header + text

        # Si on dépasse max_chars, on coupe le dernier snippet et on s'arrête
        if total + len(snippet) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                chunks.append(snippet[:remaining])
            break

        chunks.append(snippet)
        total += len(snippet)

    return "\n".join(chunks)


# Pipeline Q/A :
# 1) récupère les meilleurs passages avec get_docs(...)
# 2) construit un contexte concaténé
# 3) appelle un modèle ChatOpenAI pour formuler la réponse
def answer_question(
    question: str,
    mode: str = "hybrid",
    k: int = 8,
    model: str | None = None,
) :
    # 1) Récupération des documents pertinents
    docs = get_docs(question, mode=mode, k=k)

    if not docs:
        return (
            "Je n'ai trouvé aucun passage pertinent dans les documents pour "
            "répondre à cette question."
        )

    # 2) Construction du contexte pour le LLM
    context = _build_context(docs)

    # 3) Appel du modèle OpenAI
    qa_model = model or os.getenv("MMEL_RAG_QA_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=qa_model, temperature=0)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant helping with questions about aviation regulations "
                "(CS-25, MMEL, AMC, EU-748/2012, etc.). "
                "Answer using ONLY the provided context. "
                "If the context is insufficient, say that you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context}",
        },
    ]

    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))
