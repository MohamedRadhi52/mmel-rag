from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import OPENAI_EMBEDDING_MODEL
from .indexing import FILTERED_CHUNKS, looks_like_toc


# Utilitaires : déduplication et normalisation CS/AMC

def dedupe_by_page(docs: Sequence[Document]) -> List[Document]:
    # Enlève les doublons de pages (même source + même page)
    seen = set()
    out: List[Document] = []

    for d in docs:
        if not isinstance(d, Document):
            d = Document(page_content=str(d), metadata={})

        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        key = (src, str(page))

        if key not in seen:
            seen.add(key)
            out.append(d)

    return out


_cs_amc_pattern = re.compile(
    r"\b(?P<prefix>CS|AMC)\s*-?\s*25\s*\.?\s*(?P<section>\d+[A-Za-z0-9\-]*)\b"
)


def normalize_cs_refs(text: str) -> str:
    # Normalise les références CS/AMC (ex : "CS 25.1309")
    def _repl(match: re.Match) -> str:
        prefix = match.group("prefix")
        section = match.group("section")
        return f"{prefix}-25.{section}"

    return _cs_amc_pattern.sub(_repl, text)


# BM25 sur les chunks filtrés (sans traduction)

_bm25_retriever: BM25Retriever | None = None


def bm25() -> BM25Retriever:
    # Construit le retriever BM25 une seule fois
    global _bm25_retriever

    if _bm25_retriever is None:
        base_docs: List[Document] = [
            d for d in FILTERED_CHUNKS if not looks_like_toc(d.page_content)
        ]
        _bm25_retriever = BM25Retriever.from_documents(base_docs)

    return _bm25_retriever


# Vector store FAISS + retriever MMR

emb = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# On indexe tout le corpus filtré
vs = FAISS.from_documents(FILTERED_CHUNKS, emb)

retriever_vec = vs.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "lambda_mult": 0.7,
        "fetch_k": 80,
    },
)


# Fusion RRF pour le mode hybride

def rrf_fuse(
    pools: Sequence[Sequence[Document]],
    k: int = 5,
    k_rrf: int = 60,
) -> List[Document]:
    # Fusionne plusieurs listes de documents avec Reciprocal Rank Fusion
    scores: Dict[Tuple[str, int], float] = defaultdict(float)
    by_key: Dict[Tuple[str, int], Document] = {}

    for pool in pools:
        for rank, d in enumerate(pool, start=1):
            if not isinstance(d, Document):
                d = Document(page_content=str(d), metadata={})

            src = d.metadata.get("source", "?")
            page_raw = d.metadata.get("page", -1)

            try:
                page = int(page_raw)
            except (TypeError, ValueError):
                page = -1

            key = (src, page)
            scores[key] += 1.0 / (k_rrf + rank)

            if key not in by_key:
                by_key[key] = d

    # Tri par score décroissant
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    merged: List[Document] = [by_key[key] for key, _ in ordered]

    # Filtre les tables des matières
    merged = [d for d in merged if not looks_like_toc(d.page_content)]

    # Déduplication par page
    merged = dedupe_by_page(merged)

    return merged[:k]


# API principale

def get_docs(query: str, mode: str = "vector", k: int = 5) -> List[Document]:
    # Point d'entrée pour récupérer les documents
    q = query.strip()
    if not q:
        return []

    # Mode vectoriel (FAISS)
    if mode == "vector":
        docs = retriever_vec.get_relevant_documents(q)
        docs = [d for d in docs if not looks_like_toc(d.page_content)]
        docs = dedupe_by_page(docs)
        return docs[:k]

    # Mode BM25 classique (sans traduction)
    if mode == "bm25":
        r = bm25()
        docs = r.get_relevant_documents(q)
        docs = [d for d in docs if not looks_like_toc(d.page_content)]
        docs = dedupe_by_page(docs)
        return docs[:k]

    # Mode hybride : fusion vector + BM25
    if mode == "hybrid":
        vec_docs = retriever_vec.get_relevant_documents(q)
        bm_docs = bm25().get_relevant_documents(q)
        return rrf_fuse([vec_docs, bm_docs], k=k)

    raise ValueError("mode doit être 'vector', 'bm25' ou 'hybrid'")


__all__ = [
    "bm25",
    "emb",
    "vs",
    "retriever_vec",
    "get_docs",
    "dedupe_by_page",
    "normalize_cs_refs",
]
