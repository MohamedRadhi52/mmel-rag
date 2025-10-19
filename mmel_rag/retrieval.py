from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import OPENAI_EMBEDDING_MODEL, TRANSLATION_MODEL
from .indexing import FILTERED_CHUNKS, looks_like_toc


# Fonctions utilitaires : déduplication et normalisation des références CS/AMC

# Supprime les doublons de pages (même source/page), garde le premier chunk
def dedupe_by_page(docs) :
    seen = set()
    out: List[Document] = []
    for d in docs:
        if not isinstance(d, Document):
            d = Document(page_content=str(d), metadata={})
        key = (d.metadata.get("source", "?"), str(d.metadata.get("page", "?")))
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out


# Normalise les références CS/AMC dans le texte (CS-25.x, AMC 25.x, etc.)
def normalize_cs_refs(q) :
    q = re.sub(r"\bCS\s*[- ]?\s*25\s*[\.-]\s*(\d+)", r"CS-25.\1", q, flags=re.I)
    q = re.sub(r"\bAMC\s*[- ]?\s*25\s*[\.-]\s*(\d+)", r"AMC 25.\1", q, flags=re.I)
    return q


# BM25 : tokenisation, traduction EN et légère expansion de termes

# Tokenise un texte en mots minuscules, en enlevant la ponctuation
def tokenize(s):
    return re.findall(r"\w+", (s or "").lower())


bm25 = BM25Retriever.from_documents(
    FILTERED_CHUNKS,
    preprocess_func=tokenize,
)

bm25.k = 5

translator = ChatOpenAI(model=TRANSLATION_MODEL, temperature=0)


# Traduit la requête utilisateur en anglais (pour la recherche BM25)
def translate_to_en(q):
    msgs = [
        {
            "role": "system",
            "content": (
                "Translate the user's query into precise, technical English for search "
                "in aviation regulations. Output only the translation."
            ),
        },
        {"role": "user", "content": q},
    ]
    
    return translator.invoke(msgs).content.strip()
    
    


# Petites expansions FR -> EN pour aider BM25 sur certains termes
EXPAND = {
    "frein": ["brake", "braking"],
    "sortie d'urgence": ["emergency exit", "egress"],
    "train d'atterrissage": ["landing gear", "undercarriage"],
    "stabilité": ["stability", "longitudinal stability"],
    "dégivrage": ["de-icing", "anti-ice", "icing"],
    "évacuation": ["egress", "evacuation"],
    "commandes de vol": ["flight controls", "control surfaces"],
    "givrage": ["icing", "anti-ice", "de-icing"],
}


# Ajoute une expansion de termes (synonymes / reformulations) pour BM25
def expand_terms(q_en) :
    q_low = q_en.lower()
    extra: List[str] = []
    for fr, en_list in EXPAND.items():
        if fr in q_low:
            extra += en_list
    if not extra:
        return q_en
    return q_en + " " + " ".join(sorted(set(extra)))


# Force un objet compatible en véritable Document LangChain
def as_doc(x) :
    if isinstance(x, Document):
        return x
    if isinstance(x, str):
        return Document(page_content=x, metadata={})
    try:
        _ = x.page_content  # type: ignore[attr-defined]
        return x  # type: ignore[return-value]
    except Exception:
        return Document(page_content=str(x), metadata={})


# Construit un identifiant stable pour un document (source + page + début de texte)
def doc_id(d):
    d = as_doc(d)
    return (
        d.metadata.get("source", "?"),
        str(d.metadata.get("page", "?")),
        d.page_content[:160],
    )


# Fusionne plusieurs listes de documents avec la méthode Reciprocal Rank Fusion
def rrf_fuse(
    pools: Sequence[Sequence[Document]],
    k: int = 5,
    k_rrf: int = 20,
) :
    scores: dict = defaultdict(float)
    id2doc: dict = {}
    for pool in pools:
        for rank, d in enumerate(pool, start=1):
            d = as_doc(d)
            did = doc_id(d)
            scores[did] += 1.0 / (k_rrf + rank)
            id2doc.setdefault(did, d)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [id2doc[i] for i, _ in fused][:k]


# Pipeline BM25 : traduction FR→EN + expansion + fusion RRF, puis top-k final
def retrieve_bm25_translate(query: str, k: int = 5) :
    fetch_k = max(k, 12)
    old_k = bm25.k

    bm25.k = fetch_k
    q_en = expand_terms(translate_to_en(query))
    docs_orig = bm25.invoke(query) or []
    docs_en = bm25.invoke(q_en) or []
    
    docs_orig = [
        as_doc(d)
        for d in docs_orig
        if not looks_like_toc(as_doc(d).page_content)
    ]
    docs_en = [
        as_doc(d)
        for d in docs_en
        if not looks_like_toc(as_doc(d).page_content)
    ]
    return rrf_fuse([docs_orig[:fetch_k], docs_en[:fetch_k]], k=k)


# Vector store (FAISS + embeddings OpenAI)
emb = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vs = FAISS.from_documents(FILTERED_CHUNKS, embedding=emb)

retriever_vec = vs.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,          # top-k final
        "fetch_k": 80,   # pool avant MMR (↑ rappel)
        "lambda_mult": 0.7,  # favorise la similarité vs diversité
    },
)

print("Vector retriever MMR initialisé (lambda=0.7, fetch_k=80)")


# Routeur de requêtes vers les différents retrievers (vector, BM25, hybride)
def get_docs(query: str, mode: str = "vector", k: int = 5) :
    mode = (mode or "").lower().strip()
    query = normalize_cs_refs(query)

    # Force un résultat en liste de Documents
    def coerce_list(lst):
        return [
            d if isinstance(d, Document) else Document(page_content=str(d), metadata={})
            for d in (lst or [])
        ]

    if mode == "vector":
        cand = coerce_list(retriever_vec.invoke(query))
        cand = [d for d in cand if not looks_like_toc(d.page_content)]
        cand = dedupe_by_page(cand)
        return cand[:k]

    elif mode == "bm25translate":
        cand = coerce_list(retrieve_bm25_translate(query, k=max(k, 12)))
        cand = dedupe_by_page(cand)
        return cand[:k]

    elif mode == "hybrid":
        # v = résultats vectoriels, b = résultats BM25 traduits
        v = coerce_list(get_docs(query, "vector", k=max(k, 8)))
        b = coerce_list(get_docs(query, "bm25translate", k=max(k, 8)))

        # Identifiant stable pour la fusion RRF au niveau routeur
        def did(d: Document):
            return (
                d.metadata.get("source", "?"),
                str(d.metadata.get("page", "?")),
                d.page_content[:160],
            )

        scores: dict = defaultdict(float)
        id2doc: dict = {}
        K_RRF = 20
        for pool in (v, b):
            for r, d in enumerate(pool, 1):
                _id = did(d)
                scores[_id] += 1.0 / (K_RRF + r)
                id2doc.setdefault(_id, d)

        fused = [id2doc[i] for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        fused = [d for d in fused if not looks_like_toc(d.page_content)]
        fused = dedupe_by_page(fused)
        return fused[:k]

    else:
        raise ValueError("mode doit être 'vector', 'bm25translate' ou 'hybrid'")


__all__ = [
    "bm25",
    "emb",
    "vs",
    "retriever_vec",
    "retrieve_bm25_translate",
    "get_docs",
    "dedupe_by_page",
    "normalize_cs_refs",
]
