from __future__ import annotations

import logging
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_raw_docs():
    # Charge tous les PDFs du dossier DOCS_DIR (une page = un Document)
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    
    if not pdf_paths:
        raise FileNotFoundError(f"Aucun fichier PDF trouvé dans {DOCS_DIR.resolve()}")

    raw_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        raw_docs.extend(loader.load())

    logger.info(
        "Chargé %d documents depuis %d fichiers PDF",
        len(raw_docs),
        len(pdf_paths)
    )
    
    if raw_docs:
        logger.debug(
            "Premier document: source=%s, page=%s",
            raw_docs[0].metadata.get("source"),
            raw_docs[0].metadata.get("page")
        )
    
    return raw_docs


def clean_text(text) :
     # Quelques nettoyages PDF basiques : espaces, césures, ligatures
    if not text:
        return ""
    
    # Normalise les espaces
    text = re.sub(r"\s+", " ", text)
    
    # Corrige les mots coupés sur plusieurs lignes
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    
    # Remplace les ligatures courantes
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    
    return text.strip()


def apply_cleaning(docs) :
    # Applique le nettoyage de texte à tous les documents (modification en place)
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs


def create_splitter() :
    # Crée le splitter de texte avec la configuration du projet
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


SPLITTER = create_splitter()


def split_into_chunks(docs) :
    # Découpe les documents en chunks plus petits pour améliorer la granularité du retrieval
    chunks = SPLITTER.split_documents(docs)
    logger.info("Découpé %d documents en %d chunks", len(docs), len(chunks))
    return chunks


def looks_like_toc(text) :
    # Supprime des chunks les pages détectées comme tables des matières.
    if not text:
        return False
    
    normalized = re.sub(r"\s+", " ", text).strip()
    
    # Vérifie les mots-clés explicites de TOC
    if re.search(r"\b(table of contents|contents|sommaire)\b", normalized, re.IGNORECASE):
        return True
    
    # Vérifie les points de conduite (courants dans les TOC)
    dot_leaders = re.findall(r"\.{3,}\s*\d{1,4}\b", text)
    if len(dot_leaders) >= 5:
        return True
    
    # Vérifie la présence de nombreuses lignes courtes (structure TOC typique)
    short_lines = [
        line for line in text.splitlines()
        if 0 < len(line.strip()) <= 60
    ]
    if len(short_lines) >= 15:
        return True
    
    return False


def filter_toc(chunks) :
    # Supprime les chunks qui semblent être des tables des matières
    filtered = [doc for doc in chunks if not looks_like_toc(doc.page_content)]
    
    removed_count = len(chunks) - len(filtered)
    logger.info(
        "Filtré %d chunks TOC, conservé %d/%d chunks",
        removed_count,
        len(filtered),
        len(chunks)
    )
    
    return filtered


def prepare_filtered_chunks():
    # Pipeline complet de traitement des documents :
    # Chargement PDFs , Nettoyage texte , Découpage chunks et Filtrage TOC
    logger.info("Démarrage du pipeline de traitement des documents")
    
    raw_docs = load_raw_docs()
    raw_docs = apply_cleaning(raw_docs)
    chunks = split_into_chunks(raw_docs)
    filtered = filter_toc(chunks)
    
    logger.info("Traitement des documents terminé: %d chunks prêts", len(filtered))
    return filtered


# Initialise les chunks filtrés au moment de l'import du module
# Permet un démarrage plus rapide pour les opérations de retrieval
FILTERED_CHUNKS = prepare_filtered_chunks()